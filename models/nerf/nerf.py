import os
import math
import cv2
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdiffrast.torch as dr

import mcubes
import raymarching
from meshutils import decimate_mesh, clean_mesh, poisson_mesh_reconstruction
from models.utils import custom_meshgrid, safe_normalize
from modules import ResBlock, MLP, BasicBlock
from encoding import get_encoder
from activation import trunc_exp, biased_softplus
from utils import instantiate_from_config

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


@torch.cuda.amp.autocast(enabled=False)
def near_far_from_bound(rays_o, rays_d, bound, type='cube', min_near=0.05):
    # rays: [B, N, 3], [B, N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [B, N, 1], far [B, N, 1]

    radius = rays_o.norm(dim=-1, keepdim=True)

    if type == 'sphere':
        near = radius - bound  # [B, N, 1]
        far = radius + bound

    elif type == 'cube':
        tmin = (-bound - rays_o) / (rays_d + 1e-15)  # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection, set both near and far to inf (1e9 here)
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        # restrict near to a minimal value
        near = torch.clamp(near, min=min_near)

    return near, far

def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


def compute_edge_to_face_mapping(attr_idx):
    with torch.no_grad():
        # Get unique edges
        # Create all edges, packed by triangle
        all_edges = torch.cat((
            torch.stack((attr_idx[:, 0], attr_idx[:, 1]), dim=-1),
            torch.stack((attr_idx[:, 1], attr_idx[:, 2]), dim=-1),
            torch.stack((attr_idx[:, 2], attr_idx[:, 0]), dim=-1),
        ), dim=-1).view(-1, 2)

        # Swap edge order so min index is always first
        order = (all_edges[:, 0] > all_edges[:, 1]).long().unsqueeze(dim=1)
        sorted_edges = torch.cat((
            torch.gather(all_edges, 1, order),
            torch.gather(all_edges, 1, 1 - order)
        ), dim=-1)

        # Elliminate duplicates and return inverse mapping
        unique_edges, idx_map = torch.unique(sorted_edges, dim=0, return_inverse=True)

        tris = torch.arange(attr_idx.shape[0]).repeat_interleave(3).cuda()

        tris_per_edge = torch.zeros((unique_edges.shape[0], 2), dtype=torch.int64).cuda()

        # Compute edge to face table
        mask0 = order[:, 0] == 0
        mask1 = order[:, 0] == 1
        tris_per_edge[idx_map[mask0], 0] = tris[mask0]
        tris_per_edge[idx_map[mask1], 1] = tris[mask1]

        return tris_per_edge


@torch.cuda.amp.autocast(enabled=False)
def normal_consistency(face_normals, t_pos_idx):
    tris_per_edge = compute_edge_to_face_mapping(t_pos_idx)

    # Fetch normals for both faces sharind an edge
    n0 = face_normals[tris_per_edge[:, 0], :]
    n1 = face_normals[tris_per_edge[:, 1], :]

    # Compute error metric based on normal difference
    term = torch.clamp(torch.sum(n0 * n1, -1, keepdim=True), min=-1.0, max=1.0)
    term = (1.0 - term)

    return torch.mean(torch.abs(term))


def laplacian_uniform(verts, faces):
    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device=verts.device, dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()


@torch.cuda.amp.autocast(enabled=False)
def laplacian_smooth_loss(verts, faces):
    with torch.no_grad():
        L = laplacian_uniform(verts, faces.long())
    loss = L.mm(verts)
    loss = loss.norm(dim=1)
    loss = loss.mean()
    return loss


class InstantNGP(nn.Module):
    def __init__(self,
                 encoder_config,
                 encoder_bg_config=None,
                 bound=1,
                 dmtet=False,
                 min_near=0.01,
                 density_thresh=10,
                 tet_grid_size=128,
                 #nerf
                 num_layers=3,
                 hidden_dim=64,
                 num_layers_bg=2,
                 hidden_dim_bg=32,
                 density_activation='exp',
                 bg_radius=0.2,
                 blob_density=0,
                 blob_radius=0,
                 num_steps=64,
                 upsample_steps=32,
                 lambda_orient=1e-2,
                 lambda_2d_normal_smooth=0,
                 lambda_normal=0,
                 lambda_mesh_laplacian=0.5,
                 lambda_mesh_normal=0.5,
                 ):
        super().__init__()

        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.max_level = None

        self.min_near = min_near
        self.density_thresh = density_thresh

        self.num_steps = num_steps
        self.upsample_steps = upsample_steps
        self.lambda_orient = lambda_orient
        self.lambda_2d_normal_smooth = lambda_2d_normal_smooth
        self.lambda_normal = lambda_normal
        self.lambda_mesh_laplacian = lambda_mesh_laplacian
        self.lambda_mesh_normal = lambda_mesh_normal

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        self.glctx = None

        # nerf
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.density_activation = density_activation
        self.bg_radius = bg_radius
        self.blob_density = blob_density
        self.blob_radius = blob_radius

        self.encoder = instantiate_from_config(encoder_config)
        self.in_dim = self.encoder.output_dim

        self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True)
        # self.normal_net = MLP(self.in_dim, 3, hidden_dim, num_layers, bias=True)

        self.density_activation = trunc_exp if self.density_activation == 'exp' else biased_softplus

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg

            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg = instantiate_from_config(encoder_bg_config)
            self.in_dim_bg = self.encoder_bg.output_dim
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)

        else:
            self.bg_net = None

    @torch.no_grad()
    def density_blob(self, x):
        # x: [B, N, 3]

        d = (x ** 2).sum(-1)

        if self.density_activation == 'exp':
            g = self.blob_density * torch.exp(- d / (2 * self.blob_radius ** 2))
        else:
            g = self.blob_density * (1 - torch.sqrt(d) / self.blob_radius)

        return g

    def common_forward(self, x):
        enc = self.encoder(x, bound=self.bound, max_level=self.max_level)

        h = self.sigma_net(enc)

        sigma = self.density_activation(h[..., 0] + self.density_blob(x))
        albedo = torch.sigmoid(h[..., 1:])

        return sigma, albedo

    def forward(self, rays_o, rays_d, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, T_thresh=1e-4, binarize=False, **kwargs):
        # rays_o, rays_d: [B, N, 3]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d,
                                                     self.aabb_train if self.training else self.aabb_infer)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = safe_normalize(rays_o + torch.randn(3, device=rays_o.device))  # [N, 3]

        results = {}

        if self.training:
            xyzs, dirs, ts, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield,
                                                                self.cascade, self.grid_size, nears, fars, perturb,
                                                                self.opt.dt_gamma, self.opt.max_steps)
            dirs = safe_normalize(dirs)

            if light_d.shape[0] > 1:
                flatten_rays = raymarching.flatten_rays(rays, xyzs.shape[0]).long()
                light_d = light_d[flatten_rays]

            # sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
            sigmas, albedo = self.common_forward(xyzs)

            if shading == 'albedo':
                normals = None
                rgbs = albedo

            else:  # lambertian shading

                # normal = self.normal_net(enc)
                normals = self.normals(xyzs)

                lambertian = ambient_ratio + (1 - ambient_ratio) * (normals * light_d).sum(-1).clamp(min=0)  # [N,]

                if shading == 'textureless':
                    rgbs = lambertian.unsqueeze(-1).repeat(1, 3)
                elif shading == 'normal':
                    rgbs = (normals + 1) / 2
                else:  # 'lambertian'
                    rgbs = albedo * lambertian.unsqueeze(-1)

            weights, weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, ts, rays, T_thresh,
                                                                                  binarize)

            # normals related regularizations
            if self.opt.lambda_orient > 0 and normals is not None:
                # orientation loss
                loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.mean()

            if self.opt.lambda_3d_normal_smooth > 0 and normals is not None:
                normals_perturb = self.normal(xyzs + torch.randn_like(xyzs) * 1e-2)
                results['loss_normal_perturb'] = (normals - normals_perturb).abs().mean()

            if (self.opt.lambda_2d_normal_smooth > 0 or self.opt.lambda_normal > 0) and normals is not None:
                _, _, _, normal_image = raymarching.composite_rays_train(sigmas.detach(), (normals + 1) / 2, ts, rays,
                                                                         T_thresh, binarize)
                results['normal_image'] = normal_image

            # weights normalization
            results['weights'] = weights

        else:
            # allocate outputs
            dtype = torch.float32

            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)

            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device)  # [N]
            rays_t = nears.clone()  # [N]

            step = 0

            while step < self.opt.max_steps:  # hard coded max step

                # count alive rays
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, ts = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound,
                                                        self.density_bitfield, self.cascade, self.grid_size, nears,
                                                        fars, perturb if step == 0 else False, self.opt.dt_gamma,
                                                        self.opt.max_steps)
                dirs = safe_normalize(dirs)
                #sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
                sigmas, albedo = self.common_forward(xyzs)

                if shading == 'albedo':
                    normals = None
                    rgbs = albedo

                else:  # lambertian shading

                    # normal = self.normal_net(enc)
                    normals = self.normals(xyzs)

                    lambertian = ambient_ratio + (1 - ambient_ratio) * (normals * light_d).sum(-1).clamp(min=0)  # [N,]

                    if shading == 'textureless':
                        rgbs = lambertian.unsqueeze(-1).repeat(1, 3)
                    elif shading == 'normal':
                        rgbs = (normals + 1) / 2
                    else:  # 'lambertian'
                        rgbs = albedo * lambertian.unsqueeze(-1)

                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum, depth,
                                           image, T_thresh, binarize)

                rays_alive = rays_alive[rays_alive >= 0]
                # print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step

        # mix background color
        if bg_color is None:
            if self.opt.bg_radius > 0:
                # use the bg model to calculate bg_color
                bg_color = self.background(rays_d)  # [N, 3]
            else:
                bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, 3)

        depth = depth.view(*prefix)

        weights_sum = weights_sum.reshape(*prefix)

        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum

        return results

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        sigma, albedo = self.common_forward(x)

        return {
            'sigma': sigma,
            'albedo': albedo,
        }

    @torch.no_grad()
    def export_mesh(self, path, resolution=None, decimate_target=-1, S=128):

        if self.dmtet:

            sdf = self.sdf
            deform = torch.tanh(self.deform) / self.tet_grid_size

            vertices, triangles = self.dmtet(self.verts + deform, sdf, self.indices)

            vertices = vertices.detach().cpu().numpy()
            triangles = triangles.detach().cpu().numpy()

        else:

            if resolution is None:
                resolution = self.grid_size

            if self.cuda_ray:
                density_thresh = min(self.mean_density, self.density_thresh) \
                    if np.greater(self.mean_density, 0) else self.density_thresh
            else:
                density_thresh = self.density_thresh

            # TODO: use a larger thresh to extract a surface mesh from the density field, but this value is very empirical...
            if self.density_activation == 'softplus':
                density_thresh = density_thresh * 25

            sigmas = np.zeros([resolution, resolution, resolution], dtype=np.float32)

            # query
            X = torch.linspace(-1, 1, resolution).split(S)
            Y = torch.linspace(-1, 1, resolution).split(S)
            Z = torch.linspace(-1, 1, resolution).split(S)

            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [S, 3]
                        val = self.density(pts.to(self.aabb_train.device))
                        sigmas[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val[
                            'sigma'].reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()  # [S, 1] --> [x, y, z]

            print(f'[INFO] marching cubes thresh: {density_thresh} ({sigmas.min()} ~ {sigmas.max()})')

            vertices, triangles = mcubes.marching_cubes(sigmas, density_thresh)
            vertices = vertices / (resolution - 1.0) * 2 - 1

        # clean
        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)
        vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.01)

        # decimation
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

        v = torch.from_numpy(vertices).contiguous().float().to(self.aabb_train.device)
        f = torch.from_numpy(triangles).contiguous().int().to(self.aabb_train.device)

        # mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        # mesh.export(os.path.join(path, f'mesh.ply'))

        def _export(v, f, h0=2048, w0=2048, ssaa=1, name=''):
            # v, f: torch Tensor
            device = v.device
            v_np = v.cpu().numpy()  # [N, 3]
            f_np = f.cpu().numpy()  # [M, 3]

            print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

            # unwrap uvs
            import xatlas
            import nvdiffrast.torch as dr
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4  # for faster unwrap...
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

            # render uv maps
            uv = vt * 2.0 - 1.0  # uvs to range [-1, 1]
            uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)  # [N, 4]

            if ssaa > 1:
                h = int(h0 * ssaa)
                w = int(w0 * ssaa)
            else:
                h, w = h0, w0

            if self.glctx is None:
                if h <= 2048 and w <= 2048:
                    self.glctx = dr.RasterizeCudaContext()
                else:
                    self.glctx = dr.RasterizeGLContext()

            rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), ft, (h, w))  # [1, h, w, 4]
            xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)  # [1, h, w, 3]
            mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f)  # [1, h, w, 1]

            # masked query
            xyzs = xyzs.view(-1, 3)
            mask = (mask > 0).view(-1)

            feats = torch.zeros(h * w, 3, device=device, dtype=torch.float32)

            if mask.any():
                xyzs = xyzs[mask]  # [M, 3]

                # batched inference to avoid OOM
                all_feats = []
                head = 0
                while head < xyzs.shape[0]:
                    tail = min(head + 640000, xyzs.shape[0])
                    results_ = self.density(xyzs[head:tail])
                    all_feats.append(results_['albedo'].float())
                    head += 640000

                feats[mask] = torch.cat(all_feats, dim=0)

            feats = feats.view(h, w, -1)
            mask = mask.view(h, w)

            # quantize [0.0, 1.0] to [0, 255]
            feats = feats.cpu().numpy()
            feats = (feats * 255).astype(np.uint8)

            ### NN search as an antialiasing ...
            mask = mask.cpu().numpy()

            inpaint_region = binary_dilation(mask, iterations=3)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=2)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
            _, indices = knn.kneighbors(inpaint_coords)

            feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]

            feats = cv2.cvtColor(feats, cv2.COLOR_RGB2BGR)

            # do ssaa after the NN search, in numpy
            if ssaa > 1:
                feats = cv2.resize(feats, (w0, h0), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(os.path.join(path, f'{name}albedo.png'), feats)

            # save obj (v, vt, f /)
            obj_file = os.path.join(path, f'{name}mesh.obj')
            mtl_file = os.path.join(path, f'{name}mesh.mtl')

            print(f'[INFO] writing obj mesh to {obj_file}')
            with open(obj_file, "w") as fp:
                fp.write(f'mtllib {name}mesh.mtl \n')

                print(f'[INFO] writing vertices {v_np.shape}')
                for v in v_np:
                    fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

                print(f'[INFO] writing vertices texture coords {vt_np.shape}')
                for v in vt_np:
                    fp.write(f'vt {v[0]} {1 - v[1]} \n')

                print(f'[INFO] writing faces {f_np.shape}')
                fp.write(f'usemtl mat0 \n')
                for i in range(len(f_np)):
                    fp.write(
                        f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

            with open(mtl_file, "w") as fp:
                fp.write(f'newmtl mat0 \n')
                fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
                fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
                fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
                fp.write(f'Tr 1.000000 \n')
                fp.write(f'illum 1 \n')
                fp.write(f'Ns 0.000000 \n')
                fp.write(f'map_Kd {name}albedo.png \n')

        _export(v, f)


    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward(
            (x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward(
            (x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward(
            (x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward(
            (x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward(
            (x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward(
            (x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))

        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon,
            0.5 * (dy_pos - dy_neg) / epsilon,
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal

    def background(self, d):

        h = self.encoder_bg(d)  # [N, C]

        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        ### update density grid
        tmp_grid = - torch.ones_like(self.density_grid)

        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:

                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                                       dim=-1)  # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long()  # [N]
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1  # [N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_xyzs = xyzs * (bound - half_grid_size)
                        # add noise in [-hgs, hgs]
                        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                        # query density
                        sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                        # assign
                        tmp_grid[cas, indices] = sigmas
        # ema update
        valid_mask = self.density_grid >= 0
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid[valid_mask]).item()
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

    def reset_extra_state(self):
        if not (self.cuda_ray or self.taichi_ray):
            return
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0

