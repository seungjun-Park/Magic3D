import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


from utils import instantiate_from_config

class magic3d(pl.LightningModule):
    def __init__(self,
                 nerf_config,
                 diffusion_config,
                 lr=1e-4,
                 ):
        super().__init__()

        self.nerf = instantiate_from_config(nerf_config)
        self.diffusion = instantiate_from_config(diffusion_config)

        self.epoch = 0
        self.learning_rate = lr

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def forward(self, data):
        # perform RGBD loss instead of SDS if is image-conditioned
        do_rgbd_loss = self.opt.image is not None and \
                       (self.global_step % self.opt.known_view_interval == 0)

        # override random camera with fixed known camera
        if do_rgbd_loss:
            data = self.default_view_data

        # progressively relaxing view range
        if self.opt.progressive_view:
            r = min(1.0, 0.2 + self.global_step / (0.5 * self.opt.iters))
            self.opt.phi_range = [self.opt.default_phi * (1 - r) + self.opt.full_phi_range[0] * r,
                                  self.opt.default_phi * (1 - r) + self.opt.full_phi_range[1] * r]
            self.opt.theta_range = [self.opt.default_theta * (1 - r) + self.opt.full_theta_range[0] * r,
                                    self.opt.default_theta * (1 - r) + self.opt.full_theta_range[1] * r]
            self.opt.radius_range = [self.opt.default_radius * (1 - r) + self.opt.full_radius_range[0] * r,
                                     self.opt.default_radius * (1 - r) + self.opt.full_radius_range[1] * r]
            self.opt.fovy_range = [self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[0] * r,
                                   self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[1] * r]

        # progressively increase max_level
        if self.opt.progressive_level:
            self.model.max_level = min(1.0, 0.25 + self.global_step / (0.5 * self.opt.iters))

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        mvp = data['mvp']  # [B, 4, 4]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if do_rgbd_loss:
            ambient_ratio = 1.0
            shading = 'ambient'
            as_latent = False
            binarize = False
            bg_color = torch.rand((B * N, 3), device=rays_o.device)

            # add camera noise to avoid grid-like artifect
            if self.opt.known_view_noise_scale > 0:
                noise_scale = self.opt.known_view_noise_scale  # * (1 - self.global_step / self.opt.iters)
                rays_o = rays_o + torch.randn(3, device=self.device) * noise_scale
                rays_d = rays_d + torch.randn(3, device=self.device) * noise_scale

        elif self.global_step < self.opt.warmup_iters:
            ambient_ratio = 1.0
            shading = 'normal'
            as_latent = True
            binarize = False
            bg_color = None
        else:
            # random shading
            ambient_ratio = 0.1 + 0.9 * random.random()
            rand = random.random()
            if rand > 0.8:
                shading = 'textureless'
            else:
                shading = 'lambertian'
            as_latent = False

            # random weights binarization (like mobile-nerf) [NOT WORKING NOW]
            # binarize_thresh = min(0.5, -0.5 + self.global_step / self.opt.iters)
            # binarize = random.random() < binarize_thresh
            binarize = False

            # random background
            rand = random.random()
            if rand > 0.5:
                bg_color = None  # use bg_net
            else:
                bg_color = torch.rand(3).to(self.device)  # single color random bg

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=False, perturb=True, bg_color=bg_color,
                                    ambient_ratio=ambient_ratio, shading=shading, binarize=binarize)
        pred_depth = outputs['depth'].reshape(B, 1, H, W)
        pred_mask = outputs['weights_sum'].reshape(B, 1, H, W)
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)

        if as_latent:
            # abuse normal & mask as latent code for faster geometry initialization (ref: fantasia3D)
            pred_rgb = torch.cat([outputs['image'], outputs['weights_sum'].unsqueeze(-1)], dim=-1).reshape(B, H, W,
                                                                                                           4).permute(0,
                                                                                                                      3,
                                                                                                                      1,
                                                                                                                      2).contiguous()  # [1, 4, H, W]
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()  # [1, 3, H, W]

        # known view loss
        if do_rgbd_loss:
            gt_mask = self.mask  # [H, W]
            gt_rgb = self.rgb  # [3, H, W]

            # color loss
            gt_rgb = gt_rgb * gt_mask.float() + bg_color.reshape(H, W, 3).permute(2, 0, 1).contiguous() * (
                        1 - gt_mask.float())
            loss = self.opt.lambda_rgb * F.mse_loss(pred_rgb, gt_rgb)

            # mask loss
            loss = loss + self.opt.lambda_mask * F.mse_loss(pred_mask[0, 0], gt_mask.float())

            # normal loss
            if self.opt.lambda_normal > 0 and 'normal_image' in outputs:
                valid_gt_normal = 1 - 2 * self.normal[gt_mask]  # [B, 3]
                valid_pred_normal = 2 * pred_normal.squeeze()[gt_mask] - 1  # [B, 3]

                lambda_normal = self.opt.lambda_normal * min(1, self.global_step / self.opt.iters)
                loss = loss + lambda_normal * (1 - F.cosine_similarity(valid_pred_normal, valid_gt_normal).mean())

            # relative depth loss
            if self.opt.lambda_depth > 0:
                valid_gt_depth = self.depth[gt_mask].unsqueeze(1)  # [B, 1]
                valid_pred_depth = pred_depth.squeeze()[gt_mask].unsqueeze(1)  # [B, 1]
                # scale-invariant
                with torch.no_grad():
                    A = torch.cat([valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1)  # [B, 2]
                    X = torch.linalg.lstsq(A, valid_pred_depth).solution  # [2, 1]
                    valid_gt_depth = A @ X  # [B, 1]
                lambda_depth = self.opt.lambda_depth * min(1, self.global_step / self.opt.iters)
                loss = loss + lambda_depth * F.mse_loss(valid_pred_depth, valid_gt_depth)

        # novel view loss
        else:

            if self.opt.guidance == 'stable-diffusion':
                # interpolate text_z
                azimuth = data['azimuth']  # [-180, 180]

                if azimuth >= -90 and azimuth < 90:
                    if azimuth >= 0:
                        r = 1 - azimuth / 90
                    else:
                        r = 1 + azimuth / 90
                    start_z = self.text_z['front']
                    end_z = self.text_z['side']
                else:
                    if azimuth >= 0:
                        r = 1 - (azimuth - 90) / 90
                    else:
                        r = 1 + (azimuth + 90) / 90
                    start_z = self.text_z['side']
                    end_z = self.text_z['back']

                pos_z = r * start_z + (1 - r) * end_z
                uncond_z = self.text_z['uncond']
                text_z = torch.cat([uncond_z, pos_z], dim=0)
                loss = self.guidance.train_step(text_z, pred_rgb, as_latent=as_latent,
                                                guidance_scale=self.opt.guidance_scale,
                                                grad_scale=self.opt.lambda_guidance)

            else:  # zero123
                polar = data['polar']
                azimuth = data['azimuth']
                radius = data['radius']

                # adjust SDS scale based on how far the novel view is from the known view
                lambda_guidance = (abs(azimuth) / 180) * self.opt.lambda_guidance

                loss = self.guidance.train_step(self.image_z, pred_rgb, polar, azimuth, radius, as_latent=as_latent,
                                                guidance_scale=self.opt.guidance_scale, grad_scale=lambda_guidance)

        # regularizations
        if not self.opt.dmtet:

            if self.opt.lambda_opacity > 0:
                loss_opacity = (outputs['weights_sum'] ** 2).mean()
                loss = loss + self.opt.lambda_opacity * loss_opacity

            if self.opt.lambda_entropy > 0:
                alphas = outputs['weights'].clamp(1e-5, 1 - 1e-5)
                # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
                loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
                lambda_entropy = self.opt.lambda_entropy * min(1, 2 * self.global_step / self.opt.iters)
                loss = loss + lambda_entropy * loss_entropy

            if self.opt.lambda_2d_normal_smooth > 0 and 'normal_image' in outputs:
                # pred_vals = outputs['normal_image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
                # smoothed_vals = TF.gaussian_blur(pred_vals.detach(), kernel_size=9)
                # loss_smooth = F.mse_loss(pred_vals, smoothed_vals)
                # total-variation
                loss_smooth = (pred_normal[:, 1:, :, :] - pred_normal[:, :-1, :, :]).square().mean() + \
                              (pred_normal[:, :, 1:, :] - pred_normal[:, :, :-1, :]).square().mean()
                loss = loss + self.opt.lambda_2d_normal_smooth * loss_smooth

            if self.opt.lambda_orient > 0 and 'loss_orient' in outputs:
                loss_orient = outputs['loss_orient']
                loss = loss + self.opt.lambda_orient * loss_orient
        else:

            if self.opt.lambda_mesh_normal > 0:
                loss = loss + self.opt.lambda_mesh_normal * outputs['normal_loss']

            if self.opt.lambda_mesh_laplacian > 0:
                loss = loss + self.opt.lambda_mesh_laplacian * outputs['lap_loss']

        return pred_rgb, pred_depth, loss
    def training_step(self, batch, batch_idx):
        loss, loss_dict, output = self(batch)

        self.log_dict(loss_dict)

        if self.global_step % self.log_interval == 0:
            tb = self.logger.experiement


        return loss

    def validation_step(self, batch, batch_idx):
        return

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.nerf.parameters())
        opt = torch.optim.AdamW(params, lr=lr)

        return opt