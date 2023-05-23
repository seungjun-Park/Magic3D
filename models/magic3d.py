import cv2, imageio, os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage


from utils import instantiate_from_config

class Magic3D(pl.LightningModule):
    def __init__(self,
                 name,  # name of this experiment
                 nerf_config,
                 diffusion_config,
                 text,
                 negative,
                 O,
                 O2,
                 guidance_scale,
                 save_mesh,
                 mcubes_resolution=256,
                 decimate_target=5e4,
                 ema_decay=None,  # if use EMA, set the decay
                 fp16=False,  # amp optimize level
                 eval_interval=1,  # eval once every $ epoch
                 max_keep_ckpt=2,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 ):
        super().__init__()
        self.name = name
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval

        self.nerf = instantiate_from_config(nerf_config)
        self.diffusion = instantiate_from_config(diffusion_config)

        self.text = text
        self.negative = negative

        # text prompt
        self.diffusion.eval()
        for p in self.diffusion.parameters():
            p.requires_grad = False
        self.prepare_embeddings()

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        self.epoch = 0

    # calculate the text embs.
    def prepare_embeddings(self, text, negative=None):

        # text embeddings (stable-diffusion)
        if self.opt.text is not None:

            self.text_z = {}

            self.text_z['default'] = self.guidance.get_text_embeds([self.opt.text])
            self.text_z['uncond'] = self.guidance.get_text_embeds([self.opt.negative])

            for d in ['front', 'side', 'back']:
                self.text_z[d] = self.guidance.get_text_embeds([f"{self.opt.text}, {d} view"])
        else:
            self.text_z = None

    def training_step(self, batch, batch_idx):
        # update grid every 16 steps
        if (self.model.cuda_ray or self.model.taichi_ray) and self.global_step % self.opt.update_extra_interval == 0:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state()

        with torch.cuda.amp.autocast(enabled=self.fp16):
            pred_rgbs, pred_depths, loss, variations = self(batch)

        # hooked grad clipping for RGB space
        if self.opt.grad_clip_rgb >= 0:
            def _hook(grad):
                if self.opt.fp16:
                    # correctly handle the scale
                    grad_scale = self.scaler._get_scale_async()
                    return grad.clamp(grad_scale * -self.opt.grad_clip_rgb, grad_scale * self.opt.grad_clip_rgb)
                else:
                    return grad.clamp(-self.opt.grad_clip_rgb, self.opt.grad_clip_rgb)

            pred_rgbs.register_hook(_hook)
            # pred_rgbs.retain_grad()

        self.scaler.scale(loss).backward()

        self.post_train_step()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler_update_every_step:
            self.lr_scheduler.step()

        loss_val = loss.item()

        if self.epoch != self.current_epoch:
            self.epoch = self.current_epoch
            tb = self.logger.experiment
            self.writer.add_scalar("train/loss", loss_val, self.global_step)
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)
            self.writer.add_image('sds grad', variations[0], self.global_step)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.update()

    def forward(self, data):
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

        if self.global_step < self.opt.warmup_iters:
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
        loss, variations = self.guidance.train_step(text_z, pred_rgb, as_latent=as_latent,
                                                    guidance_scale=self.opt.guidance_scale,
                                                    grad_scale=self.opt.lambda_guidance)

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

        return pred_rgb, pred_depth, loss, variations

    def validation_step(self, batch, batch_idx):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0


        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

            self.local_step += 1

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, preds_depth, loss = self.eval_step(data)

            # all_gather/reduce the statistics (NCCL only support all_*)
            if self.world_size > 1:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss = loss / self.world_size

                preds_list = [torch.zeros_like(preds).to(self.device) for _ in
                              range(self.world_size)]  # [[B, ...], [B, ...], ...]
                dist.all_gather(preds_list, preds)
                preds = torch.cat(preds_list, dim=0)

                preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in
                                    range(self.world_size)]  # [[B, ...], [B, ...], ...]
                dist.all_gather(preds_depth_list, preds_depth)
                preds_depth = torch.cat(preds_depth_list, dim=0)

            loss_val = loss.item()
            total_loss += loss_val

            # only rank = 0 will perform evaluation.
            if self.local_rank == 0:
                # save image
                save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                save_path_depth = os.path.join(self.workspace, 'validation',
                                               f'{name}_{self.local_step:04d}_depth.png')

                # self.log(f"==> Saving validation image to {save_path}")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                pred_depth = (pred_depth * 255).astype(np.uint8)

                cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                cv2.imwrite(save_path_depth, pred_depth)

                pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)


        if self.ema is not None:
            self.ema.restore()



    def eval_step(self, data):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=False, bg_color=None,
                                    light_d=light_d, ambient_ratio=ambient_ratio, shading=shading)
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        # dummy
        loss = torch.zeros([1], device=pred_rgb.device, dtype=pred_rgb.dtype)

        return pred_rgb, pred_depth, loss

    def post_train_step(self):

        # unscale grad before modifying it!
        # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        self.scaler.unscale_(self.optimizer)

        # clip grad
        if self.opt.grad_clip >= 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.opt.grad_clip)

        if not self.opt.dmtet and self.opt.backbone == 'grid':

            if self.opt.lambda_tv > 0:
                lambda_tv = min(1.0, self.global_step / (0.5 * self.opt.iters)) * self.opt.lambda_tv
                self.model.encoder.grad_total_variation(lambda_tv, None, self.model.bound)
            if self.opt.lambda_wd > 0:
                self.model.encoder.grad_weight_decay(self.opt.lambda_wd)

    def test_step(self, data, bg_color=None, perturb=False):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=perturb, light_d=light_d,
                                    ambient_ratio=ambient_ratio, shading=shading, bg_color=bg_color)

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        return pred_rgb, pred_depth, None

    def save_mesh(self, loader=None, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(save_path, resolution=self.opt.mcubes_resolution,
                               decimate_target=self.opt.decimate_target)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------


    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, _ = self.test_step(data)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'),
                                cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)

            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8,
                             macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8,
                             macro_block_size=1)

        self.log(f"==> Finished Test.")

