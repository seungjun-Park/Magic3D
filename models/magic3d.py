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
                 nerf_config,
                 diffusion_config,
                 prompt,
                 negative=None,
                 ema_decay=None,  # if use EMA, set the decay
                 fp16=False,  # amp optimize level
                 lr=1e-3,
                 log_interval=100,
                 max_epoch=100,
                 guidance_scale=100,
                 extract_mesh=True,
                 progressive_view=True,
                 progressive_view_init_ratio=0.2,
                 progressive_level=True,
                 known_view_scale=1.5,
                 known_view_noise_scale=2e-3,
                 ):
        super().__init__()
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.lr = lr
        self.guidance_scale = guidance_scale
        self.extract_mesh = extract_mesh

        self.progressive_view = progressive_view
        self.progressive_view_init_ratio = progressive_view_init_ratio
        self.progressive_level = progressive_level

        self.known_view_cale = known_view_scale
        self.known_view_noise_scale = known_view_noise_scale

        self.log_interval = log_interval

        self.nerf = instantiate_from_config(nerf_config)
        self.diffusion = instantiate_from_config(diffusion_config)

        self.prompt = prompt
        self.negative = negative
        self.direction_prompt = ['front', 'side', 'back']

        # text prompt
        self.diffusion.eval()
        for p in self.diffusion.parameters():
            p.requires_grad = False

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

    def forward(self, data):
        # progressively relaxing view range
        if self.progressive_view:
            r = min(1.0, 0.2 + self.global_step / (0.5 * self.trainer.max_epochs * self.trainer.num_training_batches))
            self.opt.phi_range = [self.opt.default_phi * (1 - r) + self.opt.full_phi_range[0] * r,
                                  self.opt.default_phi * (1 - r) + self.opt.full_phi_range[1] * r]
            self.opt.theta_range = [self.opt.default_theta * (1 - r) + self.opt.full_theta_range[0] * r,
                                    self.opt.default_theta * (1 - r) + self.opt.full_theta_range[1] * r]
            self.opt.radius_range = [self.opt.default_radius * (1 - r) + self.opt.full_radius_range[0] * r,
                                     self.opt.default_radius * (1 - r) + self.opt.full_radius_range[1] * r]
            self.opt.fovy_range = [self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[0] * r,
                                   self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[1] * r]

        # progressively increase max_level
        if self.progressive_level:
            # total iters = total epoch * total batches
            self.nerf.max_level = min(1.0, 0.25 + self.global_step / (0.5 * self.trainer.max_epochs * self.trainer.num_training_batches))

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

        return loss

    def training_step(self, batch, batch_idx):
        with torch.cuda.amp.autocast(enabled=self.fp16):
            loss, outputs = self(batch)

        if self.global_step % self.log_interval == 0:
            tb = self.logger.experiment


        return loss

    def configure_gradient_clipping(
        self,
        optimizer,
        optimizer_idx,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):
        self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)

        # if self.lambda_tv > 0:
        #     lambda_tv = min(1.0, self.global_step / (0.5 * self.opt.iters)) * self.lambda_tv
        #     self.nerf.encoder.grad_total_variation(lambda_tv, None, self.nerf.bound)
        # if self.lambda_wd > 0:
        #     self.nerf.encoder.grad_weight_decay(self.lambda_wd)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.update()


