import cv2, imageio, os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage

from utils import instantiate_from_config
from datasets.base import NeRFDataset


class Magic3D(pl.LightningModule):
    def __init__(self,
                 nerf_config,
                 diffusion_config,
                 prompt,
                 negative=None,
                 ema_decay=0.95,  # if use EMA, set the decay
                 fp16=False,  # amp optimize level
                 lr=1e-3,
                 update_extra_interval=16,
                 log_interval=100,
                 guidance_scale=100,
                 extract_mesh=True,
                 progressive_view=True,
                 progressive_level=True,
                 known_view_scale=1.5,
                 known_view_noise_scale=2e-3,
                 stage='first',
                 lambda_orient=1e-2,
                 lambda_2d_normal_smooth=0,
                 lambda_normal=0,
                 lambda_mesh_laplacian=0.5,
                 lambda_mesh_normal=0.5,
                 lambda_entropy=1e-3
                 ):
        super().__init__()
        assert stage in ['first', 'second']
        self.stage = stage

        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.lr = lr
        self.guidance_scale = guidance_scale
        self.extract_mesh = extract_mesh

        self.progressive_view = progressive_view
        self.progressive_level = progressive_level

        self.known_view_cale = known_view_scale
        self.known_view_noise_scale = known_view_noise_scale

        self.update_extra_interval = update_extra_interval
        self.log_interval = log_interval

        self.nerf = instantiate_from_config(nerf_config)
        self.diffusion = instantiate_from_config(diffusion_config)

        self.prompt = prompt
        self.negative = negative
        self.direction_prompt = ['front', 'side', 'back']

        self.lambda_orient = lambda_orient
        self.lambda_2d_normal_smooth = lambda_2d_normal_smooth
        self.lambda_normal = lambda_normal
        self.lambda_mesh_laplacian = lambda_mesh_laplacian
        self.lambda_mesh_normal = lambda_mesh_normal
        self.lamda_entropy = lambda_entropy

        # text prompt
        self.diffusion.eval()
        for p in self.diffusion.parameters():
            p.requires_grad = False

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.nerf.parameters(), decay=ema_decay)
        else:
            self.ema = None

    def forward(self, data):
        # progressively relaxing view range

        losses = dict()
        outputs = dict()

        # current progress of training ratio
        progressive_ratio = self.global_step / (self.trainer.max_epochs * self.trainer.num_training_batches)

        if self.progressive_view:
            self.trainer.train_dataloader.dataset.datasets.progressive_update(progressive_ratio)

        # progressively increase max_level
        if self.progressive_level:
            self.nerf.max_level = min(1.0, 0.25 + progressive_ratio)

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if self.stage == 'first':
            ambient_ratio = 1.0
            shading = 'normal'
            as_latent = True
            binarize = False
            bg_color = None

        else:
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

        model_results = self.nerf(rays_o, rays_d, perturb=True,
                                   bg_color=bg_color,
                                    ambient_ratio=ambient_ratio,
                                   shading=shading,
                                   binarize=binarize)

        pred_depth = model_results['depth'].reshape(B, 1, H, W)
        outputs.update({'depth': pred_depth})
        pred_mask = model_results['weights_sum'].reshape(B, 1, H, W)
        outputs.update({'weights_sum': pred_mask})
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)

        if as_latent:
            # abuse normal & mask as latent code for faster geometry initialization (ref: fantasia3D)
            pred_rgb = torch.cat([outputs['image'], outputs['weights_sum'].unsqueeze(-1)], dim=-1).reshape(B, H, W,4).permute(0,3,1,2).contiguous()  # [B, 4, H, W]
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]

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
        if self.stage == 'first':
            if self.lambda_opacity > 0:
                loss_opacity = (outputs['weights_sum'] ** 2).mean()
                loss = loss + self.opt.lambda_opacity * loss_opacity

            if self.lambda_entropy > 0:
                alphas = outputs['weights'].clamp(1e-5, 1 - 1e-5)
                # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
                loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
                lambda_entropy = self.opt.lambda_entropy * min(1, 2 * self.global_step / self.opt.iters)
                loss = loss + lambda_entropy * loss_entropy

            if self.lambda_2d_normal_smooth > 0 and 'normal_image' in outputs:
                # pred_vals = outputs['normal_image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
                # smoothed_vals = TF.gaussian_blur(pred_vals.detach(), kernel_size=9)
                # loss_smooth = F.mse_loss(pred_vals, smoothed_vals)
                # total-variation
                loss_smooth = (pred_normal[:, 1:, :, :] - pred_normal[:, :-1, :, :]).square().mean() + \
                              (pred_normal[:, :, 1:, :] - pred_normal[:, :, :-1, :]).square().mean()
                loss = loss + self.lambda_2d_normal_smooth * loss_smooth

            if self.lambda_orient > 0 and 'loss_orient' in outputs:
                loss_orient = outputs['loss_orient']
                loss = loss + self.opt.lambda_orient * loss_orient

        else:

            if self.lambda_mesh_normal > 0:
                loss = loss + self.opt.lambda_mesh_normal * outputs['normal_loss']

            if self.lambda_mesh_laplacian > 0:
                loss = loss + self.opt.lambda_mesh_laplacian * outputs['lap_loss']

        return loss, losses, outputs

    def training_step(self, batch, batch_idx):
        with torch.cuda.amp.autocast(enabled=self.fp16):
            if self.global_step % self.update_extra_interval == 0:
                self.nerf.update_extra_state()
            loss, losses, outputs = self(batch)

        if self.global_step % self.log_interval == 0:
            self.log_images(outputs)

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

    def log_images(self, outputs):
        tb = self.logger.experiement
        for key, val in outputs:
            tb.add_image(f'side/{key}', val, self.global_step)

    def configure_optimizers(self):
        params = list(self.nerf.parameters())
        opt = torch.optim.AdamW(params, self.lr)
        return opt



