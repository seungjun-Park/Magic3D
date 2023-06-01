from diffusers import StableDiffusionPipeline

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd
from dataclasses import dataclass


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


@dataclass
class UNet2DConditionOutput:
    sample: torch.HalfTensor  # Not sure how to check what unet_traced.pt contains, and user wants. HalfTensor or FloatTensor


class DiffusionWrapper(nn.Module):
    def __init__(self,
                 device,
                 grad_scale=1.0,
                 fp16=False,
                 model_id="stabilityai/stable-diffusion-2-1-base",
                 t_range=[0.02, 0.98],
                 vram=False,
                 variance_preserving=False
                 ):
        super().__init__()

        self.device = device

        self.precision = torch.float16 if fp16 else torch.float32
        self.grad_scale = grad_scale

        # Create model
        self.model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=self.precision)
        self.t_range = t_range
        self.variance_preserving = variance_preserving

        if vram:
            self.model.enable_sequential_cpu_offload()
            self.model.enable_vae_slicing()
            self.model.unet.to(memory_format=torch.channels_last)
            self.model.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            self.model.to(self.device)

        self.model.eval()

        self.vae = self.model.vae
        self.tokenizer = self.model.tokenizer
        self.text_encoder = self.model.text_encoder
        self.unet = self.model.unet

        self.scheduler = self.model.schedular
        self.num_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_timesteps * t_range[0])
        self.max_step = int(self.num_timesteps * t_range[1]) + 1
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        print(f'[INFO] loaded {model_id} stable diffusion model.')

    def forward(self,
                pred_rgb: torch.Tensor,
                prompt: Union[str, List[str]] = None,
                height: Optional[int] = None,
                width: Optional[int] = None,
                guidance_scale: float = 7.5,
                negative_prompt: Optional[Union[str, List[str]]] = None,
                num_images_per_prompt: Optional[int] = 1,
                prompt_embeds: Optional[torch.FloatTensor] = None,
                negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                return_dict: bool = True,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        ):

        outputs = dict()

        do_clssifier_free_guidance = guidance_scale > 1.0

        prompt_embedding = self.model._encode_prompt(
            prompt,
            self.device,
            num_images_per_prompt,
            do_clssifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds
        )

        # interp to 512x512 to be fed into vae.
        pred_rgb_resize = F.interpolate(pred_rgb, (height, width), mode='bilinear', align_corners=False)
        outputs.update({'pred_rgb': pred_rgb_resize})

        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(pred_rgb_resize)
        outputs.update({'latents': latents})

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step, (latents.shape[0], ), device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            outputs.update({'latenst_noisy': latents_noisy})
            latent_model_input = torch.cat([latents_noisy] * 2) if do_clssifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embedding,
                cross_attention_kwargs=cross_attention_kwargs
            ).sample

        # perform guidance (high scale from paper!)
        if do_clssifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        outputs.update({'noise_pred': noise_pred})

        # w(t), sigma_t^2
        w = (1 - self.alphas_cumprod[t]) if self.variance_preserving else (1 - self.alphas_cumprod[t]) * torch.sqrt((1 - self.alphas_cumprod[t]) ** 2)
        difference = noise_pred - noise
        outputs.update({'difference': difference})

        loss = self.grad_scale * w * difference
        loss = torch.nan_to_num(loss)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        # loss = SpecifyGradient.apply(latents, grad)

        if return_dict:
            return loss, outputs

        else:
            return loss

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1 #?? why mul 2?

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents