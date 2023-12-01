import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from tqdm.auto import tqdm
import lpips
import time
from .utils import cosine_beta_schedule, extract, noise_like, default, linear_beta_schedule


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        context_fn,
        channels=3,
        num_timesteps=1000,
        loss_type="l1",
        clip_noise="half",
        vbr=False,
        lagrangian=1e-3,
        pred_mode="noise",
        var_schedule="linear",
        aux_loss_weight=0,
        aux_loss_type="l1",
    ):
        super().__init__()
        self.channels = channels
        self.denoise_fn = denoise_fn
        self.context_fn = context_fn
        self.clip_noise = clip_noise
        self.vbr = vbr
        self.otherlogs = {}
        self.loss_type = loss_type
        self.lagrangian_beta = lagrangian
        self.var_schedule = var_schedule
        self.sample_steps = None
        self.aux_loss_weight = aux_loss_weight
        self.aux_loss_type = aux_loss_type
        assert pred_mode in ["noise", "image", "renoise"]
        self.pred_mode = pred_mode
        to_torch = partial(torch.tensor, dtype=torch.float32)
        if aux_loss_weight > 0:
            self.loss_fn_vgg = lpips.LPIPS(net="vgg", eval_mode=False)
        else:
            self.loss_fn_vgg = None

        if var_schedule == "cosine":
            train_betas = cosine_beta_schedule(num_timesteps)
        elif var_schedule == "linear":
            train_betas = linear_beta_schedule(num_timesteps)
        train_alphas = 1.0 - train_betas
        train_alphas_cumprod = np.cumprod(train_alphas, axis=0)
        # train_alphas_cumprod_prev = np.append(1.0, train_alphas_cumprod[:-1])
        (num_timesteps,) = train_betas.shape
        self.num_timesteps = int(num_timesteps)

        self.register_buffer("train_betas", to_torch(train_betas))
        self.register_buffer("train_alphas_cumprod", to_torch(train_alphas_cumprod))
        # self.register_buffer("train_alphas_cumprod_prev", to_torch(train_alphas_cumprod_prev))
        self.register_buffer("train_sqrt_alphas_cumprod", to_torch(np.sqrt(train_alphas_cumprod)))
        self.register_buffer(
            "train_sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - train_alphas_cumprod))
        )
        self.register_buffer(
            "train_sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / train_alphas_cumprod))
        )
        self.register_buffer(
            "train_sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / train_alphas_cumprod - 1))
        )

    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if "loss_fn_vgg" not in name:
                yield param

    def get_extra_loss(self):
        return self.context_fn.get_extra_loss()

    def set_sample_schedule(self, sample_steps, device):
        self.sample_steps = sample_steps
        indice = torch.linspace(0, self.num_timesteps - 1, sample_steps, device=device).long()
        self.alphas_cumprod = self.train_alphas_cumprod[indice]
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod
        self.one_minus_alphas_cumprod_prev = 1.0 - self.alphas_cumprod_prev
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1.0 - self.alphas_cumprod_prev)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod_prev = torch.sqrt(1.0 / self.alphas_cumprod_prev)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.sigma = torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        ) * torch.sqrt(1 - self.alphas_cumprod / self.alphas_cumprod_prev)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_start_from_noise_train(self, x_t, t, noise):
        return (
            extract(self.train_sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.train_sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        return posterior_mean

    def p_mean_variance(self, x, t, context, clip_denoised):
        # noise = self.denoise_fn(x, self.sqrt_alphas_cumprod[t], context=context)
        if self.pred_mode == "noise":
            noise = self.denoise_fn(x, t.float().unsqueeze(-1) / self.sample_steps, context=context)
            x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        else:
            x_recon = self.denoise_fn(
                x, t.float().unsqueeze(-1) / self.sample_steps, context=context
            )

        if clip_denoised == "full":
            x_recon.clamp_(-1.0, 1.0)
        elif clip_denoised == "half":
            x_recon[: x_recon.shape[0] // 2].clamp_(-1.0, 1.0)

        model_mean = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        return model_mean

    def ddim(self, x, t, context, clip_denoised, eta=0):
        noise = self.denoise_fn(x, t.float().unsqueeze(-1) / self.sample_steps, context=context)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        if clip_denoised == "full":
            x_recon.clamp_(-1.0, 1.0)
        elif clip_denoised == "half":
            x_recon[: x_recon.shape[0] // 2].clamp_(-1.0, 1.0)
        x_next = (
            extract(self.sqrt_alphas_cumprod_prev, t, x.shape) * x_recon
            + torch.sqrt(
                extract(self.one_minus_alphas_cumprod_prev, t, x.shape)
                - (eta * extract(self.sigma, t, x.shape)) ** 2
            )
            * noise + eta * extract(self.sigma, t, x.shape) * torch.randn_like(noise)
        )
        return x_next

    @torch.no_grad()
    def p_sample(self, x, t, context, clip_denoised, sample_mode="ddpm", eta=0):
        if sample_mode == "ddpm":
            model_mean = self.p_mean_variance(
                x=x, t=t, context=context, clip_denoised=clip_denoised
            )
            return model_mean
        elif sample_mode == "ddim":
            return self.ddim(x=x, t=t, context=context, clip_denoised=clip_denoised, eta=eta)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def p_sample_loop(self, shape, context, sample_mode, init=None, eta=0):
        device = self.alphas_cumprod.device

        b = shape[0]
        img = torch.zeros(shape, device=device) if init is None else init
        # buffer = []
        for count, i in enumerate(
            tqdm(
                reversed(range(0, self.sample_steps)),
                desc="sampling loop time step",
                total=self.sample_steps,
            )
        ):
            time = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(
                img,
                time,
                context=context,
                clip_denoised=self.clip_noise,
                sample_mode=sample_mode,
                eta=eta,
            )
        #     if count % 50 == 0:
        #         buffer.append(img)
        # buffer.append(img)
        return img

    @torch.no_grad()
    def compress(
        self,
        images,
        sample_steps=None,
        bitrate_scale=None,
        sample_mode="ddpm",
        bpp_return_mean=True,
        init=None,
        eta=0,
    ):
        context_dict = self.context_fn(images, bitrate_scale)
        self.set_sample_schedule(
            self.num_timesteps if (sample_steps is None) else sample_steps,
            context_dict["output"][0].device,
        )
        return (
            self.p_sample_loop(
                images.shape, context_dict["output"], sample_mode, init=init, eta=eta
            ),
            context_dict["bpp"].mean() if bpp_return_mean else context_dict["bpp"]
        )

    def q_sample(self, x_start, t, noise):

        return (
            extract(self.train_sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.train_sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, context_dict, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        fx = self.denoise_fn(
            x_noisy, t.float().unsqueeze(-1) / self.num_timesteps, context=context_dict["output"]
        )
        # x_recon = self.denoise_fn(
        #     x_noisy, self.train_sqrt_alphas_cumprod[t], context=context_dict["output"]
        # )

        if self.pred_mode == "noise":
            if self.loss_type == "l1":
                err = (noise - fx).abs().mean()
            elif self.loss_type == "l2":
                err = F.mse_loss(noise, fx)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError

        aux_err = 0

        if self.aux_loss_weight > 0:
            pred_x0 = self.predict_start_from_noise_train(x_noisy, t, fx).clamp(-1.0, 1.0)
            if self.aux_loss_type == "l1":
                aux_err = F.l1_loss(x_start, pred_x0)
            elif self.aux_loss_type == "l2":
                aux_err = F.mse_loss(x_start, pred_x0)
            elif self.aux_loss_type == "lpips":
                aux_err = self.loss_fn_vgg(x_start, pred_x0).mean()
            else:
                raise NotImplementedError()

            loss = (
                self.lagrangian_beta * context_dict["bpp"].mean()
                + err * (1 - self.aux_loss_weight)
                + aux_err * self.aux_loss_weight
            )
        else:
            loss = self.lagrangian_beta * context_dict["bpp"].mean() + err
        # loss = err.mean()

        return loss

    def forward(self, images):
        device = images.device
        B, C, H, W = images.shape
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        if self.vbr:
            bitrate_scale = torch.rand(size=(B,), device=device)
            self.lagrangian_beta = self.scale_to_beta(bitrate_scale)
        else:
            bitrate_scale = None
        output_dict = self.context_fn(images, bitrate_scale)
        loss = self.p_losses(images, output_dict, t)
        return loss, self.get_extra_loss()

    def scale_to_beta(self, bitrate_scale):
        return 2 ** (3 * bitrate_scale) * 5e-4

