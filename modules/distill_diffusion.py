import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from tqdm.auto import tqdm
import lpips
import time, copy
from .utils import cosine_beta_schedule, extract, noise_like, default, linear_beta_schedule


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        context_fn,
        ae_fn=None,
        num_timesteps=1000,
        loss_type="l1",
        lagrangian=1e-3,
        pred_mode="noise",
        var_schedule="linear",
        aux_loss_weight=0,
        aux_loss_type="l1",
        use_loss_weight=False,
        loss_weight_min=5,
        use_aux_loss_weight_schedule=False
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.context_fn = context_fn
        self.ae_fn = ae_fn
        self.student_fn = copy.deepcopy(denoise_fn)
        self.otherlogs = {}
        self.loss_type = loss_type
        self.lagrangian_beta = lagrangian
        self.var_schedule = var_schedule
        self.sample_steps = None
        self.aux_loss_weight = aux_loss_weight
        self.aux_loss_type = aux_loss_type
        assert pred_mode in ["noise", "x", "v"]
        self.pred_mode = pred_mode
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.use_loss_weight = use_loss_weight
        self.loss_weight_min = float(loss_weight_min)
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

        self.register_buffer("train_snr", to_torch(train_alphas_cumprod / (1 - train_alphas_cumprod)))
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
        for name, param in self.student_fn.named_parameters(recurse=recurse):
            yield param
    
    def copy_params_from_teacher_to_student(self):
        for (_, student_params), (_, teacher_params) in zip(self.student_fn.named_parameters(), self.denoise_fn.named_parameters()):
            student_params.data.copy_(teacher_params.data)

        for (_, student_buffers), (_, teacher_buffers) in zip(self.student_fn.named_buffers(), self.denoise_fn.named_buffers()):
            student_buffers.data.copy_(teacher_buffers.data)
    
    def copy_params_from_student_to_teacher(self):
        for (_, student_params), (_, teacher_params) in zip(self.student_fn.named_parameters(), self.denoise_fn.named_parameters()):
            teacher_params.data.copy_(student_params.data)

        for (_, student_buffers), (_, teacher_buffers) in zip(self.student_fn.named_buffers(), self.denoise_fn.named_buffers()):
            teacher_buffers.data.copy_(student_buffers.data)

    def get_extra_loss(self):
        return self.context_fn.get_extra_loss()

    def set_sample_schedule(self, sample_steps, device):
        self.sample_steps = sample_steps
        indice = torch.linspace(0, self.num_timesteps - 1, sample_steps, device=device).long()
        self.alphas_cumprod = self.train_alphas_cumprod[indice]
        self.snr = self.train_snr[indice]
        self.index = torch.arange(self.num_timesteps, device=device)[indice]
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
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def ddim(self, x, t, context, clip_denoised, use_student=False, eta=0):
        if self.denoise_fn.embd_type == "01":
            if use_student:
                fx = self.student_fn(x, self.index[t].float().unsqueeze(-1) / self.num_timesteps, context=context)
            else:
                fx = self.denoise_fn(x, self.index[t].float().unsqueeze(-1) / self.num_timesteps, context=context)
        else:
            if use_student:
                fx = self.student_fn(x, self.index[t], context=context)
            else:
                fx = self.denoise_fn(x, self.index[t], context=context)
        if self.pred_mode == "noise":
            x_recon = self.predict_start_from_noise(x, t=t, noise=fx)
        elif self.pred_mode == "x":
            x_recon = fx
        elif self.pred_mode == "v":
            x_recon = self.predict_start_from_v(x, t=t, v=fx)
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        noise = fx if self.pred_mode == "noise" else self.predict_noise_from_start(x, t=t, x0=x_recon)
        x_next = (
            extract(self.sqrt_alphas_cumprod_prev, t, x.shape) * x_recon
            + torch.sqrt(
                extract(self.one_minus_alphas_cumprod_prev, t, x.shape)
                - (eta * extract(self.sigma, t, x.shape)) ** 2
            )
            * noise + eta * extract(self.sigma, t, x.shape) * torch.randn_like(noise)
        )
        return x_next

    def p_sample(self, x, t, context, clip_denoised, eta=0):
        return self.ddim(x=x, t=t, context=context, clip_denoised=clip_denoised, use_student=True, eta=eta)

    def p_sample_loop(self, shape, context, clip_denoised=False, init=None, eta=0):
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
                clip_denoised=clip_denoised,
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
        bpp_return_mean=True,
        init=None,
        eta=0,
    ):
        context_dict = self.context_fn(images)
        self.set_sample_schedule(
            self.num_timesteps if (sample_steps is None) else sample_steps,
            context_dict["output"][0].device,
        )
        if self.ae_fn is None:
            return (
                self.p_sample_loop(
                    images.shape, context_dict["output"], clip_denoised=True, init=init, eta=eta
                ),
                context_dict["bpp"].mean() if bpp_return_mean else context_dict["bpp"]
            )
        else:
            z = self.ae_fn.encode(images).mode
            dec_z = self.p_sample_loop(z.shape, context_dict["output"], clip_denoised=False, init=init, eta=eta)
            return self.ae_fn.decode(dec_z), context_dict["bpp"].mean() if bpp_return_mean else context_dict["bpp"]
            
    def q_sample(self, x_start, t, noise):

        return (
            extract(self.train_sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.train_sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def forward(self, images, k=0):
        device = images.device
        B, C, H, W = images.shape
        tidx2train = torch.arange(0, self.num_timesteps, 2**k, device=device).long()
        sidx2train = tidx2train[::2][1:]
        sample = torch.randint(0, sidx2train.shape[0], (B,), device=device).long()
        sampled_sidx2train = sidx2train[sample]
        with torch.no_grad():
            context_dict = self.context_fn(images)
        if self.ae_fn is not None:
            state = self.ae_fn.encode(images).mode
        else:
            state = images
        
        noise = torch.randn_like(state)
        x_noisy = self.q_sample(x_start=state, t=sampled_sidx2train, noise=noise)
        self.set_sample_schedule(len(tidx2train), device)
        if self.denoise_fn.embd_type == "01":
            fx = self.student_fn(
                x_noisy, sampled_sidx2train.float().unsqueeze(-1) / self.num_timesteps, context=context_dict["output"]
            )
            tidx2train_idx = torch.searchsorted(tidx2train, sampled_sidx2train)
            with torch.no_grad():
                x_next = self.ddim(x_noisy, tidx2train_idx, context_dict["output"], clip_denoised=True)
                target = self.denoise_fn(x_next, tidx2train[tidx2train_idx - 1].float().unsqueeze(-1) / self.num_timesteps, context=context_dict["output"]).detach()
        else:
            fx = self.student_fn(
                x_noisy, sampled_sidx2train, context=context_dict["output"]
            )
            tidx2train_idx = torch.searchsorted(tidx2train, sampled_sidx2train)
            with torch.no_grad():
                x_next = self.ddim(x_noisy, tidx2train_idx, context_dict["output"], clip_denoised=True)
                target = self.denoise_fn(x_next, tidx2train[tidx2train_idx - 1], context=context_dict["output"]).detach()

        if self.pred_mode == "noise":
            if self.use_loss_weight:
                if self.loss_weight_min>0:
                    weight = (self.train_snr[sampled_sidx2train].clamp(max=self.loss_weight_min) / self.train_snr[sampled_sidx2train])
                else:
                    weight = (self.train_snr[sampled_sidx2train].clamp(min=-self.loss_weight_min) / self.train_snr[sampled_sidx2train])
            else:
                weight = torch.ones(1, device=device)
            if self.loss_type == "l1":
                err = F.l1_loss(target, fx, reduction='none').mean(dim=(1, 2, 3))
                err = (err * torch.sqrt(weight)).mean()
            elif self.loss_type == "l2":
                err = F.mse_loss(target, fx, reduction='none').mean(dim=(1, 2, 3))
                err = (err * weight).mean()
            else:
                raise NotImplementedError()
        elif self.pred_mode == "x":
            if self.use_loss_weight:
                if self.loss_weight_min>0:
                    weight = self.train_snr[sampled_sidx2train].clamp(max=self.loss_weight_min)
                else:
                    weight = self.train_snr[sampled_sidx2train].clamp(min=-self.loss_weight_min)
            else:
                weight = torch.ones(1, device=device)
            if self.loss_type == "l1":
                err = F.l1_loss(target, fx, reduction='none').mean(dim=(1, 2, 3))
                err = (err * torch.sqrt(weight)).mean()
            elif self.loss_type == "l2":
                err = F.mse_loss(target, fx, reduction='none').mean(dim=(1, 2, 3))
                err = (err * weight).mean()
            else:
                raise NotImplementedError()
        elif self.pred_mode == "v":
            if self.use_loss_weight:
                if self.loss_weight_min>0:
                    weight = self.train_snr[sampled_sidx2train].clamp(max=self.loss_weight_min) / (self.train_snr[sampled_sidx2train] + 1)
                else:
                    weight = self.train_snr[sampled_sidx2train].clamp(min=-self.loss_weight_min) / (self.train_snr[sampled_sidx2train] + 1)
            else:
                weight = torch.ones(1, device=device)
            v = self.predict_v(state, sampled_sidx2train, noise)
            if self.loss_type == "l1":
                err = F.l1_loss(fx, target, reduction='none').mean(dim=(1, 2, 3))
                err = (err * torch.sqrt(weight)).mean()
            elif self.loss_type == "l2":
                err = F.mse_loss(fx, target, reduction='none').mean(dim=(1, 2, 3))
                err = (err * weight).mean()
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        return err
