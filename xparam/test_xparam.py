import argparse
import os
import torch
import torchvision
import numpy as np
import pathlib
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.compress_modules import ResnetCompressor
from ema_pytorch import EMA

parser = argparse.ArgumentParser(description="values from bash script")

parser.add_argument("--ckpt", type=str, required=True) # ckpt path
parser.add_argument("--gamma", type=float, default=0.8) # noise intensity for decoding
parser.add_argument("--n_denoise_step", type=int, default=65) # number of denoising step
parser.add_argument("--device", type=int, default=0) # gpu device index
parser.add_argument("--lpips_weight", type=float, required=True) # either 0.9 or 0.0, note that this must match the ckpt you use, because with weight>0, the lpips-vggnet weights were also saved during training. Incorrect state_dict keys may lead to load_state_dict error when loading the ckpt.

config = parser.parse_args()


def main(rank):

    
        
    denoise_model = Unet(
        dim=64,
        channels=3,
        context_channels=64,
        dim_mults=[1,2,3,4,5,6],
        context_dim_mults=[1,2,3,4],
        embd_type="01",
    )

    context_model = ResnetCompressor(
        dim=64,
        dim_mults=[1,2,3,4],
        reverse_dim_mults=[4,3,2,1],
        hyper_dims_mults=[4,4,4],
        channels=3,
        out_channels=64,
    )

    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        context_fn=context_model,
        ae_fn=None,
        num_timesteps=8193,
        loss_type="l2",
        lagrangian=0.0032,
        pred_mode="x",
        aux_loss_weight=config.lpips_weight,
        aux_loss_type="lpips",
        var_schedule="cosine",
        use_loss_weight=True,
        loss_weight_min=5,
        use_aux_loss_weight_schedule=False,
    )
    loaded_param = torch.load(
        config.ckpt,
        map_location=lambda storage, loc: storage,
    )
    ema = EMA(diffusion, beta=0.999, update_every=10, power=0.75, update_after_step=100)
    ema.load_state_dict(loaded_param["ema"])
    diffusion = ema.ema_model
    diffusion.to(rank)
    diffusion.eval()
    
    to_be_compressed = 0.5 * torch.ones(1, 3, 256, 256).to(rank)
    compressed, bpp = diffusion.compress(
        to_be_compressed * 2.0 - 1.0,
        sample_steps=config.n_denoise_step,
        bpp_return_mean=False,
        init=torch.randn_like(to_be_compressed) * config.gamma
    )


if __name__ == "__main__":
    main(config.device)
