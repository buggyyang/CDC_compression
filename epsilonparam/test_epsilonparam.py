from data import load_data
import argparse
import os
import torch
import torchvision
import numpy as np
import pathlib
import config
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.compress_modules import BigCompressor, SimpleCompressor


parser = argparse.ArgumentParser(description="values from bash script")

parser.add_argument("--ckpt", type=str, required=True) # ckpt path
parser.add_argument("--gamma", type=float, default=0.8) # noise intensity for decoding
parser.add_argument("--n_denoise_step", type=int, default=200) # number of denoising step
parser.add_argument("--device", type=int, default=0) # gpu device index
parser.add_argument("--img_dir", type=str, default='../imgs')
parser.add_argument("--out_dir", type=str, default='../compressed_imgs')
parser.add_argument("--lpips_weight", type=float, required=True) # either 0.9 or 0.0, note that this must match the ckpt you use, because with weight>0, the lpips-vggnet weights were also saved during training. Incorrect state_dict keys may lead to load_state_dict error when loading the ckpt.
args = parser.parse_args()

def main(rank):

    denoise_model = Unet(
        dim=64,
        channels=3,
        context_channels=3,
        dim_mults=(1, 2, 3, 4, 5, 6),
        context_dim_mults=(1, 2, 3, 4),
    )

    context_model = BigCompressor(
        dim=64,
        dim_mults=(1, 2, 3, 4),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        vbr=False,
    )

    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        context_fn=context_model,
        num_timesteps=20000,
        loss_type="l1",
        clip_noise="none",
        vbr=False,
        lagrangian=0.9,
        pred_mode="noise",
        var_schedule="linear",
        aux_loss_weight=args.lpips_weight,
        aux_loss_type="lpips"
    )

    loaded_param = torch.load(
        args.ckpt,
        map_location=lambda storage, loc: storage,
    )

    diffusion.load_state_dict(loaded_param["model"])
    diffusion.to(rank)
    diffusion.eval()

    for img in os.listdir(args.img_dir):
        if img.endswith(".png") or img.endswith(".jpg"):
            to_be_compressed = torchvision.io.read_image(os.path.join(args.img_dir, img)).unsqueeze(0).float().to(rank) / 255.0
            compressed, bpp = diffusion.compress(
                to_be_compressed.to(rank) * 2.0 - 1.0,
                sample_steps=args.n_denoise_step,
                sample_mode="ddim",
                bpp_return_mean=False,
                init=torch.randn_like(to_be_compressed) * args.gamma
            )
            compressed = compressed.clamp(-1, 1) / 2.0 + 0.5
            pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            torchvision.utils.save_image(compressed.cpu(), os.path.join(args.out_dir, img))
            print("bpp:", bpp)


if __name__ == "__main__":
    main(args.device)
