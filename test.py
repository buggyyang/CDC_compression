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
parser.add_argument("--device", type=int, required=True, help="gpu index")
parser.add_argument("--alpha", type=float, required=True, help="alpha (equivalent to rho in paper, it will be always 0 when it's <= 0)")
parser.add_argument("--datasetname", type=str, required=True, help="dataset name")
parser.add_argument("--betas", type=float, nargs='+', required=True, help="lagrangian multiplier")
parser.add_argument("--ss", type=int, default=1000, help="number of decoding iteration")
parser.add_argument("--img_folder", type=str, default="imgs")
parser.add_argument("--bpp_folder", type=str, default="bpps")
parser.add_argument("--eta", type=float, default=0, help="eta from DDIM paper, but we fix it as 0")
parser.add_argument("--rand_start", type=float, default=0, help="std of the x_n sample")
parser.add_argument("--multisample", type=int, default=1, help="set it as 1 if you just want to sample once")
args = parser.parse_args()

def main(rank):

    # dist.init_process_group("nccl", rank=rank, world_size=world_size)

    bpps = []
    test_config = {
        "dataset_name": f"{args.datasetname}",
        "data_path": "*",
        "img_size": 256,
        "img_channel": 3,
    }
    prefix = "*"
    save_folder = f"{prefix}/{args.img_folder}/{test_config['dataset_name']}_model_{args.alpha}{config.aux_loss_type}-loss{config.loss_type}{config.additional_note}"
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'all_bpps/{args.bpp_folder}').mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(save_folder, "truth")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(save_folder, "compressed")).mkdir(parents=True, exist_ok=True)
    if test_config["dataset_name"] == "kodak":
        bs = 1
    elif test_config["dataset_name"] == "tecnick" or test_config["dataset_name"] == "div2k":
        bs = 10
    elif test_config["dataset_name"] == "cocotest":
        bs = 35
    elif test_config["dataset_name"] == "surrealism" or test_config["dataset_name"] == "expressionism":
        bs = 60

    for bidx, beta in enumerate(args.betas):
        bpps.append([])
        train_data, val_data = load_data(
            test_config, bs, pin_memory=False, num_workers=config.n_workers,
        )

        denoise_model = Unet(
            dim=config.embed_dim,
            channels=config.data_config["img_channel"],
            context_channels=config.context_channels,
            dim_mults=config.dim_mults,
            context_dim_mults=config.context_dim_mults,
        )

        context_model = BigCompressor(
            dim=config.embed_dim,
            dim_mults=config.context_dim_mults,
            hyper_dims_mults=config.hyper_dim_mults,
            channels=config.data_config["img_channel"],
            out_channels=config.context_channels,
            vbr=config.vbr,
        )

        diffusion = GaussianDiffusion(
            denoise_fn=denoise_model,
            context_fn=context_model,
            num_timesteps=config.iteration_step,
            loss_type=config.loss_type,
            clip_noise=config.clip_noise,
            vbr=config.vbr,
            lagrangian=beta,
            pred_mode=config.pred_mode,
            var_schedule=config.var_schedule,
            aux_loss_weight=args.alpha,
            aux_loss_type=config.aux_loss_type
        )

        model_name = (
            f"big-{config.loss_type}-{config.data_config['dataset_name']}"
            f"-d{config.embed_dim}-t{config.iteration_step}-b{beta}-vbr{config.vbr}"
            f"-{config.pred_mode}-{config.var_schedule}-aux{args.alpha}{config.aux_loss_type if args.alpha>0 else ''}{config.additional_note}"
        )

        results_folder = os.path.join(config.result_root, f"{model_name}")
        loaded_param = torch.load(
            str(f"{results_folder}/{model_name}_{0}.pt"),
            map_location=lambda storage, loc: storage,
        )

        diffusion.load_state_dict(loaded_param["model"], strict=False)
        diffusion.to(rank)
        diffusion.eval()
        for i, data in enumerate(val_data):
            if bidx == 0:
                for j, img in enumerate(data[0]):
                    torchvision.utils.save_image(img, f"{save_folder}/truth/{i}-{j}.png")
            # print(args.rand_start, args.eta)
            for t in range(args.multisample):
                compressed, bpp = diffusion.compress(
                    data[0].to(rank) * 2.0 - 1.0,
                    sample_steps=args.ss,
                    sample_mode=config.sample_mode,
                    bpp_return_mean=False,
                    init=(torch.randn_like(data[0]).to(rank) * args.rand_start) if args.rand_start > 0 else None,
                    eta=args.eta
                )

                if not os.path.isdir(f'{save_folder}/compressed/{beta}'):
                    os.mkdir(f'{save_folder}/compressed/{beta}')

                for j, img in enumerate(compressed):
                    torchvision.utils.save_image(
                        ((img + 1.0) / 2.0).clamp(0, 1), f"{save_folder}/compressed/{beta}/{i}-{j}-{t}.png"
                    )
            bpps[bidx]+=list(bpp.cpu())
            if (i >= 29) and (test_config['dataset_name'] == "surrealism" or test_config['dataset_name'] == "expressionism"):
                break
    np.savetxt(f"all_bpps/{args.bpp_folder}/bpp_{test_config['dataset_name']}_model_{args.alpha}{config.aux_loss_type}_{args.betas[0]}_{args.betas[-1]}-loss{config.loss_type}{config.additional_note}.txt", bpps)


if __name__ == "__main__":
    main(args.device)
