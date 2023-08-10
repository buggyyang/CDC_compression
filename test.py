from data import load_data
import argparse
import os
import torch
import torchvision
import numpy as np
import pathlib
from modules.denoising_diffusion import GaussianDiffusion
from modules.distill_diffusion import GaussianDiffusion as DistillDiffusion
from modules.unet import Unet
from modules.compress_modules import ResnetCompressor
from modules.ae import AutoencoderKL
from ema_pytorch import EMA

parser = argparse.ArgumentParser(description="values from bash script")
parser.add_argument("--device", type=int, required=True, help="cuda device id")
parser.add_argument("--z_channels", type=int, default=3)
parser.add_argument("--ae_dim_mult", type=int, nargs='+', default=[1,2,4])
parser.add_argument("--ae_base_dim", type=int, default=128)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--decay", type=float, default=0.9)
parser.add_argument("--minf", type=float, default=0.2)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--n_step", type=int, default=1000000)
parser.add_argument("--scheduler_checkpoint_step", type=int, default=100000)
parser.add_argument("--log_checkpoint_step", type=int, default=5000)
parser.add_argument("--load_model", action="store_true")
parser.add_argument("--load_step", action="store_true")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument('--pred_mode', type=str, default='noise', help='prediction mode')
parser.add_argument('--loss_type', type=str, default='l2', help='type of loss')
parser.add_argument('--iteration_step', type=int, default=20000, help='number of iterations')
parser.add_argument('--embed_dim', type=int, default=64, help='dimension of embedding')
parser.add_argument('--embd_type', type=str, default="01", help='timestep embedding type')
parser.add_argument('--dim_mults', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6], help='dimension multipliers')
parser.add_argument('--hyper_dim_mults', type=int, nargs='+', default=[4, 4, 4], help='hyper dimension multipliers')
parser.add_argument('--context_dim_mults', type=int, nargs='+', default=[1, 2, 3, 4], help='context dimension multipliers')
parser.add_argument('--reverse_context_dim_mults', type=int, nargs='+', default=[4, 2], help='reverse context dimension multipliers')
parser.add_argument('--context_channels', type=int, default=64, help='number of context channels')
parser.add_argument('--use_weighted_loss', action='store_true', help='if use weighted loss')
parser.add_argument('--weight_clip', type=int, default=5, help='snr clip for weighted loss')
parser.add_argument('--use_mixed_precision', action='store_true', help='if use mixed precision')
parser.add_argument('--clip_noise', action='store_true', help='if clip the noise during sampling')
parser.add_argument('--val_num_of_batch', type=int, default=1, help='number of batches for validation')
parser.add_argument('--additional_note', type=str, default='', help='additional note')
parser.add_argument('--var_schedule', type=str, default='cosine', help='variance schedule')
parser.add_argument('--aux_loss_type', type=str, default='l2', help='type of auxiliary loss')
parser.add_argument("--aux_weight", type=float, default=0, help="weight for aux loss")
parser.add_argument("--data_name", type=str, default="vimeo", help="name of dataset")
parser.add_argument("--data_root", type=str, default="", help="root of dataset")
parser.add_argument("--params_root", type=str, default="")
parser.add_argument("--tensorboard_root", type=str, default="")
parser.add_argument("--ae_path", type=str, default="")
parser.add_argument("--use_aux_loss_weight_schedule", action="store_true", help="if use aux loss weight schedule")

parser.add_argument("--prefix", type=str, default="*")
parser.add_argument("--test_dataset", type=str, default="kodak")
parser.add_argument("--img_folder", type=str, default="x_param_test")
parser.add_argument("--bpp_folder", type=str, default="x_param_test")
parser.add_argument("--betas", type=float, nargs='+', required=True)
parser.add_argument('--sample_steps', type=int, default=129, help='number of steps for sampling (for validation)')
parser.add_argument("--eta", type=float, default=0)
parser.add_argument("--random_start_scale", type=float, default=0)
parser.add_argument("--version", type=str, default="-v2")
parser.add_argument("--distill", action="store_true")
parser.add_argument("--multisample", type=int, default=1)
parser.add_argument("--debug", type=float, default=float('inf'))

config = parser.parse_args()


def main(rank):

    # dist.init_process_group("nccl", rank=rank, world_size=world_size)

    bpps = []
    test_config = {
        "dataset_name": f"{config.test_dataset}",
        "data_path": "*",
        "img_size": 256,
        "img_channel": 3,
    }

    save_folder = f"{config.prefix}/{config.img_folder}/{test_config['dataset_name']}_model_{config.aux_weight}{config.aux_loss_type}-loss{config.loss_type}{config.additional_note}{config.version}"
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'bpps/{config.bpp_folder}').mkdir(parents=True, exist_ok=True)
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

    for bidx, beta in enumerate(config.betas):
        bpps.append([])
        train_data, val_data = load_data(
            test_config, bs, pin_memory=False, num_workers=config.n_workers,
        )

        context_model = ResnetCompressor(
            dim=config.embed_dim,
            dim_mults=config.context_dim_mults,
            reverse_dim_mults=config.reverse_context_dim_mults,
            hyper_dims_mults=config.hyper_dim_mults,
            channels=test_config["img_channel"],
            out_channels=config.context_channels,
        )

        if config.ae_path != "":
            ae_fn = AutoencoderKL(
                        z_channels=config.z_channels,
                        ch_mult=config.ae_dim_mult,
                        ch=config.ae_base_dim,
                        img_ch=3,
                        num_res_blocks=2
                    )
            ae_fn.load_state_dict(torch.load(config.ae_path))
        else:
            ae_fn = None
        
        denoise_model = Unet(
            dim=config.embed_dim,
            channels=test_config["img_channel"] if ae_fn is None else config.z_channels,
            context_channels=config.context_channels,
            dim_mults=config.dim_mults,
            context_dim_mults=reversed(config.reverse_context_dim_mults),
            embd_type=config.embd_type,
        )

        if config.distill:
            diffusion = DistillDiffusion(
                denoise_fn=denoise_model,
                context_fn=context_model,
                ae_fn=ae_fn,
                num_timesteps=config.iteration_step,
                loss_type=config.loss_type,
                lagrangian=beta,
                pred_mode=config.pred_mode,
                aux_loss_weight=config.aux_weight,
                aux_loss_type=config.aux_loss_type,
                var_schedule=config.var_schedule,
                use_loss_weight=config.use_weighted_loss,
                loss_weight_min=config.weight_clip,
                use_aux_loss_weight_schedule=config.use_aux_loss_weight_schedule,
            )
        else:
            diffusion = GaussianDiffusion(
                denoise_fn=denoise_model,
                context_fn=context_model,
                ae_fn=ae_fn,
                num_timesteps=config.iteration_step,
                loss_type=config.loss_type,
                lagrangian=beta,
                pred_mode=config.pred_mode,
                aux_loss_weight=config.aux_weight,
                aux_loss_type=config.aux_loss_type,
                var_schedule=config.var_schedule,
                use_loss_weight=config.use_weighted_loss,
                loss_weight_min=config.weight_clip,
                use_aux_loss_weight_schedule=config.use_aux_loss_weight_schedule,
            )

        model_name = (
            f"{'latent' if len(config.ae_path)>0 else 'image'}-{config.loss_type}-{'use_weight'+str(config.weight_clip) if config.use_weighted_loss else 'no_weight'}-vimeo"
            f"-d{config.embed_dim}-t{config.iteration_step}-b{beta}"
            f"-{config.pred_mode}-{config.var_schedule}-{config.embd_type}-{'mixed' if config.use_mixed_precision else 'float32'}-{'auxschedule-' if config.use_aux_loss_weight_schedule else ''}aux{config.aux_weight}{config.aux_loss_type if config.aux_weight>0 else ''}{config.additional_note}"
        )

        results_folder = os.path.join(config.params_root, f"{model_name}")

        if not config.distill:
            loaded_param = torch.load(
                str(f"{results_folder}/{model_name}_{2}.pt"),
                map_location=lambda storage, loc: storage,
            )
            ema = EMA(diffusion, beta=0.999, update_every=10, power=0.75, update_after_step=100)
            ema.load_state_dict(loaded_param["ema"])
            diffusion = ema.ema_model
        else:
            loaded_param = torch.load(
                str(f"{results_folder}/{model_name}_distilled_{2}.pt"),
                map_location=lambda storage, loc: storage,
            )
            diffusion.load_state_dict(loaded_param["distilled"])
        diffusion.to(rank)
        diffusion.eval()
        for i, data in enumerate(val_data):
            if bidx == 0:
                for j, img in enumerate(data[0]):
                    torchvision.utils.save_image(img, f"{save_folder}/truth/{i}-{j}.png")
            # print(args.rand_start, args.eta)
            to_be_compressed = data[0].repeat_interleave(config.multisample, 0).to(rank)
            if config.debug != float('inf'):
                compressed, bpp = diffusion.compress(
                    to_be_compressed * 2.0 - 1.0,
                    sample_steps=config.sample_steps,
                    bpp_return_mean=False,
                    init=config.debug * torch.ones_like(to_be_compressed),
                    eta=config.eta
                )
            else:
                compressed, bpp = diffusion.compress(
                    to_be_compressed * 2.0 - 1.0,
                    sample_steps=config.sample_steps,
                    bpp_return_mean=False,
                    init=(torch.randn_like(to_be_compressed) * config.random_start_scale) if config.random_start_scale > 0 else None,
                    eta=config.eta
                )
            compressed = compressed.reshape(-1, config.multisample, *compressed.shape[1:]).mean(1)

            bpps[bidx]+=list(bpp.cpu())

            if not os.path.isdir(f'{save_folder}/compressed/{beta}'):
                os.mkdir(f'{save_folder}/compressed/{beta}')

            for j, img in enumerate(compressed):
                torchvision.utils.save_image(
                    ((img + 1.0) / 2.0).clamp(0, 1), f"{save_folder}/compressed/{beta}/{i}-{j}.png"
                )
            if (i >= 29) and (test_config['dataset_name'] == "surrealism" or test_config['dataset_name'] == "expressionism"):
                break
        np.savetxt(f"bpps/{config.bpp_folder}/bpp_{test_config['dataset_name']}_model_{config.aux_weight}{config.aux_loss_type}_{beta}_{beta}-loss{config.loss_type}{config.additional_note}{config.version}.txt", [bpps[bidx]])
    np.savetxt(f"bpps/{config.bpp_folder}/bpp_{test_config['dataset_name']}_model_{config.aux_weight}{config.aux_loss_type}_{config.betas[0]}_{config.betas[-1]}-loss{config.loss_type}{config.additional_note}{config.version}.txt", bpps)


if __name__ == "__main__":
    main(config.device)
