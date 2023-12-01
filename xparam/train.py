from data import load_data
import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.trainer import Trainer
from modules.compress_modules import ResnetCompressor
from modules.ae import AutoencoderKL


parser = argparse.ArgumentParser(description="values from bash script")
parser.add_argument("--device", type=int, required=True, help="cuda device id")
parser.add_argument("--beta", type=float, required=True, help="beta for bitrate control")
parser.add_argument("--z_channels", type=int, default=3)
parser.add_argument("--ae_dim_mult", type=int, nargs='+', default=[1,2,4])
parser.add_argument("--ae_base_dim", type=int, default=128)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--decay", type=float, default=0.8)
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
parser.add_argument('--sample_steps', type=int, default=129, help='number of steps for sampling (for validation)')
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
parser.add_argument("--data_root", type=str, default="/extra/ucibdl0/shared/data", help="root of dataset")
parser.add_argument("--params_root", type=str, default="/extra/ucibdl0/ruihan/params_cdc_ldm")
parser.add_argument("--tensorboard_root", type=str, default="/extra/ucibdl0/ruihan/tblogs_cdc_ldm")
parser.add_argument("--ae_path", type=str, default="/extra/ucibdl0/ruihan/params_cdc_ldm_ae/ae_zc3_dm3_bd128_b1e-06_lossl2_final.pth", help="path to ae model")
parser.add_argument("--use_aux_loss_weight_schedule", action="store_true", help="if use aux loss weight schedule")

config = parser.parse_args()


data_config = {
    "dataset_name": config.data_name,
    "data_path": config.data_root,
    "sequence_length": 1,
    "img_size": 256,
    "img_channel": 3,
    "add_noise": False,
    "img_hz_flip": False,
}

model_name = (
    f"{'latent' if len(config.ae_path)>0 else 'image'}-{config.loss_type}-{'use_weight'+str(config.weight_clip) if config.use_weighted_loss else 'no_weight'}-{data_config['dataset_name']}"
    f"-d{config.embed_dim}-t{config.iteration_step}-b{config.beta}"
    f"-{config.pred_mode}-{config.var_schedule}-{config.embd_type}-{'mixed' if config.use_mixed_precision else 'float32'}-{'auxschedule-' if config.use_aux_loss_weight_schedule else ''}aux{config.aux_weight}{config.aux_loss_type if config.aux_weight>0 else ''}{config.additional_note}"
)

print('model name:')
print(model_name)


def schedule_func(ep):
    return max(config.decay ** ep, config.minf)


def main():

    train_data, val_data = load_data(
        data_config,
        config.batch_size,
        pin_memory=False,
        num_workers=config.n_workers,
    )


    context_model = ResnetCompressor(
        dim=config.embed_dim,
        dim_mults=config.context_dim_mults,
        reverse_dim_mults=config.reverse_context_dim_mults,
        hyper_dims_mults=config.hyper_dim_mults,
        channels=data_config["img_channel"],
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
        channels=data_config["img_channel"] if ae_fn is None else config.z_channels,
        context_channels=config.context_channels,
        dim_mults=config.dim_mults,
        context_dim_mults=reversed(config.reverse_context_dim_mults),
        embd_type=config.embd_type,
    )

    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        context_fn=context_model,
        ae_fn=ae_fn,
        num_timesteps=config.iteration_step,
        loss_type=config.loss_type,
        lagrangian=config.beta,
        pred_mode=config.pred_mode,
        aux_loss_weight=config.aux_weight,
        aux_loss_type=config.aux_loss_type,
        var_schedule=config.var_schedule,
        use_loss_weight=config.use_weighted_loss,
        loss_weight_min=config.weight_clip,
        use_aux_loss_weight_schedule=config.use_aux_loss_weight_schedule
    ).to(config.device)

    trainer = Trainer(
        rank=config.device,
        sample_steps=config.sample_steps,
        diffusion_model=diffusion,
        train_dl=train_data,
        val_dl=val_data,
        scheduler_function=schedule_func,
        scheduler_checkpoint_step=config.scheduler_checkpoint_step,
        train_lr=config.lr,
        train_num_steps=config.n_step,
        save_and_sample_every=config.log_checkpoint_step,
        results_folder=os.path.join(config.params_root, f"{model_name}/"),
        tensorboard_dir=os.path.join(config.tensorboard_root, f"{model_name}/"),
        model_name=model_name,
        val_num_of_batch=config.val_num_of_batch,
        optimizer=config.optimizer,
        use_mixed_precision=config.use_mixed_precision,
    )

    if config.load_model:
        print('loaded')
        trainer.load(idx=0, load_step=config.load_step)

    trainer.train()


if __name__ == "__main__":
    main()
