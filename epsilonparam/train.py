from data import load_data
import argparse
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.trainer import Trainer
from modules.compress_modules import BigCompressor, SimpleCompressor
import config


parser = argparse.ArgumentParser(description="values from bash script")
parser.add_argument("--device", type=int, required=True, help="cuda device number")
parser.add_argument("--beta", type=float, required=True, help="beta")
parser.add_argument("--alpha", type=float, required=True, help="alpha")
args = parser.parse_args()

model_name = (
    f"{config.compressor}-{config.loss_type}-{config.data_config['dataset_name']}"
    f"-d{config.embed_dim}-t{config.iteration_step}-b{args.beta}-vbr{config.vbr}"
    f"-{config.pred_mode}-{config.var_schedule}-aux{args.alpha}{config.aux_loss_type if args.alpha>0 else ''}{config.additional_note}"
)


def schedule_func(ep):
    return max(config.decay ** ep, config.minf)


def main():

    train_data, val_data = load_data(
        config.data_config,
        config.batch_size,
        pin_memory=False,
        num_workers=config.n_workers,
    )

    denoise_model = Unet(
        dim=config.embed_dim,
        channels=config.data_config["img_channel"],
        context_channels=config.context_channels,
        dim_mults=config.dim_mults,
        context_dim_mults=config.context_dim_mults
    )

    if config.compressor == 'big':
        context_model = BigCompressor(
            dim=config.embed_dim,
            dim_mults=config.context_dim_mults,
            hyper_dims_mults=config.hyper_dim_mults,
            channels=config.data_config["img_channel"],
            out_channels=config.context_channels,
            vbr=config.vbr
        )
    elif config.compressor == 'simple':
        context_model = SimpleCompressor(
            dim=config.embed_dim,
            dim_mults=config.context_dim_mults,
            hyper_dims_mults=config.hyper_dim_mults,
            channels=config.data_config["img_channel"],
            out_channels=config.context_channels,
            vbr=config.vbr
        )
    else:
        raise NotImplementedError

    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        context_fn=context_model,
        clip_noise=config.clip_noise,
        num_timesteps=config.iteration_step,
        loss_type=config.loss_type,
        vbr=config.vbr,
        lagrangian=args.beta,
        pred_mode=config.pred_mode,
        aux_loss_weight=args.alpha,
        aux_loss_type=config.aux_loss_type,
        var_schedule=config.var_schedule
    ).to(args.device)

    trainer = Trainer(
        rank=args.device,
        sample_steps=config.sample_steps,
        diffusion_model=diffusion,
        train_dl=train_data,
        val_dl=val_data,
        scheduler_function=schedule_func,
        scheduler_checkpoint_step=config.scheduler_checkpoint_step,
        train_lr=config.lr,
        train_num_steps=config.n_step,
        save_and_sample_every=config.log_checkpoint_step,
        results_folder=os.path.join(config.result_root, f"{model_name}/"),
        tensorboard_dir=os.path.join(config.tensorboard_root, f"{model_name}/"),
        model_name=model_name,
        val_num_of_batch=config.val_num_of_batch,
        optimizer=config.optimizer,
        sample_mode=config.sample_mode
    )

    if config.load_model:
        trainer.load(load_step=config.load_step)

    trainer.train()


if __name__ == "__main__":
    main()
