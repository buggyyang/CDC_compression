import torch
from pathlib import Path
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter

from .utils import cycle
from torch.optim.lr_scheduler import LambdaLR


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# trainer class
class Trainer(object):
    def __init__(
        self,
        rank,
        sample_steps,
        diffusion_model,
        train_dl,
        val_dl,
        scheduler_function,
        ema_decay=0.995,
        train_lr=1e-4,
        train_num_steps=1000000,
        scheduler_checkpoint_step=100000,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder="./results",
        tensorboard_dir="./tensorboard_logs/diffusion-video/",
        model_name="model",
        val_num_of_batch=1,
        optimizer="adam",
        sample_mode="ddpm"
    ):
        super().__init__()
        self.model = diffusion_model
        # self.ema = EMA(ema_decay)
        # self.ema_model = copy.deepcopy(self.model)
        self.sample_mode = sample_mode
        # self.update_ema_every = update_ema_every
        self.val_num_of_batch = val_num_of_batch
        self.sample_steps = sample_steps

        # self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.train_num_steps = train_num_steps

        self.train_dl_class = train_dl
        self.val_dl_class = val_dl
        self.train_dl = cycle(train_dl)
        self.val_dl = cycle(val_dl)
        if optimizer == "adam":
            self.opt = Adam(self.model.parameters(), lr=train_lr)
        elif optimizer == "adamw":
            self.opt = AdamW(self.model.parameters(), lr=train_lr)
        self.scheduler = LambdaLR(self.opt, lr_lambda=scheduler_function)

        self.step = 0
        self.device = rank
        self.scheduler_checkpoint_step = scheduler_checkpoint_step

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.model_name = model_name

        # if os.path.isdir(tensorboard_dir):
        #     shutil.rmtree(tensorboard_dir)
        self.writer = SummaryWriter(tensorboard_dir)

        # self.reset_parameters()

    # def reset_parameters(self):
    #     # self.ema_model.load_state_dict(self.model.state_dict())
    #     pass

    # def step_ema(self):
    #     # if self.step < self.step_start_ema:
    #     #     self.reset_parameters()
    #     # else:
    #     #     self.ema.update_model_average(self.ema_model, self.model)
    #     pass

    def save(self):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            # "ema": self.ema_model.module.state_dict(),
        }
        idx = (self.step // self.save_and_sample_every) % 3
        torch.save(data, str(self.results_folder / f"{self.model_name}_{idx}.pt"))

    def load(self, idx=0, load_step=True):
        data = torch.load(
            str(self.results_folder / f"{self.model_name}_{idx}.pt"),
            map_location=lambda storage, loc: storage,
        )

        if load_step:
            self.step = data["step"]
        try:
            self.model.module.load_state_dict(data["model"], strict=False)
        except:
            self.model.load_state_dict(data["model"], strict=False)
        # self.ema_model.module.load_state_dict(data["ema"], strict=False)

    def train(self):

        while self.step < self.train_num_steps:
            self.opt.zero_grad()
            if (self.step >= self.scheduler_checkpoint_step) and (self.step != 0):
                self.scheduler.step()
            data = next(self.train_dl).to(self.device)[0]
            self.model.train()
            loss, aloss = self.model(data * 2.0 - 1.0)
            loss.backward()
            aloss.backward()
            self.writer.add_scalar("loss", loss.item(), self.step)

            self.opt.step()

            if (self.step % self.save_and_sample_every == 0):
                # milestone = self.step // self.save_and_sample_every
                for i, batch in enumerate(self.val_dl):
                    if i >= self.val_num_of_batch:
                        break
                    if self.model.vbr:
                        scaler = torch.zeros(batch.shape[1]).unsqueeze(1).to(self.device)
                    else:
                        scaler = None
                    self.model.eval()
                    compressed, bpp = self.model.compress(
                        batch[0].to(self.device) * 2.0 - 1.0, self.sample_steps, scaler, self.sample_mode
                    )
                    compressed = (compressed + 1.0) * 0.5
                    self.writer.add_scalar(
                        f"bpp/num{i}",
                        bpp,
                        self.step // self.save_and_sample_every,
                    )
                    self.writer.add_images(
                        f"compressed/num{i}",
                        compressed.clamp(0.0, 1.0),
                        self.step // self.save_and_sample_every,
                    )
                    self.writer.add_images(
                        f"original/num{i}",
                        batch[0],
                        self.step // self.save_and_sample_every,
                    )
                self.save()

            self.step += 1
        self.save()
        print("training completed")
