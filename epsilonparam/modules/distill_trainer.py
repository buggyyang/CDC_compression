import torch
from pathlib import Path
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter

from .utils import cycle
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler


def batch_psnr(imgs1, imgs2):
    with torch.no_grad():
        batch_mse = torch.mean((imgs1 - imgs2) ** 2, axis=(1, 2, 3))
        batch_psnr = 20 * torch.log10(1.0 / torch.sqrt(batch_mse))
        return torch.mean(batch_psnr)

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
        train_lr=1e-4,
        train_num_steps=1000000,
        scheduler_checkpoint_step=100000,
        save_and_sample_every=1000,
        results_folder="./results",
        tensorboard_dir="./tensorboard_logs/diffusion-video/",
        model_name="model",
        val_num_of_batch=1,
        optimizer="adam",
        use_mixed_precision=False,
        kiter=10,
    ):
        super().__init__()
        self.model = diffusion_model
        self.val_num_of_batch = val_num_of_batch
        self.sample_steps = sample_steps
        self.save_and_sample_every = save_and_sample_every

        self.train_num_steps = train_num_steps
        self.kiter = kiter
        self.k = 0

        self.train_dl_class = train_dl
        self.val_dl_class = val_dl
        self.train_dl = cycle(train_dl)
        self.val_dl = cycle(val_dl)
        if optimizer == "adam":
            self.opt = Adam(self.model.parameters(), lr=train_lr)
        elif optimizer == "adamw":
            self.opt = AdamW(self.model.parameters(), lr=train_lr)
        if use_mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.step = 0
        self.device = rank
        self.scheduler_checkpoint_step = scheduler_checkpoint_step

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.model_name = model_name

        self.writer = SummaryWriter(tensorboard_dir)

    def save(self):
        data = {
            "distilled": self.model.state_dict(),
        }
        idx = ((self.step + self.k * self.train_num_steps) // self.save_and_sample_every) % 3
        torch.save(data, str(self.results_folder / f"{self.model_name}_distilled_{idx}.pt"))

    def load(self, idx=0, load_distilled=False, load_step=True):
        if not load_distilled:
            data = torch.load(
                str(self.results_folder / f"{self.model_name}_{idx}.pt"),
                map_location=lambda storage, loc: storage,
            )
            all_params = data["model"].keys()
            poped_params = []
            for key in all_params:
                if "train_" in key:
                    poped_params.append(key)
            for key in poped_params:
                data["model"].pop(key)
            if "ema" in data.keys():
                all_params = data["ema"].keys()
                poped_params = []
                for key in all_params:
                    if "train_" in key:
                        poped_params.append(key)
                for key in poped_params:
                    data["ema"].pop(key)
                
                poped_params = []
                all_params = data["ema"].keys()
                for key in all_params:
                    if "ema_model." in key:
                        poped_params.append(key)
                for key in poped_params:
                    data["ema"][key[10:]] = data["ema"].pop(key)
                self.model.load_state_dict(data["ema"], strict=False)
            else:
                self.model.load_state_dict(data["model"], strict=False)
        else:
            data = torch.load(
                str(self.results_folder / f"{self.model_name}_distilled_{idx}.pt"),
                map_location=lambda storage, loc: storage,
            )
            self.model.load_state_dict(data["distilled"], strict=False)

    def train(self):
        while self.k < self.kiter:
            self.step = 0
            self.model.copy_params_from_teacher_to_student()
            while self.step < self.train_num_steps:
                self.opt.zero_grad()
                data = next(self.train_dl).to(self.device)[0]
                self.model.train()
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss = self.model(data * 2.0 - 1., self.k)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss = self.model(data * 2.0 - 1., self.k)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                self.writer.add_scalar("loss", loss.item(), self.step + self.train_num_steps * self.k)

                if ((self.step + self.train_num_steps * self.k) % self.save_and_sample_every == 0):
                    # milestone = self.step // self.save_and_sample_every
                    for i, batch in enumerate(self.val_dl):
                        if i >= self.val_num_of_batch:
                            break
                        self.model.eval()
                        compressed, bpp = self.model.compress(
                            batch[0].to(self.device) * 2.0 - 1.0, self.sample_steps
                        )
                        compressed = (compressed + 1.0) * 0.5
                        self.writer.add_scalar(
                            f"bpp/num{i}",
                            bpp,
                            self.step // self.save_and_sample_every + self.step // self.save_and_sample_every * self.k,
                        )
                        self.writer.add_scalar(
                            f"psnr/num{i}",
                            batch_psnr(compressed.clamp(0.0, 1.0).to('cpu'), batch[0]),
                            self.step // self.save_and_sample_every + self.step // self.save_and_sample_every * self.k,
                        )
                        self.writer.add_images(
                            f"compressed/num{i}",
                            compressed.clamp(0.0, 1.0),
                            self.step // self.save_and_sample_every + self.step // self.save_and_sample_every * self.k,
                        )
                        self.writer.add_images(
                            f"original/num{i}",
                            batch[0],
                            self.step // self.save_and_sample_every,
                        )
                    self.save()

                self.step += 1
            self.model.copy_params_from_student_to_teacher()
            self.k += 1
        self.save()
        print("training completed")
