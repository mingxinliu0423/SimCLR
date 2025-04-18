import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import save_config_file, accuracy, save_checkpoint

# ---------------------------------------------------------------------------- #
#  SimCLR training class – single rolling checkpoint stored in ./run/          #
# ---------------------------------------------------------------------------- #

torch.manual_seed(0)

RUN_DIR = Path("runs")                # all checkpoints + last_epoch live here
RUN_DIR.mkdir(exist_ok=True)
CKPT_FILE = RUN_DIR / "checkpoint_latest.pth.tar"
LAST_EPOCH_FILE = RUN_DIR / "last_epoch.txt"


class SimCLR:
    """Self‑Supervised training loop with one rolling checkpoint in ./run/.

    • At the end of every epoch we overwrite **run/checkpoint_latest.pth.tar**
      (so disk never fills up).
    • We also write **run/last_epoch.txt** with the epoch completed, enabling
      effortless resumes via `--start-epoch`.
    """

    # ------------------------------------------------------------------ #
    #  Constructor / optional resume                                    #
    # ------------------------------------------------------------------ #
    def __init__(self, *args, **kwargs):
        self.args = kwargs["args"]
        self.model = kwargs["model"].to(self.args.device)
        self.optimizer = kwargs["optimizer"]
        self.scheduler = kwargs["scheduler"]

        # TensorBoard log directory stays under ./runs by default
        self.writer = SummaryWriter(log_dir=getattr(self.args, "log_dir", None))
        logging.basicConfig(
            filename=os.path.join(self.writer.log_dir, "training.log"), level=logging.DEBUG
        )
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        # ------------------------- resume ---------------------------------- #
        start_epoch = getattr(self.args, "start_epoch", 0)
        checkpoint_path = getattr(self.args, "checkpoint_path", str(CKPT_FILE))
        if start_epoch > 0:
            if not Path(checkpoint_path).is_file():
                raise FileNotFoundError(
                    f"start_epoch={start_epoch} but {checkpoint_path} not found"
                )
            ckpt = torch.load(checkpoint_path, map_location=self.args.device)
            self.model.load_state_dict(ckpt["state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            try:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                logging.warning("Scheduler state not loaded – continuing fresh.")
            logging.info(f"✔ Resumed from {checkpoint_path} (epoch {ckpt['epoch']}).")

    # ------------------------------------------------------------------ #
    #  Info‑NCE loss                                                    #
    # ------------------------------------------------------------------ #
    def info_nce_loss(self, features):
        labels = torch.cat(
            [torch.arange(self.args.batch_size) for _ in range(self.args.n_views)], dim=0
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(self.args.device)

        features = torch.nn.functional.normalize(features, dim=1)
        sim = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        sim = sim[~mask].view(sim.shape[0], -1)

        positives = sim[labels.bool()].view(labels.shape[0], -1)
        negatives = sim[~labels.bool()].view(sim.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1) / self.args.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        return logits, labels

    # ------------------------------------------------------------------ #
    #  Training loop                                                    #
    # ------------------------------------------------------------------ #
    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # Save hyper‑params JSON only at epoch 0
        if getattr(self.args, "start_epoch", 0) == 0:
            save_config_file(self.writer.log_dir, self.args)

        start_epoch = getattr(self.args, "start_epoch", 0)
        n_iter = start_epoch * len(train_loader)

        logging.info(
            f"SimCLR: {self.args.epochs} total epochs (starting at {start_epoch})."
        )
        logging.info(f"Device: {self.args.device}.")

        for epoch in range(start_epoch, self.args.epochs):
            for images, _ in tqdm(train_loader, leave=False):
                images = torch.cat(images, dim=0).to(self.args.device)
                with autocast(enabled=self.args.fp16_precision):
                    feats = self.model(images)
                    logits, labels = self.info_nce_loss(feats)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar("loss", loss, n_iter)
                    self.writer.add_scalar("acc/top1", top1[0], n_iter)
                    self.writer.add_scalar("acc/top5", top5[0], n_iter)
                    self.writer.add_scalar("lr", self.scheduler.get_lr()[0], n_iter)
                n_iter += 1

            # LR schedule after warm‑up
            if epoch >= 10:
                self.scheduler.step()
            logging.debug(
                f"Epoch {epoch} | Loss {loss.item():.4f} | LR {self.scheduler.get_lr()[0]:.6f}"
            )

            # ------------- save rolling checkpoint + epoch marker ----------- #
            ckpt = {
                "epoch": epoch + 1,
                "arch": self.args.arch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            }
            save_checkpoint(ckpt, is_best=False, filename=str(CKPT_FILE))
            LAST_EPOCH_FILE.write_text(str(epoch + 1))

        logging.info("✔ Training complete. Final checkpoint in ./run/ .")
