import lightning.pytorch as pl
from lightning.pytorch import Callback


import os.path
import numpy
from PIL import Image
from typing import Sequence, Any, Dict
from concurrent.futures import ThreadPoolExecutor

from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.rank_zero import rank_zero_info

def process_fn(image, path):
    Image.fromarray(image).save(path)

class SaveImagesHook(Callback):
    def __init__(self, save_dir="val", max_save_num=100, compressed=True):
       self.save_dir = save_dir
       self.max_save_num = max_save_num
       self.compressed = compressed

    def save_start(self, target_dir):
        self.target_dir = target_dir
        self.executor_pool = ThreadPoolExecutor(max_workers=8)
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir, exist_ok=True)
        else:
            if os.listdir(target_dir) and "debug" not in str(target_dir):
                raise FileExistsError(f'{self.target_dir} already exists and not empty!')
        self.samples = []
        self._have_saved_num = 0
        rank_zero_info(f"Save images to {self.target_dir}")

    def save_image(self, images, filenames):
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        for sample, filename in zip(images, filenames):
            if isinstance(filename, Sequence):
                filename = filename[0]
            path = f'{self.target_dir}/{filename}'
            if self._have_saved_num >= self.max_save_num:
                break
            self.executor_pool.submit(process_fn, sample, path)
            self._have_saved_num += 1

    def process_batch(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        samples: STEP_OUTPUT,
        batch: Any,
    ) -> None:
        b, c, h, w = samples.shape
        xT, y, metadata = batch
        all_samples = pl_module.all_gather(samples).view(-1, c, h, w)
        self.save_image(samples, metadata)
        if trainer.is_global_zero:
            all_samples = all_samples.permute(0, 2, 3, 1).cpu().numpy()
            self.samples.append(all_samples)

    def save_end(self):
        if self.compressed and len(self.samples) > 0:
            samples = numpy.concatenate(self.samples)
            numpy.savez(f'{self.target_dir}/output.npz', arr_0=samples)
        self.executor_pool.shutdown(wait=True)
        self.samples = []
        self.target_dir = None
        self._have_saved_num = 0
        self.executor_pool = None

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        target_dir = os.path.join(trainer.default_root_dir, self.save_dir, f"iter_{trainer.global_step}")
        self.save_start(target_dir)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.process_batch(trainer, pl_module, outputs, batch)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.save_end()

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        target_dir = os.path.join(trainer.default_root_dir, self.save_dir, "predict")
        self.save_start(target_dir)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        samples: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.process_batch(trainer, pl_module, samples, batch)

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.save_end()