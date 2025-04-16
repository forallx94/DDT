import time
from typing import Any, Union

from src.utils.patch_bugs import *

import os
import torch
from lightning import Trainer, LightningModule
from src.lightning_data import DataModule
from src.lightning_model import LightningModel
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback

import logging
logger = logging.getLogger("lightning.pytorch")
# log_path = os.path.join( f"log.txt")
# logger.addHandler(logging.FileHandler(log_path))

# 설정 파일을 학습 루트 디렉토리에 시간 정보 포함하여 저장하는 콜백
class ReWriteRootSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        stamp = time.strftime('%y%m%d%H%M')
        file_path = os.path.join(trainer.default_root_dir, f"config-{stage}-{stamp}.yaml")
        self.parser.save(
            self.config, file_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
        )


# LightningCLI를 상속받아 모델, 데이터, 설정을 CLI 기반으로 구성
class ReWriteRootDirCli(LightningCLI):
    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()
        config_trainer = self._get(self.config, "trainer", default={})

        # predict path & logger check
        if self.subcommand == "predict":
            config_trainer.logger = None

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # 사용자 지정 태그를 설정에 추가
        class TagsClass:
            def __init__(self, exp:str):
                ...
        parser.add_class_arguments(TagsClass, nested_key="tags")

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # torch hub, huggingface cache 경로 설정 가능하게 함
        super().add_default_arguments_to_parser(parser)
        parser.add_argument("--torch_hub_dir", type=str, default=None, help=("torch hub dir"),)
        parser.add_argument("--huggingface_cache_dir", type=str, default=None, help=("huggingface hub dir"),)

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        # trainer 저장 디렉토리 자동 생성
        config_trainer = self._get(self.config_init, "trainer", default={})
        default_root_dir = config_trainer.get("default_root_dir", None)

        if default_root_dir is None:
            default_root_dir = os.path.join(os.getcwd(), "workdirs")

        dirname = ""
        for v, k in self._get(self.config, "tags", default={}).items():
            dirname += f"{v}_{k}"
        default_root_dir = os.path.join(default_root_dir, dirname)
        is_resume = self._get(self.config_init, "ckpt_path", default=None)
        # 중복된 실험 방지
        if os.path.exists(default_root_dir) and "debug" not in default_root_dir:
            if os.listdir(default_root_dir) and self.subcommand != "predict" and not is_resume:
                raise FileExistsError(f"{default_root_dir} already exists")

        config_trainer.default_root_dir = default_root_dir
        trainer = super().instantiate_trainer(**kwargs)
        if trainer.is_global_zero:
            os.makedirs(default_root_dir, exist_ok=True)
        return trainer

    def instantiate_classes(self) -> None:
        # torch hub & huggingface cache 디렉토리 환경 변수 등록
        torch_hub_dir = self._get(self.config, "torch_hub_dir")
        huggingface_cache_dir = self._get(self.config, "huggingface_cache_dir")
        if huggingface_cache_dir is not None:
            os.environ["HUGGINGFACE_HUB_CACHE"] = huggingface_cache_dir
        if torch_hub_dir is not None:
            os.environ["TORCH_HOME"] = torch_hub_dir
            torch.hub.set_dir(torch_hub_dir)
        super().instantiate_classes()

if __name__ == "__main__":

    cli = ReWriteRootDirCli(
        LightningModel,         # 학습에 사용할 모델 클래스
        DataModule,             # 데이터 로딩 담당 클래스
        auto_configure_optimizers=False,           # 옵티마이저를 모델 내에서 수동 구성
        save_config_callback=ReWriteRootSaveConfigCallback,  # 설정 저장 방식 재정의
        save_config_kwargs={"overwrite": True}     # 설정 저장 시 덮어쓰기 허용
    )
