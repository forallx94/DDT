from typing import Callable, Iterable, Any, Optional, Union, Sequence, Mapping, Dict
import os.path
import copy
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from lightning.pytorch.callbacks import Callback

# 내부 구현된 모듈 임포트
from src.models.vae import BaseVAE, fp2uint8               # VAE 및 uint8 변환 유틸
from src.models.conditioner import BaseConditioner         # condition encoder
from src.utils.model_loader import ModelLoader             # 모델 로딩 유틸 (사용 안 됨)
from src.callbacks.simple_ema import SimpleEMA             # EMA 추적 콜백
from src.diffusion.base.sampling import BaseSampler        # 샘플링 클래스
from src.diffusion.base.training import BaseTrainer        # 학습 클래스
from src.utils.no_grad import no_grad, filter_nograd_tensors  # grad 제거 유틸
from src.utils.copy import copy_params                     # 파라미터 복사 유틸

# 타입 별칭 정의
EMACallable = Callable[[nn.Module, nn.Module], SimpleEMA]
OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]


class LightningModel(pl.LightningModule):
    def __init__(self,
                 vae: BaseVAE,
                 conditioner: BaseConditioner,
                 denoiser: nn.Module,
                 diffusion_trainer: BaseTrainer,
                 diffusion_sampler: BaseSampler,
                 ema_tracker: Optional[EMACallable] = None,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 ):
        super().__init__()
        # 모델 구성 요소 설정
        self.vae = vae                          # 입력을 latent로 인코딩하고 다시 복원하는 VAE
        self.conditioner = conditioner          # y (클래스 레이블 등)을 조건 정보로 변환
        self.denoiser = denoiser                # 디퓨전 노이즈 예측 모델
        self.ema_denoiser = copy.deepcopy(denoiser)  # EMA 버전 (지속적 평균 추적)

        self.diffusion_sampler = diffusion_sampler    # 샘플링 (x_T → x_0) 담당
        self.diffusion_trainer = diffusion_trainer    # 학습용 노이즈 제거 loss 계산

        self.ema_tracker = ema_tracker          # EMA 추적기 (콜백 형태로 등록)
        self.optimizer = optimizer              # 옵티마이저 생성 함수
        self.lr_scheduler = lr_scheduler        # 스케줄러 생성 함수
        # self.model_loader = ModelLoader()

        self._strict_loading = False            # 로딩 시 strict 옵션 (사용 안됨)

    def configure_model(self) -> None:
        self.trainer.strategy.barrier()         # 분산 학습에서 동기화
        # self.denoiser = self.model_loader.load(self.denoiser)
        copy_params(src_model=self.denoiser, dst_model=self.ema_denoiser)  # EMA 모델에 파라미터 복사

        # self.denoiser = torch.compile(self.denoiser)
        # disable grad for conditioner and vae / 학습되지 않는 모듈에 대해 no_grad 적용
        no_grad(self.conditioner)
        no_grad(self.vae)
        no_grad(self.diffusion_sampler)
        no_grad(self.ema_denoiser)

    # 콜백 구성 (EMA 추적기 등록)
    # PyTorch Lightning 콜백은 학습 중 특정 시점(예: epoch 시작/끝, batch 끝 등)에 자동으로 실행되는 사용자 정의 동작
    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        ema_tracker = self.ema_tracker(self.denoiser, self.ema_denoiser)
        return [ema_tracker]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # grad가 필요한 파라미터만 필터링
        params_denoiser = filter_nograd_tensors(self.denoiser.parameters())
        params_trainer = filter_nograd_tensors(self.diffusion_trainer.parameters())
        
        optimizer: torch.optim.Optimizer = self.optimizer([*params_trainer, *params_denoiser])
        if self.lr_scheduler is None:
            return dict(
                optimizer=optimizer
            )
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return dict(
                optimizer=optimizer,
                lr_scheduler=lr_scheduler
            )

    def training_step(self, batch, batch_idx):
        raw_images, x, y = batch   # 원본 이미지, 입력 x (VAE 를 통해 생성된 latent space 를 위한 raw image 카피), 조건 y

        # 학습 중 VAE, conditioner는 학습하지 않으므로 no_grad 사용
        with torch.no_grad():
            x = self.vae.encode(x)                         # 입력을 latent로 인코딩
            condition, uncondition = self.conditioner(y)   # 조건, 비조건 생성

        # Denoiser와 EMA Denoiser를 사용한 디퓨전 학습 loss 계산
        loss = self.diffusion_trainer(self.denoiser, self.ema_denoiser, raw_images, x, condition, uncondition)

        self.log_dict(loss, prog_bar=True, on_step=True, sync_dist=False)  # 로깅
        return loss["loss"]  # main loss만 반환

    def predict_step(self, batch, batch_idx):
        xT, y, metadata = batch
        with torch.no_grad():
            condition, uncondition = self.conditioner(y) # 조건 정보 생성
        # Sample images / x_T → x_0 로 샘플 생성
        samples = self.diffusion_sampler(self.denoiser, xT, condition, uncondition)

        # latent → 이미지 복원
        samples = self.vae.decode(samples)

        # fp32 -1,1 -> uint8 0,255 /  float [-1,1] → uint8 [0,255] 변환
        samples = fp2uint8(samples)
        return samples

    def validation_step(self, batch, batch_idx):
        samples = self.predict_step(batch, batch_idx) # predict_step 재사용
        return samples

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        
        # 하위 모듈 별로 따로 저장 (모델 로딩 시 대응 가능)
        self.denoiser.state_dict(
            destination=destination,
            prefix=prefix+"denoiser.",
            keep_vars=keep_vars)
        self.ema_denoiser.state_dict(
            destination=destination,
            prefix=prefix+"ema_denoiser.",
            keep_vars=keep_vars)
        self.diffusion_trainer.state_dict(
            destination=destination,
            prefix=prefix+"diffusion_trainer.",
            keep_vars=keep_vars)
        return destination