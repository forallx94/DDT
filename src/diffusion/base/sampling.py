from typing import Union, List

import torch
import torch.nn as nn
from typing import Callable
from src.diffusion.base.scheduling import BaseScheduler  # 타임스텝 스케줄링 담당 클래스

# 다양한 샘플링 방식 (e.g., DDPM, DDIM, ODE, Flow Matching 등)을 구현할 공통 인터페이스
class BaseSampler(nn.Module):
    def __init__(self,
                 scheduler: BaseScheduler = None,         # 타임스텝/노이즈 스케줄링 제어 객체
                 guidance_fn: Callable = None,            # CFG 등에서 사용하는 조건 조합 함수
                 num_steps: int = 250,                    # 샘플링에 사용할 스텝 수
                 guidance: Union[float, List[float]] = 1.0,  # CFG scale (상수 또는 스텝별 리스트)
                 *args,
                 **kwargs
        ):
        super(BaseSampler, self).__init__()
        self.num_steps = num_steps
        self.guidance = guidance
        self.guidance_fn = guidance_fn
        self.scheduler = scheduler

    # 샘플링 실제 구현 함수 (추상)
    def _impl_sampling(self, net, noise, condition, uncondition):
        raise NotImplementedError

    # 샘플링 호출 인터페이스
    def __call__(self, net, noise, condition, uncondition):
        denoised = self._impl_sampling(net, noise, condition, uncondition)
        return denoised


