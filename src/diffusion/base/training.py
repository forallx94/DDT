import time

import torch
import torch.nn as nn

class BaseTrainer(nn.Module):
    def __init__(self,
                 null_condition_p=0.1,   # 일정 확률로 condition을 무효화 (CFG 훈련용)
                 log_var=False,          # log variance loss 항을 쓸지 여부
        ):
        super(BaseTrainer, self).__init__()
        self.null_condition_p = null_condition_p
        self.log_var = log_var

    def preproprocess(self, raw_iamges, x, condition, uncondition):
        bsz = x.shape[0]  # batch size
        if self.null_condition_p > 0:
            # 랜덤하게 null condition 적용 여부 결정 (확률 기반 마스크 생성)
            mask = torch.rand((bsz), device=condition.device) < self.null_condition_p
            mask = mask.expand_as(condition)  # 조건 텐서 크기로 확장
            condition[mask] = uncondition[mask]  # 일부 condition을 uncondition으로 바꿈
        return raw_iamges, x, condition

    # 학습 스텝 구현용 추상 함수
    def _impl_trainstep(self, net, ema_net, raw_images, x, y):
        raise NotImplementedError

    # 학습 호출 시:
    # 조건 무시 확률에 따라 일부 condition을 uncondition으로 대체
    # 실제 학습 로직 (_impl_trainstep) 실행
    def __call__(self, net, ema_net, raw_images, x, condition, uncondition):
        raw_images, x, condition = self.preproprocess(raw_images, x, condition, uncondition)
        return self._impl_trainstep(net, ema_net, raw_images, x, condition)

