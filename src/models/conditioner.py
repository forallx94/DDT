import torch
import torch.nn as nn

class BaseConditioner(nn.Module):
    def __init__(self):
        super(BaseConditioner, self).__init__()

    def _impl_condition(self, y):
        ...
    def _impl_uncondition(self, y):
        ...
    def __call__(self, y):
        condition = self._impl_condition(y)      # 조건 정보
        uncondition = self._impl_uncondition(y)  # 비조건 정보
        return condition, uncondition

class LabelConditioner(BaseConditioner):
    def __init__(self, null_class):
        super().__init__()
        self.null_condition = null_class  # 비조건(class-agnostic)을 나타낼 class index

    def _impl_condition(self, y):
        return torch.tensor(y).long().cuda()  # 레이블을 long tensor로 변환하여 GPU에 올림

    def _impl_uncondition(self, y):
        return torch.full((len(y),), self.null_condition, dtype=torch.long).cuda()