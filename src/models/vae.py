import torch
import subprocess
import lightning.pytorch as pl

import logging


logger = logging.getLogger(__name__)
def class_fn_from_str(class_str):
    class_module, from_class = class_str.rsplit(".", 1)
    class_module = __import__(class_module, fromlist=[from_class])
    return getattr(class_module, from_class)


class BaseVAE(torch.nn.Module):
    def __init__(self, scale=1.0, shift=0.0):
        super().__init__()

        # 내부적으로 아무것도 하지 않는 Placeholder 모델
        self.model = torch.nn.Identity()

        # 인코딩 시 사용할 스케일 인자 (입력값을 나눔)
        self.scale = scale

        # 인코딩 시 사용할 쉬프트 인자 (입력값에 더함)
        self.shift = shift

    def encode(self, x):
        # 입력 x를 scale로 나누고 shift를 더함
        # 예: scale=2, shift=1 이면 x → (x / 2) + 1
        return x/self.scale+self.shift

    def decode(self, x):
        # encode의 반대 연산: shift를 빼고, scale을 곱함
        # 예: x → (x - 1) * 2
        return (x-self.shift)*self.scale


# very bad bugs with nearest sampling
class DownSampleVAE(BaseVAE):
    def __init__(self, down_ratio, scale=1.0, shift=0.0):
        super().__init__()
        self.model = torch.nn.Identity()
        self.scale = scale
        self.shift = shift
        self.down_ratio = down_ratio

    def encode(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=1/self.down_ratio, mode='bicubic', align_corners=False)
        return x/self.scale+self.shift

    def decode(self, x):
         x = (x-self.shift)*self.scale
         x = torch.nn.functional.interpolate(x, scale_factor=self.down_ratio, mode='bicubic', align_corners=False)
         return x



class LatentVAE(BaseVAE):
    def __init__(self, precompute=False, weight_path:str=None):
        super().__init__()
        self.precompute = precompute
        self.model = None
        self.weight_path = weight_path

        from diffusers.models import AutoencoderKL
        setattr(self, "model", AutoencoderKL.from_pretrained(self.weight_path))
        self.scaling_factor = self.model.config.scaling_factor

    @torch.no_grad()
    def encode(self, x):
        assert self.model is not None
        if self.precompute:
            return x.mul_(self.scaling_factor)
        return self.model.encode(x).latent_dist.sample().mul_(self.scaling_factor)

    @torch.no_grad()
    def decode(self, x):
        assert self.model is not None
        return self.model.decode(x.div_(self.scaling_factor)).sample


def uint82fp(x):
    x = x.to(torch.float32)
    x = (x - 127.5) / 127.5
    return x

def fp2uint8(x):
    x = torch.clip_((x + 1) * 127.5 + 0.5, 0, 255).to(torch.uint8)
    return x

