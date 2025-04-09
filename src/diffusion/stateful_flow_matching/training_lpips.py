import torch
from typing import Callable
from src.diffusion.base.training import *
from src.diffusion.base.scheduling import BaseScheduler
from src.utils.no_grad import no_grad
from torchmetrics.image.lpip import _NoTrainLpips

def inverse_sigma(alpha, sigma):
    return 1/sigma**2
def snr(alpha, sigma):
    return alpha/sigma
def minsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha/sigma, min=threshold)
def maxsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha/sigma, max=threshold)
def constant(alpha, sigma):
    return 1

class LPIPSTrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn:Callable=constant,
            lognorm_t=False,
            lpips_weight=1.0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.loss_weight_fn = loss_weight_fn
        self.lpips_weight = lpips_weight
        self.lpips = _NoTrainLpips(net="vgg")
        self.lpips = self.lpips.to(torch.bfloat16)
        # self.lpips = torch.compile(self.lpips)
        no_grad(self.lpips)
        
    def _impl_trainstep(self, net, ema_net, raw_images, x, y):
        batch_size = x.shape[0]
        if self.lognorm_t:
            t = torch.randn(batch_size).to(x.device, x.dtype).sigmoid()
        else:
            t = torch.rand(batch_size).to(x.device, x.dtype)
        noise = torch.randn_like(x)
        alpha = self.scheduler.alpha(t)
        dalpha = self.scheduler.dalpha(t)
        sigma = self.scheduler.sigma(t)
        dsigma = self.scheduler.dsigma(t)
        w = self.scheduler.w(t)

        x_t = alpha * x + noise * sigma
        v_t = dalpha * x + dsigma * noise
        out, _ = net(x_t, t, y)
        weight = self.loss_weight_fn(alpha, sigma)
        loss = weight*(out - v_t)**2

        pred_x0 = (x_t + out*sigma)
        target_x0 = x
        # fixbug lpips std
        lpips = self.lpips(pred_x0*0.5, target_x0*0.5)

        out = dict(
            lpips_loss=lpips.mean(),
            fm_loss=loss.mean(),
            loss=loss.mean() + lpips.mean()*self.lpips_weight,
        )
        return out

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        return