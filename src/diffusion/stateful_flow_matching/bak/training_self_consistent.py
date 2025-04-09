import torch
import math
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


class SelfConsistentTrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn:Callable=constant,
            lognorm_t=False,
            lpips_weight=1.0,
            lpips_encoder_layer=4,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.loss_weight_fn = loss_weight_fn
        self.lpips_encoder_layer = lpips_encoder_layer
        self.lpips_weight = lpips_weight

        
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

        real_features = []
        def forward_hook(net, input, output):
            real_features.append(output)
        handles = []
        for i in range(self.lpips_encoder_layer):
            handle = net.encoder.blocks[i].register_forward_hook(forward_hook)
            handles.append(handle)

        out, _ = net(x_t, t, y)

        for handle in handles:
            handle.remove()

        pred_x0 = (x_t + out * sigma)
        pred_xt = alpha * pred_x0 + noise * sigma
        weight = self.loss_weight_fn(alpha, sigma)
        loss = weight*(out - v_t)**2

        _,  fake_features = net(pred_xt, t, y, classify_layer=self.lpips_encoder_layer)

        lpips_loss = []
        for r, f in zip(real_features, fake_features):
            r = torch.nn.functional.normalize(r, dim=-1)
            f = torch.nn.functional.normalize(f, dim=-1)
            lpips_loss.append(torch.sum((r - f)**2, dim=-1).mean())
        lpips_loss = sum(lpips_loss)


        out = dict(
            lpips_loss=lpips_loss.mean(),
            fm_loss=loss.mean(),
            loss=loss.mean() + self.lpips_weight*lpips_loss.mean(),
        )
        return out
