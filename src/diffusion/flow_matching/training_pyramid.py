import torch
from typing import Callable
from src.diffusion.base.training import *
from src.diffusion.base.scheduling import BaseScheduler

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

class PyramidTrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn:Callable=constant,
            lognorm_t=False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.loss_weight_fn = loss_weight_fn


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


        output_pyramid = []
        def feature_hook(module, input, output):
           output_pyramid.extend(output)
        handle = net.decoder.register_forward_hook(feature_hook)
        net(x_t, t, y)
        handle.remove()

        loss = 0.0
        out_dict = dict()

        cur_v_t = v_t
        for i in range(len(output_pyramid)):
            cur_out = output_pyramid[i]
            loss_i = (cur_v_t - cur_out) ** 2
            loss += loss_i.mean()
            out_dict["loss_{}".format(i)] = loss_i.mean()
            cur_v_t = torch.nn.functional.interpolate(cur_v_t, scale_factor=0.5, mode='bilinear', align_corners=False)
        out_dict["loss"] = loss
        return out_dict

