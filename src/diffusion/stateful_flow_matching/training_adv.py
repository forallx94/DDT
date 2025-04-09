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

class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(kernel_size=4, in_channels=in_channels, out_channels=hidden_size, stride=2, padding=1),  # 16x16 -> 8x8
            nn.GroupNorm(num_groups=32, num_channels=hidden_size),
            nn.SiLU(),
            nn.Conv2d(kernel_size=4, in_channels=hidden_size, out_channels=hidden_size, stride=2, padding=1), # 8x8 -> 4x4
            nn.GroupNorm(num_groups=32, num_channels=hidden_size),
            nn.SiLU(),
            nn.Conv2d(kernel_size=4, in_channels=hidden_size, out_channels=hidden_size, stride=2, padding=1),# 8x8 -> 4x4
            nn.GroupNorm(num_groups=32, num_channels=hidden_size),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(kernel_size=1, in_channels=hidden_size, out_channels=1, stride=1, padding=0),  # 1x1 -> 1x1
        )

    def forward(self, feature):
        B, L, C = feature.shape
        H = W = int(math.sqrt(L))
        feature = feature.permute(0, 2, 1)
        feature = feature.view(B, C, H, W)
        out = self.head(feature).sigmoid().clamp(0.01, 0.99)
        return out

class AdvTrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn:Callable=constant,
            lognorm_t=False,
            adv_weight=1.0,
            adv_encoder_layer=4,
            adv_in_channels=768,
            adv_hidden_size=256,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.loss_weight_fn = loss_weight_fn
        self.adv_weight = adv_weight
        self.adv_encoder_layer = adv_encoder_layer

        self.dis_head = Discriminator(
            in_channels=adv_in_channels,
            hidden_size=adv_hidden_size,
        )

        
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

        adv_feature = []
        def forward_hook(net, input, output):
            adv_feature.append(output)
        handle = net.encoder.blocks[self.adv_encoder_layer - 1].register_forward_hook(forward_hook)

        out, _ = net(x_t, t, y)
        weight = self.loss_weight_fn(alpha, sigma)
        loss = weight*(out - v_t)**2

        pred_x0 = (x_t + out * sigma)
        pred_xt = alpha * pred_x0 + torch.randn_like(pred_x0) * sigma
        real_feature = adv_feature.pop()
        net(pred_xt, t, y, classify_layer=self.adv_encoder_layer)
        fake_feature = adv_feature.pop()
        handle.remove()


        real_score_gan = self.dis_head(real_feature.detach())
        fake_score_gan = self.dis_head(fake_feature.detach())
        fake_score = self.dis_head(fake_feature)

        loss_gan = -torch.log(1 - fake_score_gan) - torch.log(real_score_gan)
        acc_real = (real_score_gan > 0.5).float()
        acc_fake = (fake_score_gan < 0.5).float()
        loss_adv = -torch.log(fake_score)
        loss_adv_hack = torch.log(fake_score_gan)

        out = dict(
            adv_loss=loss_adv.mean(),
            gan_loss=loss_gan.mean(),
            fm_loss=loss.mean(),
            loss=loss.mean() + (loss_adv.mean() + loss_adv_hack.mean())*self.adv_weight + loss_gan.mean(),
            acc_real=acc_real.mean(),
            acc_fake=acc_fake.mean(),
        )
        return out
