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


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, mul=1000):
        t_freq = self.timestep_embedding(t * mul, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class BatchNormWithTimeEmbedding(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.bn = nn.GroupNorm(16, num_features, affine=False)
        # self.bn = nn.SyncBatchNorm(num_features, affine=False)
        self.embedder = TimestepEmbedder(num_features * 2)
        # nn.init.zeros_(self.embedder.mlp[-1].weight)
        nn.init.trunc_normal_(self.embedder.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.embedder.mlp[-1].bias)

    def forward(self, x, t):
        embed = self.embedder(t)
        embed = embed[:, :, None, None]
        gamma, beta = embed.chunk(2, dim=1)
        gamma = 1.0 + gamma
        normed = self.bn(x)
        out = normed * gamma + beta
        return out

class DisBlock(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.conv = nn.Conv2d(
            kernel_size=4, in_channels=in_channels, out_channels=hidden_size, stride=4, padding=0
        )
        self.norm = BatchNormWithTimeEmbedding(hidden_size)
        self.act = nn.SiLU()
    def forward(self, x, t):
        x = self.conv(x)
        x = self.norm(x, t)
        x = self.act(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, num_blocks, in_channels, hidden_size):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                DisBlock(
                    in_channels=in_channels,
                    hidden_size=hidden_size,
                )
            )
            in_channels = hidden_size
        self.classifier = nn.Conv2d(
            kernel_size=1, in_channels=hidden_size, out_channels=1, stride=1, padding=1
        )
    def forward(self, feature, t):
        B, C, H, W = feature.shape
        for block in self.blocks:
            feature = block(feature, t)
        out = self.classifier(feature).view(B, -1)
        out = out.sigmoid().clamp(0.01, 0.99)
        return out

class AdvTrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn:Callable=constant,
            lognorm_t=False,
            adv_weight=1.0,
            adv_blocks=3,
            adv_in_channels=3,
            adv_hidden_size=256,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.loss_weight_fn = loss_weight_fn
        self.adv_weight = adv_weight

        self.discriminator = Discriminator(
            num_blocks=adv_blocks,
            in_channels=adv_in_channels*2,
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

        out, _ = net(x_t, t, y)
        pred_x0 = x_t + sigma * out
        weight = self.loss_weight_fn(alpha, sigma)
        loss = weight*(out - v_t)**2

        real_feature = torch.cat([x_t, x], dim=1)
        fake_feature = torch.cat([x_t, pred_x0], dim=1)

        real_score_gan = self.discriminator(real_feature.detach(), t)
        fake_score_gan = self.discriminator(fake_feature.detach(), t)
        fake_score = self.discriminator(fake_feature, t)

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
