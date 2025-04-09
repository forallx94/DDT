import torch
import copy
import timm
from torch.nn import Parameter

from src.utils.no_grad import no_grad
from typing import Callable, Iterator, Tuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from src.diffusion.base.training import *
from src.diffusion.base.scheduling import BaseScheduler
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


class DINOv2(nn.Module):
    def __init__(self, weight_path:str):
        super(DINOv2, self).__init__()
        self.encoder = torch.hub.load(
            '/mnt/bn/wangshuai6/torch_hub/facebookresearch_dinov2_main',
            weight_path,
            source="local",
            skip_validation=True
        )
        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        self.encoder.head = torch.nn.Identity()
        self.patch_size = self.encoder.patch_embed.patch_size
        self.precomputed_pos_embed = dict()

    def fetch_pos(self, h, w):
        key = (h, w)
        if key in self.precomputed_pos_embed:
            return self.precomputed_pos_embed[key]
        value = timm.layers.pos_embed.resample_abs_pos_embed(
            self.pos_embed.data, [h, w],
        )
        self.precomputed_pos_embed[key] = value
        return value

    def forward(self, x):
        b, c, h, w = x.shape
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, (int(224*h/256), int(224*w/256)), mode='bicubic')
        b, c, h, w = x.shape
        patch_num_h, patch_num_w = h//self.patch_size[0], w//self.patch_size[1]
        pos_embed_data = self.fetch_pos(patch_num_h, patch_num_w)
        self.encoder.pos_embed.data = pos_embed_data
        feature = self.encoder.forward_features(x)['x_norm_patchtokens']
        return feature


class REPALPIPSTrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn:Callable=constant,
            feat_loss_weight: float=0.5,
            lognorm_t=False,
            lpips_weight=1.0,
            encoder_weight_path=None,
            align_layer=8,
            proj_denoiser_dim=256,
            proj_hidden_dim=256,
            proj_encoder_dim=256,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.loss_weight_fn = loss_weight_fn
        self.feat_loss_weight = feat_loss_weight
        self.align_layer = align_layer
        self.encoder = DINOv2(encoder_weight_path)
        self.proj_encoder_dim = proj_encoder_dim
        no_grad(self.encoder)

        self.lpips_weight = lpips_weight
        self.lpips = _NoTrainLpips(net="vgg")
        self.lpips = self.lpips.to(torch.bfloat16)
        no_grad(self.lpips)

        self.proj = nn.Sequential(
            nn.Sequential(
                nn.Linear(proj_denoiser_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_encoder_dim),
            )
        )

    def _impl_trainstep(self, net, ema_net, raw_images, x, y):
        batch_size, c, height, width = x.shape
        if self.lognorm_t:
            base_t = torch.randn((batch_size), device=x.device, dtype=x.dtype).sigmoid()
        else:
            base_t = torch.rand((batch_size), device=x.device, dtype=x.dtype)
        t = base_t

        noise = torch.randn_like(x)
        alpha = self.scheduler.alpha(t)
        dalpha = self.scheduler.dalpha(t)
        sigma = self.scheduler.sigma(t)
        dsigma = self.scheduler.dsigma(t)

        x_t = alpha * x + noise * sigma
        v_t = dalpha * x + dsigma * noise

        src_feature = []
        def forward_hook(net, input, output):
            src_feature.append(output)
        if getattr(net, "blocks", None) is not None:
            handle = net.blocks[self.align_layer - 1].register_forward_hook(forward_hook)
        else:
            handle = net.encoder.blocks[self.align_layer - 1].register_forward_hook(forward_hook)

        out, _ = net(x_t, t, y)
        src_feature = self.proj(src_feature[0])
        handle.remove()

        with torch.no_grad():
            dst_feature = self.encoder(raw_images)

        if dst_feature.shape[1] != src_feature.shape[1]:
            dst_length = dst_feature.shape[1]
            rescale_ratio = (src_feature.shape[1] / dst_feature.shape[1])**0.5
            dst_height = (dst_length)**0.5 * (height/width)**0.5
            dst_width =  (dst_length)**0.5 * (width/height)**0.5
            dst_feature = dst_feature.view(batch_size, int(dst_height), int(dst_width), self.proj_encoder_dim)
            dst_feature = dst_feature.permute(0, 3, 1, 2)
            dst_feature = torch.nn.functional.interpolate(dst_feature, scale_factor=rescale_ratio, mode='bilinear', align_corners=False)
            dst_feature = dst_feature.permute(0, 2, 3, 1)
            dst_feature = dst_feature.view(batch_size, -1, self.proj_encoder_dim)

        cos_sim = torch.nn.functional.cosine_similarity(src_feature, dst_feature, dim=-1)
        cos_loss = 1 - cos_sim

        weight = self.loss_weight_fn(alpha, sigma)
        fm_loss = weight*(out - v_t)**2

        pred_x0 = (x_t + out * sigma)
        target_x0 = x
        # fixbug lpips std
        lpips = self.lpips(pred_x0 * 0.5, target_x0 * 0.5)

        out = dict(
            lpips_loss=lpips.mean(),
            fm_loss=fm_loss.mean(),
            cos_loss=cos_loss.mean(),
            loss=fm_loss.mean() + self.feat_loss_weight*cos_loss.mean() + self.lpips_weight*lpips.mean(),
        )
        return out

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        self.proj.state_dict(
            destination=destination,
            prefix=prefix + "proj.",
            keep_vars=keep_vars)

