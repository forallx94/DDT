import torch
import copy
import os
import timm
import transformers
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from torchvision.transforms import Normalize

class RandViT(nn.Module):
    def __init__(self, model_id, weight_path:str=None):
        super(RandViT, self).__init__()
        self.encoder = timm.create_model(
            model_id,
            num_classes=0,
        )
        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        self.encoder.head = torch.nn.Identity()
        self.patch_size = self.encoder.patch_embed.patch_size
        self.shifts = nn.Parameter(torch.tensor([0.0
        ]), requires_grad=False)
        self.scales = nn.Parameter(torch.tensor([1.0
        ]), requires_grad=False)

    def forward(self, x):
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, (224, 224), mode='bicubic')
        b, c, h, w = x.shape
        patch_num_h, patch_num_w = h//self.patch_size[0], w//self.patch_size[1]
        feature = self.encoder.forward_features(x)[:, self.encoder.num_prefix_tokens:]
        feature = feature.transpose(1, 2)
        feature = feature.view(b, -1, patch_num_h, patch_num_w).contiguous()
        feature = (feature - self.shifts.view(1, -1, 1, 1)) / self.scales.view(1, -1, 1, 1)
        return feature

class MAE(nn.Module):
    def __init__(self, model_id, weight_path:str):
        super(MAE, self).__init__()
        if os.path.isdir(weight_path):
            weight_path = os.path.join(weight_path, "pytorch_model.bin")
        self.encoder = timm.create_model(
            model_id,
            checkpoint_path=weight_path,
            num_classes=0,
        )
        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        self.encoder.head = torch.nn.Identity()
        self.patch_size = self.encoder.patch_embed.patch_size
        self.shifts = nn.Parameter(torch.tensor([0.0
        ]), requires_grad=False)
        self.scales = nn.Parameter(torch.tensor([1.0
        ]), requires_grad=False)

    def forward(self, x):
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, (224, 224), mode='bicubic')
        b, c, h, w = x.shape
        patch_num_h, patch_num_w = h//self.patch_size[0], w//self.patch_size[1]
        feature = self.encoder.forward_features(x)[:, self.encoder.num_prefix_tokens:]
        feature = feature.transpose(1, 2)
        feature = feature.view(b, -1, patch_num_h, patch_num_w).contiguous()
        feature = (feature - self.shifts.view(1, -1, 1, 1)) / self.scales.view(1, -1, 1, 1)
        return feature

class DINO(nn.Module):
    def __init__(self, model_id, weight_path:str):
        super(DINO, self).__init__()
        if os.path.isdir(weight_path):
            weight_path = os.path.join(weight_path, "pytorch_model.bin")
        self.encoder = timm.create_model(
                model_id,
                checkpoint_path=weight_path,
                num_classes=0,
            )
        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        self.encoder.head = torch.nn.Identity()
        self.patch_size = self.encoder.patch_embed.patch_size
        self.shifts = nn.Parameter(torch.tensor([ 0.0,
        ]), requires_grad=False)
        self.scales = nn.Parameter(torch.tensor([ 1.0,
        ]), requires_grad=False)

    def forward(self, x):
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, (224, 224), mode='bicubic')
        b, c, h, w = x.shape
        patch_num_h, patch_num_w = h//self.patch_size[0], w//self.patch_size[1]
        feature = self.encoder.forward_features(x)[:, self.encoder.num_prefix_tokens:]
        feature = feature.transpose(1, 2)
        feature = feature.view(b, -1, patch_num_h, patch_num_w).contiguous()
        feature = (feature - self.shifts.view(1, -1, 1, 1)) / self.scales.view(1, -1, 1, 1)
        return feature

class CLIP(nn.Module):
    def __init__(self, model_id, weight_path:str):
        super(CLIP, self).__init__()
        self.encoder = transformers.CLIPVisionModel.from_pretrained(weight_path)
        self.patch_size = self.encoder.vision_model.embeddings.patch_embedding.kernel_size
        self.shifts = nn.Parameter(torch.tensor([0.0,
        ]), requires_grad=False)
        self.scales = nn.Parameter(torch.tensor([1.0,
        ]), requires_grad=False)

    def forward(self, x):
        x = Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)(x)
        x = torch.nn.functional.interpolate(x, (224, 224), mode='bicubic')
        b, c, h, w = x.shape
        patch_num_h, patch_num_w = h//self.patch_size[0], w//self.patch_size[1]
        feature = self.encoder(x)['last_hidden_state'][:, 1:]
        feature = feature.transpose(1, 2)
        feature = feature.view(b, -1, patch_num_h, patch_num_w).contiguous()
        feature = (feature - self.shifts.view(1, -1, 1, 1)) / self.scales.view(1, -1, 1, 1)
        return feature



class DINOv2(nn.Module):
    def __init__(self, model_id, weight_path:str):
        super(DINOv2, self).__init__()
        self.encoder = transformers.Dinov2Model.from_pretrained(weight_path)
        self.patch_size = self.encoder.embeddings.patch_embeddings.projection.kernel_size

    def forward(self, x):
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, (224, 224), mode='bicubic')
        b, c, h, w = x.shape
        patch_num_h, patch_num_w = h//self.patch_size[0], w//self.patch_size[1]
        feature = self.encoder.forward(x)['last_hidden_state'][:, 1:]
        feature = feature.transpose(1, 2)
        feature = feature.view(b, -1, patch_num_h, patch_num_w).contiguous()
        return feature