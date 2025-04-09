import functools
from typing import Tuple
import torch
import torch.nn as nn
import math

from torch.nn.init import zeros_
from torch.nn.modules.module import T

# from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn.functional import scaled_dot_product_attention
from src.utils.model_loader import ModelLoader
from src.utils.no_grad import no_grad

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class Embed(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer = None,
            bias: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels,):
        embeddings = self.embedding_table(labels)
        return embeddings

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x):
        x =  self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        return x

class ResBlock(nn.Module):
    def __init__(self, dim:int, groups:int=8, hidden_dim:int=256):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, dim)
        self.norm2 = nn.GroupNorm(groups, dim)
        self.embed_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x, c):
        c = self.embed_proj(c)[:, :, None, None]
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x  = torch.nn.functional.silu(x)
        x = x * c
        x = self.conv2(x)
        x = self.norm2(x)
        x = torch.nn.functional.silu(x)
        return residual + x

def precompute_freqs_cis_2d(dim: int, height: int, width:int, theta: float = 10000.0, scale=16.0):
    # assert  H * H == end
    # flat_patch_pos = torch.linspace(-1, 1, end) # N = end
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)) # Hc/4
    x_freqs = torch.outer(x_pos, freqs).float() # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float() # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1) # N,Hc/4,2
    freqs_cis = freqs_cis.reshape(height*width, -1)
    return freqs_cis


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis[None, :, None, :]
    # xq : B N H Hc
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # B N H Hc/2
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # B, N, H, Hc
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos, mask) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B N H Hc
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)  # B, H, N, Hc
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()  # B, H, N, Hc
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()

        x = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class FlattenDiTBlock(nn.Module):
    def __init__(self, hidden_size, groups,  mlp_ratio=4.0, ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x,  c, pos, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), pos, mask=mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FlattenDiTEncoder(nn.Module):
    def __init__(
            self,
            in_channels=4,
            num_groups=12,
            hidden_size=1152,
            num_blocks=18,
            patch_size=2,
            num_classes=1000,
            learn_sigma=True,
            weight_path=None,
            load_ema=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.s_embedder = Embed(in_channels*patch_size**2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes+1, hidden_size)
        self.weight_path = weight_path

        self.load_ema = load_ema
        self.blocks = nn.ModuleList([
            FlattenDiTBlock(self.hidden_size, self.num_groups) for _ in range(self.num_blocks)
        ])
        self.initialize_weights()
        self.precompute_pos = dict()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)]
        else:
            pos_rope = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
            pos_ape = precompute_freqs_cis_2d(self.hidden_size*2, height, width).to(device)
            self.precompute_pos[(height, width)] = (pos_rope, pos_ape)
            return (pos_rope, pos_ape)

    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, t, y, mask=None, classify_layer=None):
        B, _, H, W = x.shape
        pos_rope, pos_ape = self.fetch_pos(H//self.patch_size, W//self.patch_size, x.device)
        x = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        t = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)
        y = self.y_embedder(y).view(B, 1, self.hidden_size)
        c = nn.functional.silu(t + y)
        s = self.s_embedder(x)
        # s = s + pos_ape.to(s.dtype)
        classify_feats = []
        for i in range(self.num_blocks):
            s = self.blocks[i](s, c, pos_rope, mask)
            if classify_layer is not None and i < classify_layer:
                classify_feats.append(s)
                if i == classify_layer - 1:
                    return _, classify_feats
        return None, s


class FlattenDiTDecoder(nn.Module):
    def __init__(
            self,
            in_channels=4,
            num_groups=12,
            hidden_size=1152,
            num_mid_blocks=18,
            num_res_blocks=[1, 1, 1],
            num_res_channels=[64, 384, 768],
            num_classes=1000,
            learn_sigma=True,
            weight_path=None,
            load_ema=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_mid_blocks = num_mid_blocks
        self.num_res_blocks = num_res_blocks
        self.num_res_channels = num_res_channels
        self.patch_size = 2**(len(num_res_blocks))

        self.y_embedder = LabelEmbedder(num_classes + 1, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.down_res_blocks = nn.ModuleList()
        previous_channel = self.in_channels
        for num, channels in zip(num_res_blocks, num_res_channels):
            self.down_res_blocks.append(
                nn.Conv2d(previous_channel, channels, kernel_size=2, stride=2, padding=0),
            )
            self.down_res_blocks.extend([ResBlock(channels, hidden_dim=hidden_size) for _ in range(num)])
            previous_channel = channels
        self.up_res_blocks = []
        previous_channel = self.in_channels
        for num, channels in zip(num_res_blocks, num_res_channels):
            self.up_res_blocks.append(
                nn.ConvTranspose2d(channels, previous_channel, kernel_size=2, stride=2, padding=0)
            )
            self.up_res_blocks.extend([ResBlock(channels, hidden_dim=hidden_size) for _ in range(num)])
            previous_channel = channels
        self.up_res_blocks = nn.ModuleList(self.up_res_blocks[::-1])

        self.blocks = nn.ModuleList([
            FlattenDiTBlock(self.hidden_size, self.num_groups) for _ in range(self.num_mid_blocks)
        ])

        self.initialize_weights()
        self.precompute_pos = dict()
        self.weight_path = weight_path
        self.load_ema = load_ema

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)]
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def initialize_weights(self):
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, t, y, s, mask=None):
        B, _, H, W = x.shape
        t = self.t_embedder(t.view(-1)).view(B, self.hidden_size)
        y = self.y_embedder(y).view(B, self.hidden_size)
        s = torch.nn.functional.normalize(s, dim=-1, eps=1e-6)
        c = torch.nn.functional.silu(t + y)

        residual = []
        for i, block in enumerate(self.down_res_blocks):
            if isinstance(block, nn.Conv2d):
                residual.append(x)
                x = block(x)
            else:
                x = block(x, c)

        pos = self.fetch_pos(H//self.patch_size, W//self.patch_size, x.device)
        x = x.view(B, self.hidden_size, -1).transpose(1, 2)
        mid_c = torch.nn.functional.silu(t[:, None, :] + s)
        for i in range(self.num_mid_blocks):
            x = self.blocks[i](x, mid_c, pos, None)
        x = x.transpose(1, 2).view(B, self.hidden_size, H//self.patch_size, W//self.patch_size)

        residual[0] = 0.0
        for i, block in enumerate(self.up_res_blocks):
            if isinstance(block, nn.ConvTranspose2d):
                x = block(x) + residual.pop()
            else:
                x = block(x, c)
        return x


class FlattenDiT(nn.Module):
    def __init__(
        self,
        encoder:FlattenDiTEncoder,
        decoder:FlattenDiTDecoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        ModelLoader().load(encoder)
        self.encoder = self.encoder.to(torch.bfloat16)
        no_grad(self.encoder)

    def forward(self, x, t, y, s=None, classify_layer=None):
        if s is None:
            _, s = self.encoder(x, t, y, classify_layer=classify_layer)
        if classify_layer is not None:
            return None, s
        x = self.decoder(x, t, y, s)
        return x, s

class FlattenDiT_jointtraining(nn.Module):
    def __init__(
        self,
        encoder:FlattenDiTEncoder,
        decoder:FlattenDiTDecoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, t, y, s=None):
        if s is None:
            _, s = self.encoder(x, t, y)
        x = self.decoder(x, t, y, s)
        return x, s

