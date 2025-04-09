import torch
import torch.nn as nn
import math

from numba.cuda.cudadrv.devicearray import lru_cache
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn.functional import scaled_dot_product_attention

from torch.nn.attention import SDPBackend, sdpa_kernel

flex_attention = torch.compile(flex_attention)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class Embed(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer = False,
            bias: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm =  nn.Identity()
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
    def timestep_embedding(t, dim, max_period=10000):
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
        self.norm = nn.LayerNorm(hidden_size , elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm(x)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU(approximate="tanh")
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def precompute_freqs_cis_2d(dim: int, height: int, width:int, theta: float = 10000.0, scale: float=16):
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    x_pos = x_pos.reshape(-1)
    y_pos = y_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)) # Hc/4
    x_freqs = torch.outer(x_pos, freqs).float()  # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float()  # N Hc/4
    freqs_cis = torch.cat([x_freqs.sin(), x_freqs.cos(), y_freqs.sin(), y_freqs.cos()], dim=1)
    return freqs_cis


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
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
        # import pdb; pdb.set_trace()
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B N H Hc
        q = self.q_norm(q).to(q.dtype)
        k = self.k_norm(k).to(k.dtype)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)  # B, H, N, Hc
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()  # B, H, N, Hc
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()
        # x = flex_attention(q, k, v, block_mask=mask)
        # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        x = scaled_dot_product_attention(q, k, v, mask)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class DiTBlock(nn.Module):
    def __init__(self, hidden_size, groups,  mlp_ratio=4.0, ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size , elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=groups, qkv_bias=True, qk_norm=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x,  c, pos, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), pos, mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ConDiT(nn.Module):
    def __init__(
            self,
            in_channels=4,
            out_channels=4,
            num_groups=12,
            hidden_size=1152,
            num_blocks=18,
            num_cond_blocks=4,
            patch_size=2,
            num_classes=1000,
            learn_sigma=True,
            deep_supervision=0,
            weight_path=None,
            load_ema=False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.x_embedder = Embed(in_channels * patch_size ** 2, hidden_size, bias=True)
        self.s_embedder = Embed(in_channels * patch_size ** 2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes + 1, hidden_size)
        self.final_layer = FinalLayer(hidden_size, out_channels * patch_size ** 2)
        self.num_cond_blocks = num_cond_blocks


        self.weight_path = weight_path
        self.load_ema = load_ema
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_size, self.num_groups) for _ in range(self.num_blocks)
        ])
        self.initialize_weights()

    @lru_cache
    def fetch_pos(self, height, width, device):
        pos = precompute_freqs_cis_2d(self.hidden_size, height//self.patch_size, width//self.patch_size).to(device)[None, ...]
        return pos


    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, s=None):
        B, _, H, W = x.shape
        pos = self.fetch_pos(H, W, x.device)
        x = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

        t = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)
        y = self.y_embedder(y).view(B, 1, self.hidden_size)

        if s is None:
            # semantic encoder
            s = self.s_embedder(x) + pos
            c = nn.functional.silu(t + y)
            for i in range(self.num_cond_blocks):
                s = self.blocks[i](s, c, pos)
            s = nn.functional.silu(t + s)

        x = self.x_embedder(x)
        for i in range(self.num_cond_blocks, self.num_blocks):
            x = self.blocks[i](x, s, pos)
        x = self.final_layer(x, s)
        x = torch.nn.functional.fold(x.transpose(1, 2), (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return x, s
