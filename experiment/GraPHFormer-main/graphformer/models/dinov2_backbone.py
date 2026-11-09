"""
Minimal DINOv2 ViT backbone.

The released ImageEncoder loads DINOv2 through ``torch.hub`` which downloads
model code from GitHub at run time. The pretrained weights themselves are
already stored inside the released GraPHFormer checkpoints, so this module
re-implements the architecture locally with exactly the same parameter names
(DINOv2 state dict layout) and no network dependency.

Supported variants: vits14 (384/12 blocks), vitb14 (768/12), vitl14 (1024/24),
vitg14 (1536/24). Only the architecture is defined here; weights are always
loaded from the GraPHFormer checkpoint.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_VARIANT_SPECS = {
    "dinov2_vits14": dict(embed_dim=384, depth=12, num_heads=6, patch_size=14),
    "dinov2_vitb14": dict(embed_dim=768, depth=12, num_heads=12, patch_size=14),
    "dinov2_vitl14": dict(embed_dim=1024, depth=24, num_heads=16, patch_size=14),
    "dinov2_vitg14": dict(embed_dim=1536, depth=24, num_heads=24, patch_size=14),
}


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1.0):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0,
                 attn_drop=0.0, init_values=1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=518, patch_size=14, in_chans=3, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class DINOv2Backbone(nn.Module):
    """DINOv2 ViT whose forward returns the (normalised) CLS token."""

    def __init__(self, variant="dinov2_vits14", img_size=518, init_values=1.0):
        super().__init__()
        spec = _VARIANT_SPECS[variant]
        embed_dim, depth = spec["embed_dim"], spec["depth"]
        self.embed_dim = embed_dim
        self.patch_size = spec["patch_size"]
        self.num_features = embed_dim

        num_patches = (img_size // self.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.patch_embed = PatchEmbed(img_size, self.patch_size, 3, embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, spec["num_heads"], init_values=init_values)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self._init_weights()

    def _init_weights(self):
        trunc_normal = lambda t, std=0.02: nn.init.trunc_normal_(t, std=std)
        trunc_normal(self.cls_token)
        trunc_normal(self.mask_token)
        trunc_normal(self.pos_embed)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                trunc_normal(m.weight)
                nn.init.zeros_(m.bias)

    def interpolate_pos_encoding(self, x, w, h):
        """Bicubic interpolation of the stored positional embedding (DINOv2 style)."""
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # keep interpolation numerically stable for odd shifts
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=True,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, W, H)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, x):
        return self.forward_features(x)[:, 0]
