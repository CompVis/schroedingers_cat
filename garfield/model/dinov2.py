import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from einops import rearrange


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(3072, 768)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 768, kernel_size=14, stride=14)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(768, 2304)
        self.proj = nn.Linear(768, 768)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, 12, 64).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, scale=0.125)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class LayerScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(768))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(768, eps=1e-6)
        self.attn = Attention()
        self.ls1 = LayerScale()
        self.norm2 = nn.LayerNorm(768, eps=1e-6)
        self.mlp = Mlp()
        self.ls2 = LayerScale()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoVisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, 768))
        self.register_tokens = nn.Parameter(torch.zeros(1, 4, 768))
        self.blocks = nn.ModuleList([Block() for _ in range(12)])
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.mask_token = nn.Parameter(torch.zeros(1, 768))

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)

        cls_pos = self.pos_embed[:, :1]
        patch_pos = F.interpolate(
            self.pos_embed[:, 1:].float().reshape(1, 37, 37, 768).permute(0, 3, 1, 2),
            size=(16, 16),
            mode="bicubic",
            antialias=True,
        ).permute(0, 2, 3, 1).reshape(1, 256, 768)
        x = x + torch.cat((cls_pos, patch_pos), dim=1).to(x.dtype)
        x = torch.cat((x[:, :1], self.register_tokens.expand(B, -1, -1), x[:, 1:]), dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 5:]


def better_resize(imgs):
    H, W = imgs.shape[-2:]
    side = min(H, W)
    imgs = TF.center_crop(imgs, [side, side])
    if (factor := side // 224) > 1:
        imgs = F.avg_pool2d(imgs, factor)
    return F.interpolate(imgs, [224, 224], mode="bilinear")


class MinDinoV2Reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.resize = 224
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*xFormers.*")
            original = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14_reg", verbose=False
            )
        self.model = DinoVisionTransformer()
        self.model.load_state_dict(original.state_dict(), strict=True)

    def forward(self, imgs):
        imgs = better_resize(imgs.movedim(-1, 1))
        imgs = (imgs + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device)[:, None, None]
        patches = self.model((imgs - mean) / std)
        return rearrange(patches, "b (h w) d -> b h w d", h=16, w=16)
