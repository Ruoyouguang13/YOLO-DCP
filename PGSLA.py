import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['C2f_PGSLA', 'PGSLA']

# def get_valid_group(in_channels, max_groups=32):
#     """
#     自动返回最适合的 groups 值，使 in_channels 能被整除，尽量接近 max_groups。
#     如果无法整除，逐级减小，直到找到合法值（至少为1）。
#     """
#     for g in reversed(range(1, max_groups + 1)):
#         if in_channels % g == 0:
#             print(f"[PGSLA AutoGroup] in_channels={in_channels}, using groups={g}")
#             return g
#     return 1  # fallback

def get_valid_group(in_channels):
    """
    为小目标检测优化的自动分组策略：
    - 尽量使用较大的 group 数（小 group size）增强通道注意力精度。
    - 仅选择可被 in_channels 整除的 group。
    """
    # 更激进的通道组策略（优先尝试较大的 group 数）
    if in_channels <= 64:
        candidates = [32, 16, 8]
    elif in_channels <= 128:
        candidates = [64, 16, 32]
    elif in_channels <= 256:
        candidates = [128, 8, 16]
    else:
        candidates = [256, 128, 64]

    # if in_channels <= 64:
    #     candidates = [8, 16, 32]
    # elif in_channels <= 128:
    #     candidates = [16, 32, 64]
    # elif in_channels <= 256:
    #     candidates = [32, 64, 128]
    # else:
    #     candidates = [64, 128, 256]

    for g in candidates:
        if in_channels % g == 0:
            print(f"[PGSLA AutoGroup] in_channels={in_channels}, using groups={g} (for small object focus)")
            return g

    print(f"[PGSLA AutoGroup] fallback for in_channels={in_channels}, using groups=1")
    return 1


class PGSLA(nn.Module):
    def __init__(self, in_channels, groups=None, reduction=4):
        super(PGSLA, self).__init__()
        groups = 1
        # if groups is None:
        #     groups = get_valid_group(in_channels, max_groups=32)
        #
        # elif in_channels % groups != 0:
        #     print(f'[PGSLA Warning] in_channels={in_channels} not divisible by groups={groups}, auto-adjusting...')
        #     groups = get_valid_group(in_channels)

        self.in_channels = in_channels
        self.groups = groups
        self.inter_channels = max(in_channels // reduction, 8)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AdaptiveAvgPool2d(3)

        self.conv_global = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=groups, bias=False)
        self.conv_local = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=groups, bias=False)

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, self.inter_channels, 1, bias=False),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        b, c, h, w = x.size()

        # Global attention
        g_pool = self.global_pool(x).view(b, c, -1)  # [B, C, 1]
        g_attn = self.conv_global(g_pool).view(b, c, 1, 1)  # [B, C, 1, 1]

        # Local attention
        l_pool = self.local_pool(x).view(b, c, -1)  # [B, C, 9]
        l_attn = self.conv_local(l_pool).view(b, c, 3, 3)
        l_attn = F.interpolate(l_attn, size=(h, w), mode='bilinear', align_corners=False)  # [B, C, H, W]

        # Fuse and weight
        attn_map = self.fuse(g_attn + l_attn)
        return x * attn_map


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p if p is not None else k // 2, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C2f_PGSLA(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_PGSLA(self.c, self.c, shortcut, g) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck_PGSLA(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=1.0):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.attn = PGSLA(c2)

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        out = self.attn(out)
        return x + out if self.add else out


if __name__ == "__main__":
    attention = PGSLA(in_channels=64)
    inputs = torch.randn((2, 64, 32, 32))
    result = attention(inputs)
    print(result.shape)
