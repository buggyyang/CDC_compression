import torch
from torch import nn
from .utils import exists, default
from .network_components import (
    LayerNorm,
    Residual,
    # SinusoidalPosEmb,
    Upsample,
    Downsample,
    PreNorm,
    LinearAttention,
    # Block,
    ResnetBlock,
    ImprovedSinusoidalPosEmb
)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        context_dim_mults=(1, 2, 3, 3),
        channels=3,
        context_channels=3,
        with_time_emb=True,
        embd_type="01"
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        context_dims = [context_channels, *map(lambda m: dim * m, context_dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.embd_type = embd_type

        if with_time_emb:
            if embd_type == "01":
                time_dim = dim
                self.time_mlp = nn.Sequential(nn.Linear(1, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
            elif embd_type == "index":
                time_dim = dim
                self.time_mlp = nn.Sequential(
                    ImprovedSinusoidalPosEmb(time_dim // 2),
                    nn.Linear(time_dim // 2 + 1, time_dim * 4),
                    nn.GELU(),
                    nn.Linear(time_dim * 4, time_dim)
                )
            else:
                raise NotImplementedError
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                                dim_in + context_dims[ind]
                                if (not is_last) and (ind < (len(context_dims) - 1))
                                else dim_in,
                                dim_out,
                                time_dim,
                                True if ind == 0 else False
                            ),
                        ResnetBlock(dim_out, dim_out, time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        # nn.Identity(),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        # self.mid_attn = nn.Identity()
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_dim),
                        ResnetBlock(dim_in, dim_in, time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        # nn.Identity(),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(LayerNorm(dim), nn.Conv2d(dim, out_dim, 7, padding=3))

    def encode(self, x, t, context):
        h = []
        for idx, (backbone, backbone2, attn, downsample) in enumerate(self.downs):
            x = torch.cat([x, context[idx]], dim=1) if idx < len(context) else x
            x = backbone(x, t)
            x = backbone2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        return x, h

    def decode(self, x, h, t):
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for backbone, backbone2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = backbone(x, t)
            x = backbone2(x, t)
            x = attn(x)
            x = upsample(x)
        return self.final_conv(x)

    def forward(self, x, time=None, context=None):
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        x, h = self.encode(x, t, context)
        return self.decode(x, h, t)
