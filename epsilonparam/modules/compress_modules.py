import torch.nn as nn
from .network_components import ResnetBlock, VBRCondition, FlexiblePrior, Downsample, Upsample, GDN1
from .utils import quantize, NormalDistribution


class Compressor(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 3),
        hyper_dims_mults=(3, 3, 3),
        channels=3,
        out_channels=3,
        vbr=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.dims = [channels, *map(lambda m: dim * m, dim_mults)]
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))
        self.reversed_dims = list(reversed([out_channels, *map(lambda m: dim * m, dim_mults)]))
        self.reversed_in_out = list(zip(self.reversed_dims[:-1], self.reversed_dims[1:]))
        self.hyper_dims = [self.dims[-1], *map(lambda m: dim * m, hyper_dims_mults)]
        self.hyper_in_out = list(zip(self.hyper_dims[:-1], self.hyper_dims[1:]))
        self.reversed_hyper_dims = list(
            reversed([self.dims[-1] * 2, *map(lambda m: dim * m, hyper_dims_mults)])
        )
        self.reversed_hyper_in_out = list(
            zip(self.reversed_hyper_dims[:-1], self.reversed_hyper_dims[1:])
        )
        self.vbr = vbr
        self.prior = FlexiblePrior(self.hyper_dims[-1])

    def get_extra_loss(self):
        return self.prior.get_extraloss()

    def build_network(self):
        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

    def encode(self, input, cond=None):
        for i, (resnet, vbrscaler, down) in enumerate(self.enc):
            input = resnet(input)
            if self.vbr:
                input = vbrscaler(input, cond)
            input = down(input)
        latent = input
        for i, (conv, vbrscaler, act) in enumerate(self.hyper_enc):
            input = conv(input)
            if self.vbr and i != (len(self.hyper_enc) - 1):
                input = vbrscaler(input, cond)
            input = act(input)
        hyper_latent = input
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        input = q_hyper_latent
        for i, (deconv, vbrscaler, act) in enumerate(self.hyper_dec):
            input = deconv(input)
            if self.vbr and i != (len(self.hyper_dec) - 1):
                input = vbrscaler(input, cond)
            input = act(input)

        mean, scale = input.chunk(2, 1)
        latent_distribution = NormalDistribution(mean, scale.clamp(min=0.1))
        q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        state4bpp = {
            "latent": latent,
            "hyper_latent": hyper_latent,
            "latent_distribution": latent_distribution,
        }
        return q_latent, q_hyper_latent, state4bpp

    def decode(self, input, cond=None):
        output = []
        for i, (resnet, vbrscaler, down) in enumerate(self.dec):
            input = resnet(input)
            if self.vbr:
                input = vbrscaler(input, cond)
            input = down(input)
            output.append(input)
        return output[::-1]

    def bpp(self, shape, state4bpp):
        B, _, H, W = shape
        latent = state4bpp["latent"]
        hyper_latent = state4bpp["hyper_latent"]
        latent_distribution = state4bpp["latent_distribution"]
        if self.training:
            q_hyper_latent = quantize(hyper_latent, "noise")
            q_latent = quantize(latent, "noise")
        else:
            q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
            q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        hyper_rate = -self.prior.likelihood(q_hyper_latent).log2()
        cond_rate = -latent_distribution.likelihood(q_latent).log2()
        bpp = (hyper_rate.sum(dim=(1, 2, 3)) + cond_rate.sum(dim=(1, 2, 3))) / (H * W)
        return bpp

    def forward(self, input, cond=None):
        q_latent, q_hyper_latent, state4bpp = self.encode(input, cond)
        bpp = self.bpp(input.shape, state4bpp)
        output = self.decode(q_latent, cond)
        return {
            "output": output,
            "bpp": bpp,
            "q_latent": q_latent,
            "q_hyper_latent": q_hyper_latent,
        }


class BigCompressor(Compressor):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 3, 3, 3),
        hyper_dims_mults=(3, 3, 3),
        channels=3,
        out_channels=3,
        vbr=False,
    ):
        super().__init__(dim, dim_mults, hyper_dims_mults, channels, out_channels, vbr)
        self.build_network()

    def build_network(self):

        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, None, True if ind == 0 else False),
                        VBRCondition(1, dim_out) if self.vbr else nn.Identity(),
                        Downsample(dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            self.dec.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out if not is_last else dim_in),
                        VBRCondition(1, dim_out if not is_last else dim_in)
                        if self.vbr
                        else nn.Identity(),
                        Upsample(dim_out if not is_last else dim_in, dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if ind == 0
                        else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if is_last
                        else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )


class SimpleCompressor(Compressor):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 3),
        hyper_dims_mults=(3, 3, 3),
        channels=3,
        out_channels=3,
        vbr=False,
    ):
        super().__init__(dim, dim_mults, hyper_dims_mults, channels, out_channels, vbr)
        self.build_network()

    def build_network(self):

        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        GDN1(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            self.dec.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        GDN1(dim_out, True) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if ind == 0
                        else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_dec.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if is_last
                        else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )
