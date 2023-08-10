import torch
import torch.nn as nn
from .network_components import ResnetBlock, LinearAttention, LayerNorm, Downsample, Upsample
import torch.nn.functional as F
from torch.distributions.normal import Normal

class Encoder(nn.Module):
    def __init__(self, *, ch=64, z_channels=64, ch_mult=(1,2,4,8), num_res_blocks=2, in_channels=3):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(dim=block_in,
                                         dim_out=block_out,
                                         time_emb_dim=None,
                                         large_filter=True if (i_level == 0) and (i_block == 0) else False))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(dim=block_in,
                                       dim_out=block_in,
                                       time_emb_dim=None,
                                       large_filter=False)
        self.mid.attn_1 = LinearAttention(block_in)
        self.mid.block_2 = ResnetBlock(dim=block_in,
                                       dim_out=block_in,
                                       time_emb_dim=None,
                                       large_filter=False)

        # end
        self.norm_out = LayerNorm(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = F.leaky_relu(h, 0.2)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch=64, out_ch=3, ch_mult=(1,2,4,8), num_res_blocks=2, z_channels=64):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch*ch_mult[self.num_resolutions-1]

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(dim=block_in,
                                       dim_out=block_in,
                                       time_emb_dim=None,
                                       large_filter=False)
        self.mid.attn_1 = LinearAttention(block_in)
        self.mid.block_2 = ResnetBlock(dim=block_in,
                                       dim_out=block_in,
                                       time_emb_dim=None,
                                       large_filter=False)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(dim=block_in,
                                         dim_out=block_out,
                                         time_emb_dim=None,
                                         large_filter=False))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, block_in)
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = LayerNorm(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end

        h = self.norm_out(h)
        h = F.leaky_relu(h, 0.2)
        h = self.conv_out(h)
        return h
    

class AutoencoderKL(nn.Module):
    def __init__(self, ch=64, z_channels=64, ch_mult=(1,2,4,8), num_res_blocks=2, img_ch=3
                 ):
        super().__init__()
        self.encoder = Encoder(ch=ch, z_channels=z_channels, ch_mult=ch_mult, num_res_blocks=num_res_blocks, in_channels=img_ch)
        self.decoder = Decoder(ch=ch, z_channels=z_channels, ch_mult=ch_mult, num_res_blocks=num_res_blocks, out_ch=img_ch)

    def encode(self, x):
        m, s = self.encoder(x).chunk(2, dim=1)
        posterior = Normal(m, s.exp())
        return posterior

    def decode(self, z):
        dec = self.decoder(z)
        return dec

    def forward(self, input):
        posterior = self.encode(input)
        if self.training:
            z = posterior.rsample()
        else:
            z = posterior.mode
        dec = self.decode(z)
        return dec, posterior