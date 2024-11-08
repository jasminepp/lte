import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

import numpy as np
import torch.fft
from pytorch_wavelets import DWTForward, DWTInverse


@register('vv')
class vv(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=256):
        super().__init__()
        self.encoder = models.make(encoder_spec)

        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)

        self.out0 = MLP((2 + hidden_dim) * 4 + 2, hidden_dim, 256)
        self.out1 = MLP((2 + hidden_dim) * 1 + 2, hidden_dim, 256)
        self.conv0 = qkv_attn(hidden_dim, 16)
        self.conv1 = qkv_attn(hidden_dim, 16)
        self.fc1 = MLP(hidden_dim, hidden_dim, 256)
        self.fc2 = MLP(hidden_dim, 3, 256)

    def gen_feat(self, inp):
        self.inp = inp
        self.feat_coord = make_coord(inp.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(inp.shape[0], 2, *inp.shape[-2:])

        self.feat = self.encoder(inp)
        # self.coeff = self.coef(self.feat)
        self.freqq = self.freq(self.feat)

        # self.hr = self.hrconv(self.feat)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        # coef = self.coeff
        freq = self.freqq

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = self.feat_coord

        preds = []
        areas = []
        pred1 = []
        rel_coords = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # q_freq = self.featureFusionmodule(coef, freq)
                q_freq = freq
                q_freq = F.grid_sample(
                    q_freq, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                preds.append(q_freq)
                rel_coords.append(rel_coord)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0];
        areas[0] = areas[3];
        areas[3] = t
        t = areas[1];
        areas[1] = areas[2];
        areas[2] = t

        # prepare cell
        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat.shape[-2]
        rel_cell[:, :, 1] *= feat.shape[-1]

        for index, area in enumerate(areas):
            preds[index] = preds[index] * (area / tot_area).unsqueeze(-1)
            # preds[index] = preds[index] * (torch.exp(-area)).unsqueeze(-1)

        grid0 = torch.cat([*rel_coords, *preds, rel_cell], dim=-1)

        grid_pred = F.grid_sample(
            freq, coord.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        grid1 = torch.cat([coord, grid_pred, rel_cell], dim=-1)

        x0 = self.out0(grid0)

        x1 = self.out1(grid1)

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)

        feat = x0 + x1
        result = self.fc2(feat + F.gelu(self.fc1(feat)))

        result_bil = F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear', \
                                   padding_mode='border', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        ret = result + result_bil

        # ret = 0
        # for pred, area in zip(preds, areas):
        #     ret = ret + pred * (area / tot_area).unsqueeze(-1)
        # ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
        #               padding_mode='border', align_corners=False)[:, :, 0, :] \
        #               .permute(0, 2, 1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


# class SmallScaleFeatureExtractor(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(SmallScaleFeatureExtractor, self).__init__()
#
#         self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
#
#     def forward(self, x):
#         return self.conv1x1(x)
#
#
# class LargeScaleFeatureExtractor(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(LargeScaleFeatureExtractor, self).__init__()
#
#         self.dwconv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         x = self.dwconv3x3(x)
#         return self.pointwise(x)


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, wavelet_channels):
        super(FeatureFusionModule, self).__init__()

        # mid_channels = in_channels // 2
        mid_channels = in_channels
        wavelet_channels = in_channels
        # self.small_scale_extractor = SmallScaleFeatureExtractor(in_channels, mid_channels)
        # self.large_scale_extractor = LargeScaleFeatureExtractor(in_channels, in_channels)

        self.wavelet_attention = WaveletAttention(mid_channels)

        self.feature_modulation = FeatureModulation(mid_channels, wavelet_channels, scale_factor=1)

        self.fusion_conv = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1)

        # Fix the parameters in this module
        for param in self.parameters():
            param.requires_grad = True  # False/True

    def forward(self, x, y):
        # small_scale_features = self.small_scale_extractor(x)
        # large_scale_features = self.large_scale_extractor(x)
        small_scale_features = x
        large_scale_features = y

        # cA, attn_cH, attn_cV, attn_cD = self.wavelet_attention(small_scale_features)
        #
        # recombined_features = torch.cat((cA, attn_cH, attn_cV, attn_cD), dim=1)

        recombined_features = self.wavelet_attention(small_scale_features)

        modulated_large_scale_features = self.feature_modulation(large_scale_features, recombined_features)

        combined_features = torch.cat([modulated_large_scale_features, small_scale_features], dim=1)
        # combined_features = torch.cat([large_scale_features, small_scale_features], dim=1)
        fused_features = self.fusion_conv(combined_features)

        return fused_features


class WaveletAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(WaveletAttention, self).__init__()

        self.dwt = DWT()
        # self.dwt = DWTForward(J=1, mode='zero', wave='haar')

    def forward(self, x):
        dwt_result = self.dwt(x)

        cA, cH, cV, cD = dwt_result

        # cA = F.interpolate(cA, scale_factor=2, mode='bicubic', align_corners=False)
        # cH = F.interpolate(cH, scale_factor=2, mode='bicubic', align_corners=False)
        # cV = F.interpolate(cV, scale_factor=2, mode='bicubic', align_corners=False)
        # cD = F.interpolate(cD, scale_factor=2, mode='bicubic', align_corners=False)

        combined_dwt = torch.cat((cA, cH, cV, cD), dim=1)

        combined_dwt = F.interpolate(combined_dwt.cuda(), scale_factor=2, mode='bilinear',
                                     align_corners=False)  # or bicubic bilinear

        return combined_dwt


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    min_height = min(x1.size(2), x2.size(2), x3.size(2), x4.size(2))
    min_width = min(x1.size(3), x2.size(3), x3.size(3), x4.size(3))

    x1 = x1[:, :, :min_height, :min_width]
    x2 = x2[:, :, :min_height, :min_width]
    x3 = x3[:, :, :min_height, :min_width]
    x4 = x4[:, :, :min_height, :min_width]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class ChannelAttentionModified(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModified, self).__init__()
        # Correctly initialize AdaptiveMaxPool2d to squeeze spatial dimensions to 1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Output size is (1, 1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Use max pooling to capture detail information
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(max_out)


class FeatureMapping(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureMapping, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.silu(self.conv1(x))
        x = self.conv2(x)
        return x


class FeatureModulation(nn.Module):
    def __init__(self, in_channels, wavelet_channels, scale_factor=1):
        super(FeatureModulation, self).__init__()
        self.mapping = FeatureMapping(in_channels * 4, wavelet_channels * scale_factor)
        self.scale_factor = scale_factor

    def forward(self, large_feature_map, detail_feature_map):
        # Generate modulation parameters from detail features
        modulation_params = self.mapping(detail_feature_map)

        if self.scale_factor > 1:
            modulation_params = F.interpolate(modulation_params, scale_factor=self.scale_factor, mode='bilinear')

        # Apply modulation to the large scale feature map
        desired_size = (large_feature_map.size(2), large_feature_map.size(3))
        modulation_params = F.interpolate(modulation_params, size=desired_size, mode='bilinear', align_corners=False)
        modulated_feature_map = large_feature_map * modulation_params
        return modulated_feature_map


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class qkv_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Linear(midc, midc * 3, bias=True)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.proj = nn.Linear(midc, midc)

        self.proj_drop = nn.Dropout(0.)

        self.act = nn.GELU()

    def forward(self, x):
        B, HW, C = x.shape
        bias = x

        qkv = self.qkv_proj(x).reshape(B, HW, self.heads, 3 * self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)  # B, heads, HW, headc

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2, -1), v) / (HW)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, HW, C)

        ret = v + bias
        bias = self.proj_drop(self.act(self.proj(ret))) + bias

        return bias


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out