import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

def compute_locations_per_level(h, w):
    shifts_x = torch.arange(
        0, 1, step=1 / w,
        dtype=torch.float32, device='cuda'
    )
    shifts_y = torch.arange(
        0, 1, step=1 / h,
        dtype=torch.float32, device='cuda'
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    locations = torch.stack((shift_x, shift_y), dim=0)
    return locations


@SEM_SEG_HEADS_REGISTRY.register()
class NRDDecoder(nn.Module):
    @configurable
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],
            *,
            conv_dim: int,
            mask_dim: int,
            norm: Optional[Union[str, Callable]] = None,
    ):
        super().__init__()

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_channels = [v.channels for k, v in input_shape]

        self.mask_dim = mask_dim
        norm = "SyncBN"
        # norm = "BN"
        use_bias = norm == ""
        # num_classes as 256
        num_out_channel = (2 + 16) * 16 + 16 + 16 * 16 + 16 + 16 * 256 + 256
        channel = 512
        self.dyn_ch = 16
        self.mask_ch = 16
        self.upsample_f = 8
        self.use_low_level_info = True
        self.pad_out_channel = 256

        self.bottleneck = Conv2d(
            feature_channels[-1],
            channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=get_norm(norm, channel),
            activation=F.relu,
        )
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            Conv2d(
                channel * 2,
                channel * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, channel * 2),
                activation=F.relu,
            ),
            nn.Conv2d(channel * 2, num_out_channel, 1)
        )
        nn.init.xavier_normal_(self.classifier[-1].weight)
        param = self.classifier[-1].weight / num_out_channel
        self.classifier[-1].weight = nn.Parameter(param)
        nn.init.constant_(self.classifier[-1].bias, 0)

        self.c1_bottleneck = nn.Sequential(
            Conv2d(
                feature_channels[0],
                48,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, 48),
                activation=F.relu,
            ),
            Conv2d(48, 16, 1, ),
        )
        nn.init.xavier_normal_(self.c1_bottleneck[-1].weight)
        nn.init.constant_(self.c1_bottleneck[-1].bias, 0)

        self.cat_norm = get_norm(norm, 16 + 2)
        nn.init.constant_(self.cat_norm.weight, 1)
        nn.init.constant_(self.cat_norm.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        return ret

    def forward_features(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        x = features["res5"]
        h, w = x.size()[-2:]
        x = self.bottleneck(x)
        gp = self.image_pool(x).repeat(1, 1, h, w)
        x = torch.cat((x, gp), dim=1)
        x = self.classifier(x)
        c1 = self.c1_bottleneck(features["res2"])
        output = self.interpolate_fast(x, c1, self.cat_norm)
        return output, None

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)

    def interpolate_fast(self, x, x_cat=None, norm=None):
        dy_ch = self.dyn_ch
        B, conv_ch, H, W = x.size()
        weights, biases = self.get_subnetworks_params_fast(x, channels=dy_ch)
        f = self.upsample_f
        self.coord_generator(H, W)
        coord = self.coord.reshape(1, H, W, 2, f, f).permute(0, 3, 1, 4, 2, 5).reshape(1, 2, H * f, W * f)
        coord = coord.repeat(B, 1, 1, 1)
        if x_cat is not None:
            coord = torch.cat((coord, x_cat), 1)
            coord = norm(coord)

        output = self.subnetworks_forward_fast(coord, weights, biases, B * H * W)
        return output

    def get_subnetworks_params_fast(self, attns, num_bases=0, channels=16):
        assert attns.dim() == 4
        B, conv_ch, H, W = attns.size()
        if self.use_low_level_info:
            num_bases = self.mask_ch
        else:
            num_bases = 0

        w0, b0, w1, b1, w2, b2 = torch.split_with_sizes(attns, [
            (2 + num_bases) * channels, channels,
            channels * channels, channels,
            channels * self.pad_out_channel, self.pad_out_channel
        ], dim=1)

        # out_channels x in_channels x 1 x 1
        w0 = F.interpolate(w0, scale_factor=self.upsample_f, mode='nearest')
        b0 = F.interpolate(b0, scale_factor=self.upsample_f, mode='nearest')
        w1 = F.interpolate(w1, scale_factor=self.upsample_f, mode='nearest')
        b1 = F.interpolate(b1, scale_factor=self.upsample_f, mode='nearest')
        w2 = F.interpolate(w2, scale_factor=self.upsample_f, mode='nearest')
        b2 = F.interpolate(b2, scale_factor=self.upsample_f, mode='nearest')

        return [w0, w1, w2], [b0, b1, b2]

    def subnetworks_forward_fast(self, inputs, weights, biases, n_subnets):
        assert inputs.dim() == 4
        n_layer = len(weights)
        x = inputs
        if self.use_low_level_info:
            num_bases = self.mask_ch
        else:
            num_bases = 0
        for i, (w, b) in enumerate(zip(weights, biases)):
            if i == 0:
                x = self.padconv(x, w, b, cin=2 + num_bases, cout=self.dyn_ch, relu=True)
            if i == 1:
                x = self.padconv(x, w, b, cin=self.dyn_ch, cout=self.dyn_ch, relu=True)
            if i == 2:
                x = self.padconv(x, w, b, cin=self.dyn_ch, cout=self.pad_out_channel, relu=False)
        return x

    def padconv(self, input, w, b, cin, cout, relu):
        input = input.repeat(1, cout, 1, 1)
        x = input * w
        conv_w = torch.ones((cout, cin, 1, 1), device=input.device)
        x = F.conv2d(
            x, conv_w, stride=1, padding=0,
            groups=cout
        )
        x = x + b
        if relu:
            x = F.relu(x)
        return x

    def coord_generator(self, height, width):
        f = self.upsample_f
        coord = compute_locations_per_level(f, f)
        H = height
        W = width
        coord = coord.repeat(H * W, 1, 1, 1)
        self.coord = coord.to(device='cuda')

    def compute_locations_per_level(h, w):
        shifts_x = torch.arange(
            0, 1, step=1 / w,
            dtype=torch.float32, device='cuda'
        )
        shifts_y = torch.arange(
            0, 1, step=1 / h,
            dtype=torch.float32, device='cuda'
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        locations = torch.stack((shift_x, shift_y), dim=0)
        return locations
