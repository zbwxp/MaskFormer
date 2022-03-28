from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from .pixel_decoder import BasePixelDecoder
from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
import fvcore.nn.weight_init as weight_init
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
import logging

@SEM_SEG_HEADS_REGISTRY.register()
class TPNDecoder(nn.Module):
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

        lateral_convs = []
        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            lateral_norm = get_norm(norm, conv_dim)
            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            weight_init.c2_xavier_fill(lateral_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            lateral_convs.append(lateral_conv)

        self.lateral_convs = lateral_convs[::-1]

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
        laterals = []
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
        # add laterals before
            if idx:
                h, w = x.size()[-2:]
                out = lateral_conv(x) + F.interpolate(laterals[idx-1], size=(h, w), mode="nearest")
                laterals.append(out)
            else:
                laterals.append(lateral_conv(x))

        return laterals, None

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)