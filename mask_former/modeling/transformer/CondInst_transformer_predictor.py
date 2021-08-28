# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer
from .conv_with_kaiming_uniform import conv_with_kaiming_uniform


def compute_locations(h, w, stride, device):
    shifts_x = (torch.arange(
        0, w, step=stride,
        dtype=torch.float32, device=device
    ) + stride / 2) / w
    shifts_y = (torch.arange(
        0, h, step=stride,
        dtype=torch.float32, device=device
    ) + stride / 2) / h
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1)
    return locations


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 4
    assert len(weight_nums) == len(bias_nums)
    assert params.size(-1) == sum(weight_nums) + sum(bias_nums)

    num_heads, b, num_insts, _ = params.size()

    num_insts = params.size(-2)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=-1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_heads, b * num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_heads, b * num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_heads, b * num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_heads, b * num_insts)

    return weight_splits, bias_splits


class CondInstTransformerPredictor(nn.Module):
    @configurable
    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dropout: float,
            dim_feedforward: int,
            enc_layers: int,
            dec_layers: int,
            pre_norm: bool,
            deep_supervision: bool,
            mask_dim: int,
            enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dropout: dropout in Transformer
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            deep_supervision: whether to add supervision to every decoder layers
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = Conv2d(in_channels, hidden_dim, kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()
        self.aux_loss = deep_supervision

        # output FFNs
        if self.mask_classification:
            # +2 for relative coordinates
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1 + 2)
        self.mask_embed = MLP(hidden_dim, hidden_dim, 169, 3)

        self.low_level_in_channels = 8
        self.num_outputs = 1
        self.dyn_channels = 8
        self.weight_nums = [80, 64, 8]
        self.bias_nums = [8, 8, 1]
        conv_block = conv_with_kaiming_uniform('GN', activation=True)
        tower = []
        channels = 128
        tower.append(conv_block(
            hidden_dim, channels, 3, 1
        ))
        tower.append(conv_block(
            channels, channels, 3, 1
        ))
        tower.append(conv_block(
            channels, channels, 3, 1
        ))
        tower.append(nn.Conv2d(
            channels, self.low_level_in_channels, 1
        ))
        self.add_module('tower', nn.Sequential(*tower))

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["enc_layers"] = cfg.MODEL.MASK_FORMER.ENC_LAYERS
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["deep_supervision"] = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret

    def forward(self, x, mask_features):
        pos = self.pe_layer(x)

        src = x
        mask = None
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)

        if self.mask_classification:
            outputs_class = self.class_embed(hs)
            out = {"pred_logits": outputs_class[-1][:, :, :-2]}
            m = nn.Sigmoid()
            outputs_coord = m(outputs_class[:, :, :, -2:])
            out["pred_coords"] = outputs_coord[-1]
            outputs_class = outputs_class[:, :, :, :-2]
        else:
            out = {}

        b, c, h, w = mask_features.size()
        n_inst = hs.size()[2]
        mask_features = self.tower(mask_features)
        locations = compute_locations(h, w, stride=1, device=mask_features.device)
        locations = locations.reshape(1, 1, -1, 2).repeat(b, 1, 1, 1)

        if self.aux_loss:
            # [l, bs, queries, embed]
            mask_embed = self.mask_embed(hs)
            # outputs_seg_masks = torch.einsum("lbqc,bchw->lbqhw", mask_embed, mask_features)
            weights, biases = parse_dynamic_params(
                mask_embed, self.dyn_channels,
                self.weight_nums, self.bias_nums
            )
            num_heads = mask_embed.size()[0]
            outputs_seg_masks = []
            for i in range(num_heads):
                coords = outputs_coord[i].reshape(b, -1, 1, 2)
                mask_head_inputs = self.get_input_for_head(coords, locations, mask_features, n_inst)
                mask_logits = self.mask_heads_forward(mask_head_inputs.reshape(1, -1, h, w),
                                                      weights, biases, n_inst*b, i).reshape(b, n_inst, h, w)
                outputs_seg_masks.append(mask_logits)
            # outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)
            out["pred_masks"] = outputs_seg_masks[-1]
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class if self.mask_classification else None, outputs_seg_masks,
                outputs_coord
            )
        else:
            # FIXME h_boxes takes the last one computed, keep this in mind
            # [bs, queries, embed]
            mask_embed = self.mask_embed(hs[-1])
            # outputs_seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            weights, biases = parse_dynamic_params(
                mask_embed.unsqueeze(0), self.dyn_channels,
                self.weight_nums, self.bias_nums
            )
            coords = outputs_coord[-1].reshape(b, -1, 1, 2)
            mask_head_inputs = self.get_input_for_head(coords, locations, mask_features, n_inst)
            mask_logits = self.mask_heads_forward(mask_head_inputs.reshape(1, -1, h, w),
                                                  weights, biases, n_inst * b, -1).reshape(b, n_inst, h, w)
            outputs_seg_masks = mask_logits
            out["pred_masks"] = outputs_seg_masks
        return out

    def get_input_for_head(self, coords, locations, mask_features, n_inst):
        b, c, h, w = mask_features.size()
        relative_coords = coords - locations
        relative_coords = relative_coords.permute(0, 1, 3, 2).to(dtype=mask_features.dtype)
        mask_features = mask_features.reshape(b, self.low_level_in_channels, 1, -1) \
            .permute(0, 2, 1, 3).repeat(1, n_inst, 1, 1)
        mask_head_inputs = torch.cat([relative_coords, mask_features
                                      ], dim=2).reshape(b, n_inst, -1, h, w)
        return mask_head_inputs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_seg_coords):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_coords": c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_seg_coords[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

    def mask_heads_forward(self, features, weights, biases, num_insts, index):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w[index], bias=b[index],
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
