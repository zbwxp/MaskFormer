import torch
import torch.nn as nn
from detectron2.config import configurable

from .transformer_predictor import TransformerPredictor
from .decoder_attn import *

class TPNPredictor(TransformerPredictor):
    @configurable
    def __init__(self, is_reverse, **kwargs):
        super().__init__(**kwargs)
        self.is_reverse = is_reverse
        dim = kwargs["mask_dim"]
        num_heads = kwargs['nheads']
        num_expand_layer = 3
        num_queries = kwargs['num_queries']
        decoder_layer = TPN_DecoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim * 4)
        up_decoder = []
        for i in range(4):
            decoder = TPN_Decoder(decoder_layer, num_expand_layer)
            self.add_module("decoder_{}".format(i + 1), decoder)
            up_decoder.append(decoder)

        self.up_decoder = up_decoder

        self.q = nn.Embedding(num_queries, dim)

    def forward(self, x, mask_features):
        maps = []
        maps_size = []
        if mask_features[0].dim() > 3:
            for map in mask_features:
                maps_size.append(map.size()[-2:])
                maps.append(map.flatten(-2).permute(-1, 0, 1))
        else:
            maps = mask_features
        bs = maps[0].size()[1]
        map_outs = []
        attns = []
        map_out = self.q.weight.repeat(bs, 1, 1).transpose(0, 1)
        num_query = self.q.num_embeddings
        if self.is_reverse:
            maps.reverse()
            maps_size.reverse()
        for map, decoder, map_size in zip(maps, self.up_decoder, maps_size):
            map_out, attn = decoder(map_out, map)
            map_outs.append(map_out.transpose(0, 1))
            attns.append(attn.reshape(bs, num_query, map_size[0], map_size[1]))

        map_outs = torch.stack(map_outs, dim=0)
        if self.mask_classification:
            outputs_class = self.class_embed(map_outs)
            out = {"pred_logits": outputs_class[-1]}
        else:
            out = {}
        if self.is_reverse:
            size = maps_size[0]
        else:
            size = maps_size[-1]
        outputs_seg_masks = []
        for i_attn, attn in enumerate(attns):
            # if i_attn == 0:
            #     outputs_seg_masks.append(F.interpolate(attn, size=size, mode='bilinear', align_corners=False))
            # else:
            #     outputs_seg_masks.append(outputs_seg_masks[i_attn-1] +
            #                     F.interpolate(attn, size=size, mode='bilinear', align_corners=False))
            outputs_seg_masks.append(F.interpolate(attn, size=size, mode='bilinear', align_corners=False))

        if self.aux_loss:
            # [l, bs, queries, embed]
            outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)
            out["pred_masks"] = outputs_seg_masks[-1]
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class if self.mask_classification else None, outputs_seg_masks
            )
        else:
            # FIXME h_boxes takes the last one computed, keep this in mind
            # [bs, queries, embed]
            out["pred_masks"] = outputs_seg_masks[-1]
        return out
