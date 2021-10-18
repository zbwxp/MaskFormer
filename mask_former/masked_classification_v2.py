# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .modeling.test_matcher import HungarianMatcher_diceonly
from .modeling.entity_matcher import HungarianMatcher_entity
from detectron2.structures import Instances, Boxes
import matplotlib.pyplot as plt
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from typing import Callable, Dict, List, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init
from .modeling.transformer.transformer_predictor import MLP
from PIL import Image
import json


@META_ARCH_REGISTRY.register()
class MaskedClassification_v2(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            sem_seg_head: nn.Module,
            criterion: nn.Module,
            num_queries: int,
            panoptic_on: bool,
            object_mask_threshold: float,
            overlap_threshold: float,
            metadata,
            size_divisibility: int,
            sem_seg_postprocess_before_inference: bool,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            entity_test: bool,
            entity: bool,
            conv_dim: int,
            mask_dim: int,
            norm: Optional[Union[str, Callable]] = None,
            num_classes: int,
            use_pred_loss: bool,
            iter_matcher: bool,
            iter_loss: bool,
            pixel_decoder_name: str,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            panoptic_on: bool, whether to output panoptic segmentation prediction
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        # self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.panoptic_on = panoptic_on
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.entity_test_on = entity_test
        self.entity = entity
        self.iter_matcher = iter_matcher
        self.iter_loss = iter_loss
        self.register_buffer("_iter", torch.zeros([1]))
        self._warmup_iters = 4000

        self.pool = nn.AdaptiveAvgPool2d(1)
        if pixel_decoder_name =="ClsDecoder_light":
            self.classifier = MLP(480, 256, num_classes, 3)
        else:
            self.classifier = MLP(1024, 1024, num_classes, 3)



    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        entity = cfg.MODEL.MASK_FORMER.ENTITY

        # building criterion
        if cfg.MODEL.MASK_FORMER.MATCHER == "HungarianMatcher":
            matcher = HungarianMatcher(
                cost_class=1,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
            )
        elif cfg.MODEL.MASK_FORMER.MATCHER == "EntityHungarianMatcher":
            print("use hungarian_entity matcher!!!!!!!!!!!!!!!!!")
            matcher = HungarianMatcher_entity(
                cost_class=1,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
            )
        elif cfg.MODEL.MASK_FORMER.MATCHER == "HungarianMatcher_diceonly":
            print("use hungarian_diceonly matcher!!!!!!!!!!!!!!!!!")
            matcher = HungarianMatcher_diceonly(
                cost_class=1,
                cost_mask=20.0,
                cost_dice=1.0,
            )
        else:
            print("no matcher is defined!!!")
            assert False

        weight_dict = {"loss_ce": 1, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if cfg.MODEL.MASK_FORMER.ENTITY_WEIGHT is not None:
            weight_dict.update({"loss_ce_entity": cfg.MODEL.MASK_FORMER.ENTITY_WEIGHT})
        if cfg.MODEL.MASK_FORMER.ENTITY:
            weight_dict.update({"loss_entity_cls": 1})
        if cfg.MODEL.MASK_FORMER.USE_PRED_LOSS:
            weight_dict.update({"loss_entity_cls_pred": 1})
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            entity=entity,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                    or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "entity_test": cfg.MODEL.MASK_FORMER.TEST.ENTITY_ON,
            "entity": entity,
            "conv_dim": cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            "mask_dim": cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
            "norm": cfg.MODEL.SEM_SEG_HEAD.NORM,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "use_pred_loss": cfg.MODEL.MASK_FORMER.USE_PRED_LOSS,
            "iter_matcher": cfg.MODEL.MASK_FORMER.ITER_MATCHER,
            "iter_loss": cfg.MODEL.MASK_FORMER.ITER_LOSS,
            "pixel_decoder_name": cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)


        self._iter += 1
        # mask classification target
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, images, self.training)
        else:
            targets = None

        # bipartite matching-based loss
        _iter = self._iter
        if not self.iter_matcher:
            _iter = 0.0
        masked_pool_vec = []
        masked_pool_weights = []
        for i, target in enumerate(targets):
            per_img_masks = target['masks']
            h, w = outputs.size()[-2:]
            per_img_masks = F.interpolate(per_img_masks[None, :], size=(h, w), mode='nearest')[0]
            per_img_features = outputs[i]
            masked_map = torch.einsum("qhw,chw->qchw", per_img_masks, per_img_features)
            masked_map = F.max_pool2d(masked_map, kernel_size=7, stride=4, padding=1)
            map_weights = (masked_map.sum(dim=1) > 0).flatten(-2).sum(dim=-1)
            s_h, s_w = masked_map.size()[-2:]
            area = s_h * s_w
            masked_pool = self.pool(masked_map).squeeze() * (area/(map_weights+1))[:,None]
            masked_pool_vec.append(masked_pool)

        masked_pool_vec = torch.cat(masked_pool_vec, dim=0).squeeze()
        pred_logits = self.classifier(masked_pool_vec)

        if self.training:
            labels = torch.cat([x['labels'] for x in targets])
            loss_ce = F.cross_entropy(pred_logits, labels)
            losses = {"loss_ce": loss_ce}

            return losses
        else:
            pred = {"pred_classes": pred_logits.argmax(-1)}
            return [pred, pred_logits]



    def prepare_targets(self, targets, images, is_training):
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for iter, targets_per_image in enumerate(targets):
            # pad gt
            gt_masks = targets_per_image.gt_masks.float()
            gt_classes = targets_per_image.gt_classes
            # apply aug to masks
            shirnk_mask = F.interpolate(gt_masks[None, :], scale_factor=0.125, mode='bilinear',
                                        align_corners=False)[0]
            # augs
            if is_training:
                for i, mask in enumerate(gt_masks):
                    orig_mask = mask.clone()
                    # random erosion
                    if torch.rand(1) > 0.2:
                        mask = shirnk_mask[i]
                        mask_height, mask_width = mask.size()
                        new_mask = torch.zeros_like(mask)
                        finds_y, finds_x = torch.nonzero(mask == 1, as_tuple=True)
                        if len(finds_y) == 0:
                            continue
                        x1 = torch.min(finds_x)
                        x2 = torch.max(finds_x)
                        y1 = torch.min(finds_y)
                        y2 = torch.max(finds_y)
                        if x2 - x1 == 0 or y2 - y1 == 0:
                            continue
                        width = x2 - x1
                        height = y2 - y1
                        rand1 = torch.rand(1, device=self.device)
                        rand2 = torch.rand(1, device=self.device)
                        rand3 = torch.randn(1, device=self.device) + 1
                        rand4 = torch.randn(1, device=self.device) + 1
                        rand5 = torch.rand(1, device=self.device)
                        rand6 = torch.rand(1, device=self.device) - 0.2

                        finds_y = (torch.rand(finds_y.size(), device=self.device) - 0.5 * rand3) \
                                  * height * rand1 * 0.2 + finds_y.float()
                        finds_x = (torch.rand(finds_x.size(), device=self.device) - 0.5 * rand4) \
                                  * width * rand2 * 0.2 + finds_x.float()

                        finds_y[finds_y > mask_height - 1] = mask_height - 1
                        finds_x[finds_x > mask_width - 1] = mask_width - 1

                        new_mask[finds_y.long(), finds_x.long()] = 1
                        new_mask += 0.2 * rand5 * mask[0]
                        scale_factor = 0.25
                        if torch.rand(1) > 0.5:
                            scale_factor *= 2

                        shirnk = F.interpolate(new_mask[None, None, :], scale_factor=scale_factor, mode='bilinear',
                                               align_corners=False)
                        expand = F.interpolate(shirnk, orig_mask.size()[-2:], mode='bilinear', align_corners=False)
                        new_mask = expand[0, 0] + 0.2 * rand6 * orig_mask
                        new_mask = (new_mask > 0.5).float()
                        if new_mask.sum() < 64:
                            continue
                        mask = new_mask

                    if torch.rand(1) > 0.5:
                        shirnk = F.interpolate(mask[None, None, :], scale_factor=0.125, mode='bilinear', align_corners=False)
                        expand = F.interpolate(shirnk, orig_mask.size()[-2:], mode='bilinear', align_corners=False)
                        mask = (expand[0, 0] > 0.5).float()
                        if mask.sum() < 64:
                            continue

                    gt_masks[i] = mask

                    # f, axarr = plt.subplots(2, 2)
                    # axarr[0, 0].imshow(orig_mask.to('cpu'))
                    # axarr[0, 1].imshow(gt_masks[i].to('cpu'))
                    # axarr[1, 1].imshow(images.tensor[iter].permute(1, 2, 0).cpu())
                    # print()

            padded_masks = torch.zeros((gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            # elimate too small masks
            areas = padded_masks.flatten(-2).sum(-1)
            if is_training:
                keep = areas > 16
                gt_classes = gt_classes[keep]
                padded_masks = padded_masks[keep]

            new_targets.append(
                {
                    "labels": gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets