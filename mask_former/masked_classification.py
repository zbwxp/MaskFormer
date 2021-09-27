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
from .modeling.dual_criterion import SetDualCriterion
from .modeling.backbone.resnet_150cls import build_resnet_classification_backbone
from detectron2.layers import ShapeSpec


@META_ARCH_REGISTRY.register()
class MaskedClassification(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            size_divisibility: int,
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
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.register_buffer("_iter", torch.zeros([1]))

    @classmethod
    def from_config(cls, cfg):
        input_shape = ShapeSpec(channels=3 + 1)
        backbone = build_resnet_classification_backbone(cfg, input_shape)
        losses = ["labels"]
        return {
            "backbone": backbone,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
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
        if self.training:
            masks = [x["instances"].gt_masks.to(self.device) for x in batched_inputs]
            masks = ImageList.from_tensors(masks, self.size_divisibility)
            input_images = images.tensor
            input_masks = masks.tensor.float()
            inputs = torch.cat((input_images, input_masks), dim=1)
        else:
            input_images = images.tensor
            masks = (batched_inputs[0]["sem_seg"] == batched_inputs[0]["category"]).to(self.device)
            masks = ImageList.from_tensors([masks], self.size_divisibility)
            input_masks = masks.tensor.float()
            inputs = torch.cat((input_images, input_masks[None, :]), dim=1)

        features = self.backbone(inputs)
        outputs = features

        if self.training:
            self._iter += 1
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            # _iter = self._iter

            pred_logits = outputs["linear"]
            tgt = torch.cat([x["labels"] for x in targets])

            loss_ce = F.cross_entropy(pred_logits, tgt)
            losses = {"loss_ce": loss_ce}

            return losses
        else:
            pred = {"pred_classes": outputs["linear"].argmax()}
            return [pred, ]

    def prepare_targets(self, targets, images):
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                }
            )
        return new_targets
