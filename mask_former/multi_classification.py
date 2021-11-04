# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from fvcore.nn import sigmoid_focal_loss_jit

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
from skimage.morphology import erosion, dilation


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

@META_ARCH_REGISTRY.register()
class MultiClassification(nn.Module):
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
            num_cls: int,
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

        self.num_cls = num_cls

        self.register_buffer("_iter", torch.zeros([1]))

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        losses = ["labels"]
        return {
            "backbone": backbone,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "num_cls": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
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

            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_tgt = targets.sum()
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_tgt)
            num_tgt = torch.clamp(num_tgt / get_world_size(), min=1).item()


            pred_logits = outputs["linear"]
            tgt = targets

            loss_fce = sigmoid_focal_loss_jit(
                pred_logits,
                tgt,
                alpha=0.25,
                gamma=2.0,
                reduction="sum"
            ) / num_tgt

            losses = {"loss_fce": loss_fce}

            return losses
        else:
            pred = {"pred_classes": outputs["linear"]}
            return [pred, ]

    def prepare_targets(self, targets, images):
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            one_hot = torch.zeros(self.num_cls, device=self.device)
            one_hot[targets_per_image.gt_classes] = 1
            new_targets.append(one_hot)

        new_targets = torch.stack(new_targets)
        return new_targets
