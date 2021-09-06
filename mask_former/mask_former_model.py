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
from .utils.misc import nested_tensor_from_tensor_list

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .modeling.entity_matcher import EntityHungarianMatcher
import matplotlib.pyplot as plt


@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
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
            regress_coords: bool,
            max_iter: int,
            entity: bool,
            num_classes: int,
            hidden_dim: int,
            entity_criterion: nn.Module,
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
        self.criterion = criterion
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
        self.regress_coords = regress_coords
        self.entity = entity
        self.register_buffer("_iter", torch.zeros([1]))
        self.max_iter = max_iter

        if self.entity:
            self.cls_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, num_classes),
            )
            self.entity_criterion = entity_criterion

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        coord_weight = cfg.MODEL.MASK_FORMER.COORD_WEIGHT
        matcher_name = cfg.MODEL.MASK_FORMER.MATCHER
        entity_weight = cfg.MODEL.MASK_FORMER.ENTITY_WEIGHT
        entity = cfg.MODEL.MASK_FORMER.ENTITY

        # building criterion
        if matcher_name == "HungarianMatcher":
            matcher = HungarianMatcher(
                cost_class=1,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
            )
        elif matcher_name == "EntityHungarianMatcher":
            matcher = EntityHungarianMatcher(
                cost_class=1,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
            )
        else:
            raise AssertionError('The matcher: ', matcher_name, 'is not implemented!!')

        weight_dict = {"loss_ce": 1, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if coord_weight is not None:
            weight_dict.update({"loss_coord": coord_weight})
        if entity_weight is not None:
            weight_dict.update({"loss_ce_entity": entity_weight})
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]
        if coord_weight is not None:
            losses.append("coords")
        if matcher_name == "EntityHungarianMatcher":
            losses.remove("labels")
            losses.append("entity")

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
        )
        entity_criterion = None
        if entity:
            entity_criterion = SetCriterion(
                sem_seg_head.num_classes,
                matcher=matcher,
                weight_dict={"loss_entity_cls": 1},
                eos_coef=no_object_weight,
                losses=["entity_cls"],
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
            "regress_coords": cfg.MODEL.MASK_FORMER.REGRESS_COORDS,
            "max_iter": cfg.SOLVER.MAX_ITER,
            "entity": cfg.MODEL.MASK_FORMER.ENTITY,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "hidden_dim": cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            "entity_criterion": entity_criterion
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

        if self.training:
            self._iter += 1
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
                if self.regress_coords:
                    self.add_coords_targets(targets)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            if self.entity:
                entity_cls_logits = outputs["entity_cls_logits"]
                labels = [t["labels"] for t in targets]
                masks = [t["masks"] for t in targets]
                bs = len(masks)
                ch = entity_cls_logits.size(1)
                h, w = outputs["pred_masks"].size()[-2:]
                masked_avg_pool = []
                for b in range(bs):
                    n_inst = masks[b].size(0)
                    if n_inst == 0:
                        print("got zero instance per img!!!!!!!!!!")
                        continue
                    per_im_masks = F.interpolate(masks[b].unsqueeze(0).float(), size=[h, w],
                                                 mode="nearest")
                    per_im_mask_weights = per_im_masks.reshape(n_inst, -1).sum(dim=-1)
                    masked_avg_pool.append(((entity_cls_logits[b].unsqueeze(1)
                                             * per_im_masks).reshape(ch, n_inst, -1)
                                            .sum(dim=-1) / (per_im_mask_weights[None, :] + 1.0))
                                           .permute(1, 0))

                if not masked_avg_pool == []:
                    masked_avg_pool = torch.cat(masked_avg_pool)
                    labels = torch.cat(labels)
                    cls_preds = self.cls_head(masked_avg_pool)

                    loss_entity_cls = F.cross_entropy(cls_preds, labels)
                    losses.update({"loss_entity_cls": loss_entity_cls})

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                    # let the loss_coord decay as iteration grows
                    if "loss_coord" in k:
                        losses[k] *= max(((self.max_iter - self._iter.item()) / self.max_iter) ** 2, 0)
                        # print("loss_coord decay:", max((self.max_iter - self._iter.item()) / self.max_iter, 0))
                elif k in self.entity_criterion.weight_dict:
                    losses[k] *= self.entity_criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            processed_results = []
            if self.entity:
                mask_pred_entity_cls_logits = outputs["entity_cls_logits"]
                for mask_cls_result, mask_pred_result, input_per_image, \
                    image_size, per_im_entity_cls_logits in zip(
                    mask_cls_results, mask_pred_results, batched_inputs,
                    images.image_sizes, mask_pred_entity_cls_logits
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = self.entity_semantic_inference(mask_cls_result, mask_pred_result, per_im_entity_cls_logits)
                    r = F.interpolate(
                        r.unsqueeze(0),
                        size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                    )
                    if not self.sem_seg_postprocess_before_inference:
                        r = sem_seg_postprocess(r[0], image_size, height, width)

                    processed_results.append({"sem_seg": r})

                return processed_results

            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = sem_seg_postprocess(
                        mask_pred_result, image_size, height, width
                    )

                # semantic segmentation inference
                r = self.semantic_inference(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = sem_seg_postprocess(r, image_size, height, width)
                processed_results.append({"sem_seg": r})

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = self.panoptic_inference(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

            return processed_results

    def add_coords_targets(self, targets):
        bitmasks = [target['masks'] for target in targets]
        for i, bitmasks_per_image in enumerate(bitmasks):
            _, h, w = bitmasks_per_image.size()

            ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks_per_image.device)
            xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks_per_image.device)

            m00 = bitmasks_per_image.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (bitmasks_per_image * xs).sum(dim=-1).sum(dim=-1)
            m01 = (bitmasks_per_image * ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_x = m10 / m00 / (w - 1.0)
            center_y = m01 / m00 / (h - 1.0)
            targets[i]['coords'] = torch.stack([center_x, center_y], dim=-1)

    def prepare_targets(self, targets, images):
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            masks_sum = padded_masks.flatten(-2, -1).sum(dim=-1)
            valid_masks = masks_sum != 0
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes[valid_masks],
                    "masks": padded_masks[valid_masks],
                }
            )
        return new_targets

    def entity_semantic_inference(self, entity_score, mask_pred, cls_logits):
        device = self.device
        ch = cls_logits.size(0)
        h, w = mask_pred.size()[-2:]
        mask_pred = mask_pred.sigmoid()
        # upsample cls_logits
        # cls_logits = F.interpolate(cls_logits.unsqueeze(0), size=[h, w],
        #                            mode="bilinear", align_corners=False)
        cls_logits = cls_logits.unsqueeze(0)
        # use all preds
        keep = entity_score[:, 0] > -4
        output = torch.zeros([self.sem_seg_head.num_classes, h, w], device=device)
        if (keep == True).sum() == 0:
            # no prediction
            return output

        keep_pred = mask_pred[keep]
        n_inst = keep_pred.size(0)
        keep_pred_weights = keep_pred.flatten(-2, -1).sum(dim=-1)
        masked_avg_pool = (cls_logits * keep_pred.unsqueeze(1)) \
                              .flatten(-2, -1).sum(dim=-1) / \
                          (keep_pred_weights[:, None] + 1.0)
        cls_pred_values, cls_pred_indices = self.cls_head(masked_avg_pool).max(dim=-1)

        # overlap the output by reverse sorted order
        # naive output!!!!!!!!
        inst_ids = {0: 7, 1: 8, 2: 10, 3: 12, 4: 14, 5: 15, 6: 18, 7: 19, 8: 20, 9: 22, 10: 23, 11: 24, 12: 27, 13: 30,
                    14: 31, 15: 32, 16: 33, 17: 35, 18: 36, 19: 37, 20: 38, 21: 39, 22: 41, 23: 42, 24: 43, 25: 44,
                    26: 45, 27: 47, 28: 49, 29: 50, 30: 53, 31: 55, 32: 56, 33: 57, 34: 58, 35: 62, 36: 64, 37: 65,
                    38: 66, 39: 67, 40: 69, 41: 70, 42: 71, 43: 72, 44: 73, 45: 74, 46: 75, 47: 76, 48: 78, 49: 80,
                    50: 81, 51: 82, 52: 83, 53: 85, 54: 86, 55: 87, 56: 88, 57: 89, 58: 90, 59: 92, 60: 93, 61: 95,
                    62: 97, 63: 98, 64: 102, 65: 103, 66: 104, 67: 107, 68: 108, 69: 110, 70: 111, 71: 112, 72: 115,
                    73: 116, 74: 118, 75: 119, 76: 120, 77: 121, 78: 123, 79: 124, 80: 125, 81: 126, 82: 127, 83: 129,
                    84: 130, 85: 132, 86: 133, 87: 134, 88: 135, 89: 136, 90: 137, 91: 138, 92: 139, 93: 142, 94: 143,
                    95: 144, 96: 146, 97: 147, 98: 148, 99: 149}
        # isthing = int(cat_id in inst_ids.values())

        score = (entity_score[:, 0][keep].sigmoid()) ** 2 * cls_pred_values.sigmoid()
        sorted, indices = score.sort(0)
        # add sem_seg first
        for indice in indices:
            cat_id = cls_pred_indices[indice]
            isthing = int(cat_id in inst_ids.values())
            if isthing:
                continue
            output[cls_pred_indices[indice]] = keep_pred[indice]

        for indice in indices:
            cat_id = cls_pred_indices[indice]
            isthing = int(cat_id in inst_ids.values())
            if not isthing:
                continue
            output[cls_pred_indices[indice]] = keep_pred[indice]

        return output

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()

                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info
