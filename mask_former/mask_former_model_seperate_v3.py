# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple, Optional, Union, Callable

import torch
from torch import nn
from torch.nn import functional as F

import fvcore.nn.weight_init as weight_init

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.layers import get_norm, Conv2d

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .modeling.dice_matcher import HungarianMatcher_DICE
from .modeling.transformer.transformer_predictor import MLP
import matplotlib.pyplot as plt

@META_ARCH_REGISTRY.register()
class MaskFormer_seperatev3(nn.Module):
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
        num_classes: int,
        cls_test: bool,
        norm: Optional[Union[str, Callable]] = None,
        cls_head_dim: int,
        cls_head_layers: int,
        freeze_maskformer: bool,
        use_gt_targets: bool,
        cls_head_kernel_size: int,

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

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = MLP(1024, 1024, num_classes + 1, 3)
        # self.classifier = MLP(256, 256, num_classes, 3)
        self.num_classes = num_classes

        self.cls_test = cls_test

        self.matcher = HungarianMatcher(
            cost_class=1,
            cost_mask=20.0,
            cost_dice=1.0,
        )
        if cls_head_kernel_size == 1:
            conv = Conv2d(cls_head_dim, cls_head_dim, 1, 1, norm=get_norm(norm, 256), activation=F.relu)
        else:
            conv = Conv2d(cls_head_dim, cls_head_dim, 3, 1, 1, norm=get_norm(norm, 256), activation=F.relu)
        # each stage assign a nnsequence
        multi_stage_heads = []
        for l in range(4):
            head_convs = []
            for i in range(cls_head_layers):
                head_convs.append(conv)
            for conv in head_convs:
                weight_init.c2_xavier_fill(conv)
            self.add_module('cls_head_res{}'.format(l+2), nn.Sequential(*head_convs))
            multi_stage_heads.append(nn.Sequential(*head_convs))
        self.cls_head = multi_stage_heads[::-1]

        self.freeze_maskformer = freeze_maskformer

        self.use_gt_targets = use_gt_targets
        if not self.use_gt_targets:
            self.dice_matcher = HungarianMatcher_DICE(
            cost_class=1,
            cost_mask=20.0,
            cost_dice=1.0,
        )

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=1,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
        )

        weight_dict = {"loss_ce": 1, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        weight_dict.update({"loss_cls_by_tgt": cfg.MODEL.MASK_FORMER.CLS_WEIGHT})
        weight_dict.update({"loss_cls_by_pred": cfg.MODEL.MASK_FORMER.CLS_WEIGHT * 0.2})
        # weight_dict.update({"loss_ce_tgt": cfg.MODEL.MASK_FORMER.TGT_WEIGHT})

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
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
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "cls_test": cfg.MODEL.MASK_FORMER.TEST.CLASSIFICATION,
            "norm": cfg.MODEL.SEM_SEG_HEAD.NORM,
            "cls_head_dim": cfg.MODEL.MASK_FORMER.CLS_HEAD_DIM,
            "cls_head_layers": cfg.MODEL.MASK_FORMER.CLS_HEAD_LAYERS,
            "freeze_maskformer": cfg.MODEL.MASK_FORMER.FREEZE,
            "use_gt_targets": cfg.MODEL.MASK_FORMER.USE_GT_TARGETS,
            "cls_head_kernel_size": cfg.MODEL.MASK_FORMER.CLS_HEAD_KERNEL_SIZE,
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

        # mask classification target
        if True:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None


        # mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        losses = {}
        if not self.freeze_maskformer and self.training:
            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

        # individual cls branck
        maps = outputs['multi_level_feature_maps']
        new_maps = []
        for i, map in enumerate(maps):
            cls_head = self.cls_head[i]
            map = cls_head(map)
            # map = F.interpolate(map, size=features['res2'].size()[-2:], mode='nearest')
            new_maps.append(map)

        if self.training:
            # gt target loss
            pred_cls_logits = self.get_cls_vec_loop_orig(new_maps, targets)
            labels = torch.cat([x['labels'] for x in targets])
            loss_ce_cls = F.cross_entropy(pred_cls_logits, labels)
            losses.update({"loss_cls_by_tgt": loss_ce_cls})


        if not self.training:
            pred_targets = []
            for mask_pred_result in mask_pred_results:
                keep = self.filter_masks(mask_pred_result)
                mask_pred_result = mask_pred_result[keep]
                mask_dict = {"aug_masks": (mask_pred_result > 0.5).float()}
                mask_dict.update({"pred_masks": mask_pred_result})
                pred_targets.append(mask_dict)

            batch_size = len(batched_inputs)
            pred_cls_logits = self.get_cls_vec_loop_orig_test(new_maps, pred_targets)
            pred_cls_logits = pred_cls_logits.reshape(batch_size, -1, self.num_classes + 1)

            # use pred targets to train negtive samples

            # maps = torch.cat(new_maps, dim=1)
            # maps = outputs['mask_features']
            # prepare targets from maskformer pred

            # generate pred
            idx = 0
            preds = []
            for per_img_pred, per_img_cls in zip(pred_targets, pred_cls_logits):
                mask_pred_result = per_img_pred["pred_masks"]
                num_masks = mask_pred_result.size(0)
                cls_logit = per_img_cls
                mask_cls = F.softmax(cls_logit, dim=-1)[..., :-1]
                mask_pred = mask_pred_result.sigmoid()
                semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
                preds.append(semseg)
                idx += num_masks

            preds = torch.stack(preds, dim=0)
            sem_seg_gts = [x["sem_seg"].to(self.device) for x in batched_inputs]
            sem_seg_gts = ImageList.from_tensors(sem_seg_gts, self.size_divisibility).tensor
            preds = F.interpolate(preds, size=sem_seg_gts.size()[-2:], mode='bilinear', align_corners=False)

        if self.training:
            # loss_ce = F.cross_entropy(preds, sem_seg_gts, ignore_index=255, reduction='mean')
            # losses.update({"loss_ce_tgt": loss_ce})

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
        else:
            r = preds
            image_size = images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])

            processed_results = []
            if not self.sem_seg_postprocess_before_inference:
                r = sem_seg_postprocess(r, image_size, height, width)
            processed_results.append({"sem_seg": r})

            return processed_results

    def filter_masks(self, masks):
        # argmax thresh
        argmax_thresh, counts = masks.argmax(dim=0).unique(return_counts=True)
        # min area thresh
        q, h, w = masks.size()
        total_area = h * w
        thresh_area = counts > total_area * 0.0005

        return argmax_thresh[thresh_area]

    def get_cls_vec(self, maps, targets):
        masked_pool_vec = []
        for i, target in enumerate(targets):
            per_img_masks = target['aug_masks']
            h, w = maps.size()[-2:]
            if per_img_masks.size()[0] == 0:
                print("no mask in this img")
                continue
            per_img_masks = F.interpolate(per_img_masks[None, :], size=(h, w), mode='nearest')[0]
            per_img_features = maps[i]
            masked_map = torch.einsum("qhw,chw->qchw", per_img_masks, per_img_features)
            # masked_map = F.max_pool2d(masked_map, kernel_size=3, stride=2, padding=1)
            map_weights = (masked_map.sum(dim=1) > 0).flatten(-2).sum(dim=-1)
            s_h, s_w = masked_map.size()[-2:]
            area = s_h * s_w
            masked_pool = self.pool(masked_map).squeeze() * (area / (map_weights + 1))[:, None]
            masked_pool_vec.append(masked_pool)

        masked_pool_vec = torch.cat(masked_pool_vec, dim=0)
        pred_logits = self.classifier(masked_pool_vec)
        return pred_logits

    def get_cls_vec_loop(self, maps, targets):
        masked_pool_vec = []
        masks = [x['aug_masks'] for x in targets]
        for i, per_img_masks in enumerate(masks):
            per_img_features = [maps[0][i], maps[1][i], maps[2][i], maps[3][i]]
            per_img_pool_vec = []
            for per_stage_features in per_img_features:
                per_stage_masks = F.interpolate(per_img_masks[None, :], size=per_stage_features.size()[-2:], mode="nearest")
                masked_map = torch.einsum("bqhw,chw->qchw", per_stage_masks, per_stage_features)
                # masked_map = F.max_pool2d(masked_map, kernel_size=3, stride=2, padding=1)
                map_weights = (masked_map.sum(dim=1) > 0).flatten(-2).sum(dim=-1)
                s_h, s_w = masked_map.size()[-2:]
                area = s_h * s_w
                masked_pool = self.pool(masked_map).squeeze() * (area / (map_weights + 1))[:, None]
                per_img_pool_vec.append(masked_pool)
            masked_pool_vec.append(torch.cat(per_img_pool_vec, dim=-1))
        masked_pool_vec = torch.cat(masked_pool_vec, dim=0)
        pred_logits = self.classifier(masked_pool_vec)
        return pred_logits

    def get_cls_vec_loop_orig(self, maps, targets):
        masked_pool_vec = []
        masks = [x['aug_masks'] for x in targets]
        for i, per_img_masks_orig in enumerate(masks):
            per_img_features = maps[i]
            per_img_masks = F.interpolate(per_img_masks_orig[None, :], size=per_img_features.size()[-2:], mode="nearest")
            total_area = per_img_masks.size(-1) * per_img_masks.size(-2)
            areas = per_img_masks[0].flatten(-2).sum(-1)
            if 0 in areas:
                tmp = F.interpolate(per_img_masks_orig[None, :],
                                    size=per_img_features.size()[-2:],
                                    mode="bilinear",
                                    align_corners=False)
                per_img_masks[0][areas==0] = (tmp[0][areas == 0] > 0).float()
                areas = per_img_masks[0].flatten(-2).sum(-1)
            masked_map = torch.einsum("bqhw,chw->qchw", per_img_masks, per_img_features)
            # masked_map = F.max_pool2d(masked_map, kernel_size=3, stride=2, padding=1)
            map_weights = total_area / (areas + 1)
            masked_pool = self.pool(masked_map).squeeze() * map_weights[:, None]
            masked_pool_vec.append(masked_pool)
        masked_pool_vec = torch.cat(masked_pool_vec, dim=0)
        pred_logits = self.classifier(masked_pool_vec)
        return pred_logits

    def get_cls_vec_loop_orig_test(self, maps, targets):
        masked_pool_vec = []
        masks = [x['aug_masks'] for x in targets]
        for i, per_img_masks_orig in enumerate(masks):
            per_img_features = maps[i]
            per_img_masks = F.interpolate(per_img_masks_orig[None, :], size=per_img_features.size()[-2:], mode="nearest")
            total_area = per_img_masks.size(-1) * per_img_masks.size(-2)
            areas = per_img_masks[0].flatten(-2).sum(-1)
            if 0 in areas:
                tmp = F.interpolate(per_img_masks_orig[None, :],
                                    size=per_img_features.size()[-2:],
                                    mode="bilinear",
                                    align_corners=False)
                per_img_masks[0][areas==0] = (tmp[0][areas == 0] > 0).float()
                areas = per_img_masks[0].flatten(-2).sum(-1)

            for i_mask, mask in enumerate(per_img_masks[0]):
                masked_map = torch.einsum("hw,chw->chw", mask, per_img_features)
                # masked_map = F.max_pool2d(masked_map, kernel_size=3, stride=2, padding=1)
                map_weights = total_area / (areas + 1)
                masked_pool = self.pool(masked_map).squeeze() * map_weights[i_mask, None]
                masked_pool_vec.append(masked_pool)
        masked_pool_vec = torch.stack(masked_pool_vec, dim=0)
        pred_logits = self.classifier(masked_pool_vec)
        return pred_logits


    def prepare_targets(self, targets, images):
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            gt_classes = targets_per_image.gt_classes
            aug_masks = gt_masks.clone().float()
            # apply aug to masks
            shirnk_mask = F.interpolate(aug_masks[None, :], scale_factor=0.125, mode='bilinear',
                                        align_corners=False)[0]
            if self.training:
                # augs
                for i, mask in enumerate(aug_masks):
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
                        shirnk = F.interpolate(mask[None, None, :], scale_factor=0.125, mode='bilinear',
                                               align_corners=False)
                        expand = F.interpolate(shirnk, orig_mask.size()[-2:], mode='bilinear', align_corners=False)
                        mask = (expand[0, 0] > 0.5).float()
                        if mask.sum() < 64:
                            continue

                    aug_masks[i] = mask

                    # f, axarr = plt.subplots(2, 2)
                    # axarr[0, 0].imshow(orig_mask.to('cpu'))
                    # axarr[0, 1].imshow(gt_masks[i].to('cpu'))
                    # axarr[1, 0].imshow(aug_masks[i].to('cpu'))
                    # print()

            padded_masks = torch.zeros((gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            padded_aug_masks = torch.zeros((aug_masks.shape[0], h, w), dtype=aug_masks.dtype, device=aug_masks.device)
            padded_aug_masks[:, : aug_masks.shape[1], : aug_masks.shape[2]] = aug_masks
            if self.training:
                # elimate too small masks
                areas = padded_masks.flatten(-2).sum(-1)
                keep = areas > 16
                gt_classes = gt_classes[keep]
                padded_masks = padded_masks[keep]
                padded_aug_masks = padded_aug_masks[keep]

            new_targets.append(
                {
                    "labels": gt_classes,
                    "masks": padded_masks,
                    "aug_masks": padded_aug_masks
                }
            )
        return new_targets

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
