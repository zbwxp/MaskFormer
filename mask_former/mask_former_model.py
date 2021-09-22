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
            entity_test: bool,
            entity: bool,
            conv_dim: int,
            mask_dim: int,
            norm: Optional[Union[str, Callable]] = None,
            num_classes: int,
            use_pred_loss: bool,
            iter_matcher: bool,
            iter_loss: bool,
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
        self.entity_test_on = entity_test
        self.entity = entity
        self.iter_matcher = iter_matcher
        self.iter_loss = iter_loss
        if self.entity_test_on:
            self.matcher = HungarianMatcher_diceonly(
                cost_class=1,
                cost_mask=20.0,
                cost_dice=1.0,
            )
        if self.entity:
            self.use_pred_loss = use_pred_loss
            self.cls_head = MLP(mask_dim, mask_dim, num_classes, 3)
            use_bias = norm == ""
            output_norm = get_norm(norm, conv_dim)
            self.classifyer = nn.Sequential(
                Conv2d(conv_dim,
                       mask_dim,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       bias=use_bias,
                       norm=output_norm,
                       activation=F.relu, ),
                Conv2d(mask_dim,
                       mask_dim,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       bias=use_bias,
                       norm=output_norm,
                       activation=F.relu, ),
                Conv2d(mask_dim,
                       mask_dim,
                       kernel_size=3,
                       stride=1,
                       padding=1, ),
            )

            for layer in self.classifyer:
                weight_init.c2_xavier_fill(layer)
        self.register_buffer("_iter", torch.zeros([1]))
        self._warmup_iters = 4000

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
            else:
                targets = None

            # bipartite matching-based loss
            _iter = self._iter
            if not self.iter_matcher:
                _iter = 0.0
            losses = self.criterion(outputs, targets, _iter)

            if self.entity:
                warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
                cls_feature_map = outputs["cls_feature_map"]
                cls_feature_map = self.classifyer(cls_feature_map)
                labels = [t["labels"] for t in targets]
                masks = [t["masks"] for t in targets]
                bs, ch, h, w = cls_feature_map.size()
                masked_avg_pool = []
                for b in range(bs):
                    n_inst = masks[b].size(0)
                    if n_inst == 0:
                        print("got zero instance per img!!!!!!!!!!")
                        continue
                    per_im_masks = F.interpolate(masks[b].unsqueeze(0).float(), size=[h, w],
                                                 mode="bilinear", align_corners=False)
                    per_im_mask_weights = per_im_masks.flatten(-2).sum(dim=-1)
                    masked_avg_pool.append(
                        (torch.einsum("chw,bqhw->qc", cls_feature_map[b], per_im_masks)
                         / (per_im_mask_weights[0][:, None] + 1.0)))

                if not masked_avg_pool == []:
                    masked_avg_pool = torch.cat(masked_avg_pool)
                    ce_labels = torch.cat(labels)
                    ce_cls_preds_logits = self.cls_head(masked_avg_pool)
                    loss_entity_cls = F.cross_entropy(ce_cls_preds_logits, ce_labels)
                    losses.update({"loss_entity_cls": loss_entity_cls * warmup_factor})


                if self.use_pred_loss:
                    pred = outputs["pred_masks"].sigmoid()
                    pred = (pred > 0.5).float()
                    mask_weights = pred.flatten(-2).sum(dim=-1)
                    masked_avg_pool = (torch.einsum("bqhw,bchw->bqc", pred, cls_feature_map)
                                       / (mask_weights[:, :, None] + 1.0))
                    bce_cls_preds_logits = self.cls_head(masked_avg_pool)
                    # prepare target
                    bce_targets = torch.zeros_like(bce_cls_preds_logits)
                    indices = losses["indices"]
                    outputs = {"pred_masks": pred}
                    indices = self.matcher(outputs, targets)

                    for b in range(bs):
                        label = labels[b]
                        mask = masks[b]
                        pred_idxs = indices[b][0]
                        gt_idxs = indices[b][1]
                        bce_targets[b][pred_idxs, label[gt_idxs]] = 1.0

                    loss_entity_pred_cls = F.binary_cross_entropy_with_logits(bce_cls_preds_logits, bce_targets)
                    losses.update({"loss_entity_cls_pred": loss_entity_pred_cls * warmup_factor})


            for k in list(losses.keys()):
                if self.iter_loss:
                    for key in self.criterion.weight_dict:
                        if "loss_mask" in key:
                            cooldown_factor = max(1 - self._iter / float(16000), 0.0).item()
                            self.criterion.weight_dict.update({key: 20.0 * cooldown_factor})
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = sem_seg_postprocess(
                        mask_pred_result, image_size, height, width
                    )

                if not self.entity:
                    # semantic segmentation inference
                    r = self.semantic_inference(mask_cls_result, mask_pred_result)
                else:
                    r = self.semantic_inference_entity(mask_cls_result, mask_pred_result)

                if self.entity_test_on:
                    r = mask_pred_results[0]
                    r = r[:, : image_size[0], : image_size[1]]
                    tmp = r.argmax(dim=0)
                    pred = r.sigmoid()
                    # pred = r

                    sem_seg_gt = batched_inputs[0]["sem_seg"].to(self.device)
                    targets = []
                    labels = []
                    masks = []
                    for gt_cls in sem_seg_gt.unique():
                        if gt_cls == 255:
                            continue
                        # pad gt
                        gt_mask = sem_seg_gt == gt_cls
                        labels.append(gt_cls)
                        masks.append(gt_mask)
                    labels = torch.as_tensor(labels)
                    masks = torch.stack(masks)
                    targets.append(
                        {
                            "labels": labels,
                            "masks": masks,
                        }
                    )

                    outputs = {"pred_masks": pred.unsqueeze(0)}
                    indices = self.matcher(outputs, targets)
                    gt_cls_indices = targets[0]["labels"][indices[0][1]]
                    # plot miss classification masks
                    # err = indices[0][1] != torch.arange(len(indices[0][1]))
                    # if err.sum() > 0:
                    #     print()
                    # renew r
                    r = torch.zeros((150, pred.size(1), pred.size(-1)),
                                    dtype=pred.dtype, device=self.device)
                    for gt_idx, pred_idx in zip(gt_cls_indices, indices[0][0]):
                        r[gt_idx] = pred[pred_idx]

                    # f, axarr = plt.subplots(2, 2)
                    # sem_seg_gt[sem_seg_gt ==255] = 0
                    # axarr[0,0].imshow(sem_seg_gt.to('cpu'))
                    # axarr[0,1].imshow(tmp.to('cpu'))
                    # new = r.argmax(dim=0)
                    # axarr[1,0].imshow(new.to('cpu'))
                    # axarr[1,1].imshow((new == tmp).to('cpu'))
                    # filename = batched_inputs[0]["file_name"].split('/')[-1]
                    # plt.savefig("/media/bz/D/美团/MaskFormer/plot/orig_argmax/{}".format(filename))

                if not self.sem_seg_postprocess_before_inference:
                    r = sem_seg_postprocess(r, image_size, height, width)
                processed_results.append({"sem_seg": r})

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = self.panoptic_inference(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

            return processed_results

    def entity_inference(self, mask_cls, mask_pred):
        scores, labels = mask_cls.sigmoid().max(dim=-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        result = Instances([0, 0])
        result.scores = cur_scores
        result.classes = cur_classes
        result.masks = cur_masks

        return result

    def prepare_targets(self, targets, images):
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def semantic_inference_entity(self, mask_cls, mask_pred):
        mask_cls = mask_cls.sigmoid()
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->qhw", mask_cls, mask_pred)
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
