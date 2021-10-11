# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances
import matplotlib.pyplot as plt
from skimage import measure
import cv2

__all__ = ["MaskFormerCCLAnnoDatasetMapper"]


class MaskFormerCCLAnnoDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
            self,
            is_train=True,
            *,
            augmentations,
            image_format,
            ignore_label,
            size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD and is_train:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )
        # get crop region here
        bbox = dataset_dict["bbox_xyxy"]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        long_edge = max(width, height)
        ratio = 1.5
        if (width*height) / (dataset_dict['width'] * dataset_dict['height']) < 0.001:
            ratio = 5
        elif (width*height) / (dataset_dict['width'] * dataset_dict['height']) > 0.25:
            ratio = 1.2
        crop_to_centre = (ratio * long_edge) * 0.5
        # [y,x]
        center = [(bbox[1] + height / 2), (bbox[0] + width / 2)]
        crop_xyxy = [int(center[1] - crop_to_centre), int(center[0] - crop_to_centre),
                     int(center[1] + crop_to_centre) + 1, int(center[0] + crop_to_centre) + 1]
        crop_pad = []
        crop_pad.append(abs(min(0, crop_xyxy[0])))
        crop_pad.append(abs(min(0, crop_xyxy[1])))
        crop_pad.append(abs(min(0, dataset_dict['width'] - crop_xyxy[2])))
        crop_pad.append(abs(min(0, dataset_dict['height'] - crop_xyxy[3])))

        pad_image = np.pad(image,
                           ((crop_pad[1], crop_pad[3]), (crop_pad[0], crop_pad[2]), (0, 0)),
                           'constant',
                           constant_values=(0, 0))
        pad_semseg_gt = np.pad(sem_seg_gt,
                               ((crop_pad[1], crop_pad[3]), (crop_pad[0], crop_pad[2])),
                               'constant',
                               constant_values=(255, 255))

        crop_xyxy[0] += crop_pad[0]
        crop_xyxy[2] += crop_pad[0]
        crop_xyxy[1] += crop_pad[1]
        crop_xyxy[3] += crop_pad[1]

        croped_image = pad_image[crop_xyxy[1]:crop_xyxy[3], crop_xyxy[0]:crop_xyxy[2], :]
        croped_semseg_gt = pad_semseg_gt[crop_xyxy[1]:crop_xyxy[3], crop_xyxy[0]:crop_xyxy[2]]

        # f, axarr = plt.subplots(2, 2)
        # axarr[0, 0].imshow(image)
        # axarr[0, 1].imshow(croped_image)
        # axarr[1, 0].imshow(sem_seg_gt)
        # axarr[1, 1].imshow(croped_semseg_gt)

        image = croped_image
        sem_seg_gt = croped_semseg_gt

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # NOTE here, only load one mask for this annotation!
            cls_for_anno = dataset_dict["category"]

            if cls_for_anno not in classes:
                # print("the anno is masked out by augmentation!!!!")
                return None
            classes = [cls_for_anno]
            # remove ignored region
            # classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

            h, w = masks.image_size
            if h!=224 or w!= 224:
                print()
            dataset_dict['width'] = w
            dataset_dict['height'] = h

            # f, axarr = plt.subplots(2, 2)
            # axarr[0, 0].imshow(dataset_dict['image'].permute(1, 2, 0))
            # axarr[0, 1].imshow(dataset_dict['instances'].gt_masks[0])

            if self.is_train:
                # augs to mask
                masks = dataset_dict['instances'].gt_masks.float()[None, :]
                shirnk_mask = F.interpolate(masks, scale_factor=0.125, mode='bilinear',
                                               align_corners=False)
                for i, mask in enumerate(masks):
                    orig_mask = mask.clone()
                    if torch.rand(1) > 0.2:
                        mask = shirnk_mask[i]
                        _, mask_height, mask_width = mask.size()
                        new_mask = torch.zeros_like(mask[0])
                        finds_y, finds_x = torch.nonzero(mask[0] == 1, as_tuple=True)
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
                        rand1 = torch.rand(1)
                        rand2 = torch.rand(1)
                        rand3 = torch.randn(1) + 1
                        rand4 = torch.randn(1) + 1
                        rand5 = torch.rand(1)
                        rand6 = torch.rand(1) - 0.2

                        finds_y = (torch.rand(finds_y.size()) - 0.5 * rand3) \
                                  * height * rand1 * 0.2 + finds_y.float()
                        finds_x = (torch.rand(finds_x.size()) - 0.5 * rand4) \
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
                            new_mask = orig_mask
                        mask = new_mask

                    if torch.rand(1) > 0.5:
                        shirnk = F.interpolate(mask[None, :], scale_factor=0.125, mode='bilinear', align_corners=False)
                        expand = F.interpolate(shirnk, orig_mask.size()[-2:], mode='bilinear', align_corners=False)
                        if expand.sum() < 64:
                            expand = orig_mask
                        mask = (expand[0] > 0.5).float()

                    masks[i] = mask

                dataset_dict["instances"].gt_masks = masks[0]

                # f, axarr = plt.subplots(2, 2)
                # axarr[0, 0].imshow(orig_mask[0].to('cpu'))
                # axarr[0, 1].imshow(dataset_dict["instances"].gt_masks[0].to('cpu'))
                # axarr[1, 1].imshow(dataset_dict['image'].permute(1,2,0))
                # print()

        return dataset_dict
