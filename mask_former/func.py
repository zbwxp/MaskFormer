import torch
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def getbox(mask):
    finds_y, finds_x = torch.nonzero(mask == 1, as_tuple=True)
    if len(finds_y) == 0:
        return None
    x1 = torch.min(finds_x)
    x2 = torch.max(finds_x)
    y1 = torch.min(finds_y)
    y2 = torch.max(finds_y)
    if x2 - x1 == 0 or y2 - y1 == 0:
        return None
    return [int(x1), int(y1), int(x2), int(y2)]

def getccl(mask):
    ccl = measure.label(mask)
    values = np.unique(ccl)
    per_mask_ccls_bbox = []
    per_ccl_bbox_mask = []
    for val in values:
        per_ccl_mask = ccl == val
        # check if mask is background
        if mask[per_ccl_mask].sum() == 0:
            continue
        # check if mask is large enough
        if per_ccl_mask.sum() < 16:
            continue
        per_ccl_mask = torch.as_tensor(per_ccl_mask).float()
        bbox = getbox(per_ccl_mask)
        if bbox is None:
            continue
        per_mask_ccls_bbox.append(bbox)
        box_mask = torch.zeros_like(per_ccl_mask)
        box_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        per_ccl_bbox_mask.append(box_mask)

    # 2nd stage merge
    if len(per_mask_ccls_bbox) > 1:
        new_mask = torch.stack(per_ccl_bbox_mask).sum(dim=0)
        new_mask = (new_mask > 0).float()
        ccl = measure.label(new_mask)
        values = np.unique(ccl)

        per_mask_ccls_bbox = []
        mask_save = []
        for val in values:
            per_ccl_mask = ccl == val
            # check if mask is background
            if new_mask[per_ccl_mask].sum() == 0:
                continue
            # check if mask is large enough
            if per_ccl_mask.sum() < 16:
                continue
            per_ccl_mask = torch.as_tensor(per_ccl_mask).float()
            bbox = getbox(per_ccl_mask)
            if bbox is not None:
                per_mask_ccls_bbox.append(bbox)
                mask_save.append(per_ccl_mask)

    return per_mask_ccls_bbox

def getccl_gpu(mask):
    ccl = measure.label(mask)
    values = np.unique(ccl)
    per_mask_ccls_bbox = []
    per_ccl_bbox_mask = []
    for val in values:
        per_ccl_mask = ccl == val
        # check if mask is background
        if mask[per_ccl_mask].sum() == 0:
            continue
        # check if mask is large enough
        if per_ccl_mask.sum() < 16:
            continue
        per_ccl_mask = torch.as_tensor(per_ccl_mask).float()
        bbox = getbox(per_ccl_mask)
        if bbox is None:
            continue
        per_mask_ccls_bbox.append(bbox)
        box_mask = torch.zeros_like(per_ccl_mask)
        box_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        per_ccl_bbox_mask.append(box_mask)

    # 2nd stage merge
    if len(per_mask_ccls_bbox) > 1:
        new_mask = torch.stack(per_ccl_bbox_mask).sum(dim=0)
        new_mask = (new_mask > 0).float()
        ccl = measure.label(new_mask)
        values = np.unique(ccl)

        per_mask_ccls_bbox = []
        mask_save = []
        for val in values:
            per_ccl_mask = ccl == val
            # check if mask is background
            if new_mask[per_ccl_mask].sum() == 0:
                continue
            # check if mask is large enough
            if per_ccl_mask.sum() < 16:
                continue
            per_ccl_mask = torch.as_tensor(per_ccl_mask).float()
            bbox = getbox(per_ccl_mask)
            if bbox is not None:
                per_mask_ccls_bbox.append(bbox)
                mask_save.append(per_ccl_mask)

    return per_mask_ccls_bbox


def gen_crops(bboxes, inputs):
    image = inputs[0, 0:-1]
    mask = inputs[0, -1]
    _, h, w = image.size()
    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.cpu().numpy()
    croped_input = []
    for bbox in bboxes:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        long_edge = max(width, height)
        ratio = 1.5
        if (width * height) / (w * h) < 0.001:
            ratio = 5
        elif (width * height) / (w * h) > 0.25:
            ratio = 1.2
        crop_to_centre = (ratio * long_edge) * 0.5
        center = [(bbox[1] + height / 2), (bbox[0] + width / 2)]
        crop_xyxy = [int(center[1] - crop_to_centre), int(center[0] - crop_to_centre),
                     int(center[1] + crop_to_centre) + 1, int(center[0] + crop_to_centre) + 1]
        crop_pad = []
        crop_pad.append(abs(min(0, crop_xyxy[0])))
        crop_pad.append(abs(min(0, crop_xyxy[1])))
        crop_pad.append(abs(min(0, w - crop_xyxy[2])))
        crop_pad.append(abs(min(0, h - crop_xyxy[3])))

        pad_image = np.pad(image,
                           ((crop_pad[1], crop_pad[3]), (crop_pad[0], crop_pad[2]), (0, 0)),
                           'constant',
                           constant_values=(0, 0))
        pad_mask = np.pad(mask,
                               ((crop_pad[1], crop_pad[3]), (crop_pad[0], crop_pad[2])),
                               'constant',
                               constant_values=(0, 0))
        crop_xyxy[0] += crop_pad[0]
        crop_xyxy[2] += crop_pad[0]
        crop_xyxy[1] += crop_pad[1]
        crop_xyxy[3] += crop_pad[1]

        croped_image = pad_image[crop_xyxy[1]:crop_xyxy[3], crop_xyxy[0]:crop_xyxy[2], :]
        croped_mask = pad_mask[crop_xyxy[1]:crop_xyxy[3], crop_xyxy[0]:crop_xyxy[2]]

        croped_image = torch.as_tensor(croped_image).permute(-1, 0, 1)
        croped_mask = torch.as_tensor(croped_mask)
        croped = torch.cat((croped_image, croped_mask[None, :]), dim=0)
        croped = F.interpolate(croped[None, :], (224, 224), mode='bilinear', align_corners=False)
        croped_input.append(croped)

    return croped_input

def getcls(input, backbone):
    mask = input[0, -1].cpu().numpy()
    bboxes = getccl(mask)
    croped_inputs = gen_crops(bboxes, input)
    if len(croped_inputs)==0:
        return None
    # plt.imshow(input[0, -1].cpu())
    # plt.imshow(croped_inputs[0][0, -1].cpu())
    cls_vec = []
    confidence = 0
    for per_input in croped_inputs:
        cls = backbone(per_input.to(input.device))['linear'][0]
        confi = F.softmax(cls, dim=0).max()
        if confi < confidence:
            continue
        else:
            cls_vec = cls
            confidence = confi
    if len(cls_vec) == 0:
        cls_vec = torch.zeros_like(cls) - 40

    return cls_vec

def mask_to_ccl_masks(masks, classes):
    new_masks = []
    new_classes = []
    for mask, cls in zip(masks, classes):
        bboxes = getccl(mask.cpu())
        for box in bboxes:
            ccl_mask = torch.zeros_like(mask)
            ccl_mask[box[1]:box[3], box[0]:box[2]] = mask[box[1]:box[3], box[0]:box[2]]
            new_masks.append(ccl_mask)
            new_classes.append(cls)

    return torch.stack(new_masks), torch.stack(new_classes)