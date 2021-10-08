import torch
from detectron2.data.datasets.coco import load_sem_seg
from detectron2.data import detection_utils as utils
import numpy as np
import json
from shutil import copyfile
import matplotlib.pyplot as plt
from skimage import measure

image_dir = 'datasets/ADEChallengeData2016/images/training'
gt_anno_dir = 'datasets/ADEChallengeData2016/annotations_detectron2/training'
json_anno_dir = '/media/bz/D/data/ADEChallengeData2016/annotations_detectron2/ade20k_anno_list_train.json'

with open(json_anno_dir, 'r') as f:
    annos = json.load(f)

ccl_anno_id = 1
ccl_annos = []
per_cls_counts = np.zeros(150)
base = '/media/bz/D/data'
min_mask = 1000
count = 0


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


def get_dicts(ccl, values, per_cls_gt, count):
    per_anno_ccls = []
    for val in values:
        mask = ccl == val
        # check if mask is background
        if per_cls_gt[mask].sum() == 0:
            continue
        # check if mask is large enough
        if mask.sum() < 16:
            count += 1
            if count % 1000 == 0:
                print("drop ", count)
            continue
        mask = torch.as_tensor(mask).float()
        bbox = getbox(mask)
        if bbox is None:
            continue

        box_mask = torch.zeros_like(mask)
        box_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        per_anno_dict = {}
        per_anno_dict.update({"file_name": anno["file_name"]})
        per_anno_dict.update({"sem_seg_file_name": anno["sem_seg_file_name"]})
        per_anno_dict.update({"category": anno['category']})
        per_anno_dict.update({"bbox_xyxy": bbox})
        per_anno_dict.update({"box_mask": box_mask})

        per_anno_ccls.append(per_anno_dict)
    return per_anno_ccls, count


for anno in annos:
    if anno['anno_id'] % 1000 == 0:
        print("processing ", str(anno['anno_id']))
    # img_path = base + anno['file_name'][8:]
    semseg_gt_path = base + anno['sem_seg_file_name'][8:]
    sem_seg_gt = utils.read_image(semseg_gt_path)
    per_cls_gt = (sem_seg_gt == anno['category']).astype("double")
    area = per_cls_gt.sum()
    # if area < min_mask:
    #     print("new min mask area:", area)
    #     min_mask = area
    # if area < 32:
    #     print("small area:", area)
    #     count += 1

    ccl = measure.label(per_cls_gt)
    values = np.unique(ccl)
    # if len(values) > 3:
    #     print()
    per_anno_ccls, count = get_dicts(ccl, values, per_cls_gt, count)
    if len(per_anno_ccls)==0:
        continue

    if len(per_anno_ccls) > 1:
        # 2nd stage ccl
        bboxes = [a['bbox_xyxy'] for a in per_anno_ccls]
        box_masks = [a['box_mask'] for a in per_anno_ccls]
        new_mask = torch.stack(box_masks).sum(dim=0)
        new_mask = (new_mask > 0).float()

        ccl = measure.label(new_mask)
        values = np.unique(ccl)
        per_anno_ccls, count = get_dicts(ccl, values, new_mask, count)

        for anno in per_anno_ccls:
            anno.pop("box_mask")
            anno.update({"ccl_anno_id": ccl_anno_id})
            ccl_anno_id += 1
            per_cls_counts[anno['category']] += 1

            ccl_annos.append([anno])
    else:
        per_anno_ccls[0].pop("box_mask")
        per_anno_ccls[0].update({"ccl_anno_id": ccl_anno_id})
        ccl_anno_id += 1
        per_cls_counts[anno['category']] += 1

        ccl_annos.append(per_anno_ccls)
save = {}
save.update({"ccl_annos": ccl_annos})
save.update({"per_cls_anno_counts": per_cls_counts.astype(int).tolist()})

with open("ade20k_ccl_annos_train.json", 'w') as f:
    json.dump(save, f)

print()
