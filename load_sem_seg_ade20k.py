from detectron2.data.datasets.coco import load_sem_seg
from detectron2.data import detection_utils as utils
import numpy as np
import json


image_dir = 'datasets/ADEChallengeData2016/images/validation'
gt_dir = 'datasets/ADEChallengeData2016/annotations_detectron2/validation'

dataset_dicts = load_sem_seg(gt_dir, image_dir, gt_ext="png", image_ext="jpg")

annos = []
for dict in dataset_dicts:
    anno = {}
    anno.update(dict)
    sem_seg_gt = utils.read_image(dict.pop("sem_seg_file_name")).astype("double")
    unique = np.unique(sem_seg_gt)
    anno.update({"sem_seg_cls": list(unique[unique != 255])})
    annos.append(anno)

# give id to each instance
id = 0
ade20k_with_id = []
for img in annos:
    per_img_id = (np.arange(len(img['sem_seg_cls'])) + 1) + id
    id += len(img['sem_seg_cls'])
    img.update({"anno_id": [int(x) for x in per_img_id]})
    img.update({"sem_seg_cls": [int(a) for a in img['sem_seg_cls']]})
    ade20k_with_id.append(img)

# write per anno information as semseg_info
n = 0
ade20k_instance = []
for image in ade20k_with_id:
    anno_ids = image["anno_id"]
    anno_cls = image["sem_seg_cls"]
    semseg_info = []
    for i in range(len(anno_ids)):
        per_anno_dict = {}
        per_anno_dict.update({"file_name": image["file_name"]})
        per_anno_dict.update({"sem_seg_file_name": image["sem_seg_file_name"]})
        per_anno_dict.update({"category": anno_cls[i]})
        per_anno_dict.update({"anno_id": anno_ids[i]})
        semseg_info.append(per_anno_dict)

    image.update({"semseg_info": semseg_info})
    ade20k_instance.append(image)

# rearrange as list of annos
annos = [a["semseg_info"] for a in ade20k_instance]

list_annos = [item for sublist in annos for item in sublist]

with open("ade20k_anno_list_val.json", 'w') as f:
    json.dump(list_annos, f)

print()