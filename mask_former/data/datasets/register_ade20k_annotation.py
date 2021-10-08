import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
from detectron2.data.datasets.coco import load_sem_seg
from detectron2.data import detection_utils as utils
import numpy as np
import json

def load_anno_ade20k(json_path, gt_ext="png", image_ext="jpg"):
    # dataset_dicts = load_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg")
    # annos = []
    # for dict in dataset_dicts:
    #     anno = {}
    #     anno.update(dict)
    #     sem_seg_gt = utils.read_image(dict.pop("sem_seg_file_name")).astype("double")
    #     unique = np.unique(sem_seg_gt)
    #     anno.update({"sem_seg_cls": list(unique[unique != 255])})
    #     annos.append(anno)
    with open(json_path, 'r') as f:
        annos = json.load(f)

    dataset_dicts = annos
    return dataset_dicts

def load_ccl_anno_ade20k(json_path, gt_ext="png", image_ext="jpg"):
    with open(json_path, 'r') as f:
        annos = json.load(f)

    ccl_annos = annos['ccl_annos']
    dataset_dicts = []
    for anno in ccl_annos:
        dataset_dicts.append(anno[0])

    return dataset_dicts


def register_anno_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        json_dir = os.path.join(root, "annotations_detectron2")
        json_path = os.path.join(json_dir, f"ade20k_anno_list_{name}.json")
        name = f"ade20k_multi_cls_{name}"
        DatasetCatalog.register(
            name, lambda x=json_path: load_anno_ade20k(x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="classification",
            ignore_label=255,
            json_file=json_path,
        )

def register_ccl_anno_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        json_dir = os.path.join(root, "annotations_detectron2")
        json_path = os.path.join(json_dir, f"ade20k_ccl_annos_{name}.json")
        name = f"ade20k_ccl_annos_{name}"
        DatasetCatalog.register(
            name, lambda x=json_path: load_ccl_anno_ade20k(x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="classification",
            ignore_label=255,
            json_file=json_path,
        )



_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_anno_ade20k(_root)
register_ccl_anno_ade20k(_root)