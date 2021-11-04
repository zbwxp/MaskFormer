# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation.fast_eval_api import COCOeval_opt as COCOeval
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table
import pdb

class MultiClsEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(self, dataset_name, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            # prediction = {"image_id": input["image_id"]}
            prediction = {}
            gt = input["sem_seg"].unique()
            gt = gt[gt != 255]
            prediction["gt"] = gt
            prediction["pred"] = output["pred_classes"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        topk = 1
        target = []
        pred = []
        for p in self._predictions:
            one_hot = torch.zeros(150)
            one_hot[p['gt']] = 1
            num_gt = int(one_hot.sum())
            target.append(one_hot)

            per_pred = (p['pred'].sigmoid() > 0.5).float()
            pred.append(per_pred)
        target = torch.stack(target)
        pred = torch.cat(pred)

        results = pred.eq(target)

        total_acc = target[results].sum()/target.sum()
        total_recall = target[results].sum()/pred.sum()

        cls_names = self._metadata.stuff_classes
        per_cls_accs = []
        per_cls_recall = []
        for i in range(150):
            acc = target[:, i][results[:, i]].sum() / target[:, i].sum()
            per_cls_accs.append(acc.item()*100)
            recall = target[:, i][results[:, i]].sum() / pred[:, i].sum()
            per_cls_recall.append(recall.item()*100)



        topk_acc = []
        # for k in range(1, topk + 1):
        #     correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        #     topk_acc.append(correct_k.mul_(100.0 / num_samples))
        result = OrderedDict(
            per_cls_acc={"{}".format(name): acc for name, acc in zip(cls_names, per_cls_accs)},
            per_cls_recall={"{}".format(name): re for name, re in zip(cls_names, per_cls_recall)},
            total={"total_acc": total_acc.item() * 100,
                   "total_recall": total_recall.item() * 100},
        )
        return result
