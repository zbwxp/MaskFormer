# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_mask_former_config

# dataset loading
from .data.dataset_mappers.detr_panoptic_dataset_mapper import DETRPanopticDatasetMapper
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

# models
from .mask_former_model import MaskFormer
from .mask_former_model_seperate import MaskFormer_seperate
from .mask_former_model_seperate_v2 import MaskFormer_seperatev2
from .mask_former_model_seperate_v3 import MaskFormer_seperatev3
from .test_time_augmentation import SemanticSegmentorWithTTA
