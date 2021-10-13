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
from .data.dataset_mappers.mask_former_multicls_dataset_mapper import (
    MaskFormerMultiClsDatasetMapper,
)
from .data.dataset_mappers.mask_former_ccl_anno_dataset_mapper import (
    MaskFormerCCLAnnoDatasetMapper
)
# models
from .mask_former_model import MaskFormer
from .masked_classification import MaskedClassification
from .masked_classification_v2 import MaskedClassification_v2
from .test_time_augmentation import SemanticSegmentorWithTTA
