"""Base models, ensembles, meta-learners, and augmentations."""
from models.base_models import (
    BASE_MODEL_REGISTRY,
    list_models,
    get_model_factory,
    get_optuna_space,
    get_model_speed,
    get_recommended_features,
    HAS_XGBOOST,
)
from models.ensembles import (
    average_topk,
    average_all,
    bagging_predict,
    weighted_blend_ccc,
)
from models.meta_learners import MetaStack, all_meta_variants, META_REGISTRY
from models.augmentation import apply_augmentation, AUGMENTATION_REGISTRY

__all__ = [
    "BASE_MODEL_REGISTRY",
    "list_models",
    "get_model_factory",
    "get_optuna_space",
    "get_model_speed",
    "get_recommended_features",
    "HAS_XGBOOST",
    "average_topk",
    "average_all",
    "bagging_predict",
    "weighted_blend_ccc",
    "MetaStack",
    "all_meta_variants",
    "META_REGISTRY",
    "apply_augmentation",
    "AUGMENTATION_REGISTRY",
]
