"""Nine base regression models (Ridge, Lasso, ElasticNet, PLS, SVR-RBF, GPR, KNN, RF, XGBoost)
with default hyperparameters, Optuna search spaces, and a fast/slow tag for nested-CV budgets."""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from config import SEED

log = logging.getLogger(__name__)


try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    log.warning("xgboost not installed; XGB models will not be available")


def make_ridge(alpha: float = 10.0, **kwargs) -> Ridge:
    return Ridge(alpha=alpha, random_state=SEED, **kwargs)


def ridge_optuna_space(trial) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-2, 1e3, log=True)}


RIDGE_SPEED = "fast"


def make_lasso(alpha: float = 1e-3, **kwargs) -> Lasso:
    return Lasso(alpha=alpha, random_state=SEED, max_iter=5000, **kwargs)


def lasso_optuna_space(trial) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-5, 1e0, log=True)}


LASSO_SPEED = "fast"


def make_elasticnet(alpha: float = 1e-3,
                    l1_ratio: float = 0.5,
                    **kwargs) -> ElasticNet:
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                      random_state=SEED, max_iter=5000, **kwargs)


def elasticnet_optuna_space(trial) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 1e-5, 1e0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
    }


ELASTICNET_SPEED = "fast"


def make_pls(n_components: int = 30, **kwargs) -> PLSRegression:
    # PLS has no random_state (no stochastic components).
    return PLSRegression(n_components=n_components, max_iter=500, **kwargs)


def pls_optuna_space(trial) -> dict:
    return {"n_components": trial.suggest_int("n_components", 5, 50)}


PLS_SPEED = "fast"


def make_svr(C: float = 1.0,
             gamma: str | float = "scale",
             epsilon: float = 0.01,
             **kwargs) -> SVR:
    return SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon, **kwargs)


def svr_optuna_space(trial) -> dict:
    return {
        "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1e0, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-3, 1e-1, log=True),
    }


SVR_SPEED = "slow"   # SVR scales with n^2 and kernel computation


def make_gpr(length_scale: float = 1.0,
             noise_level: float = 1e-2,
             **kwargs) -> GaussianProcessRegressor:
    kernel = (ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
              * RBF(length_scale=length_scale,
                    length_scale_bounds=(1e-2, 1e3))
              + WhiteKernel(noise_level=noise_level,
                            noise_level_bounds=(1e-6, 1e1)))
    return GaussianProcessRegressor(kernel=kernel,
                                     normalize_y=True,
                                     random_state=SEED,
                                     n_restarts_optimizer=2,
                                     **kwargs)


def gpr_optuna_space(trial) -> dict:
    # GPR optimises the kernel internally; we only tune the initial length scale,
    # which influences the optimiser's starting point.
    return {
        "length_scale": trial.suggest_float("length_scale", 1e-1, 1e2, log=True),
        "noise_level": trial.suggest_float("noise_level", 1e-4, 1e-1, log=True),
    }


GPR_SPEED = "slow"   # cubic in n, kills us on big batches


def make_knn(n_neighbors: int = 15,
             weights: str = "distance",
             **kwargs) -> KNeighborsRegressor:
    return KNeighborsRegressor(n_neighbors=n_neighbors,
                                weights=weights,
                                n_jobs=1, **kwargs)


def knn_optuna_space(trial) -> dict:
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
        "weights": trial.suggest_categorical("weights",
                                              ["uniform", "distance"]),
    }


KNN_SPEED = "fast"


def make_rf(n_estimators: int = 100,
            max_depth: int = 8,
            min_samples_leaf: int = 3,
            **kwargs) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf,
                                  n_jobs=1,
                                  random_state=SEED,
                                  **kwargs)


def rf_optuna_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }


RF_SPEED = "fast"


def make_xgb(n_estimators: int = 200,
             max_depth: int = 4,
             learning_rate: float = 0.05,
             subsample: float = 0.8,
             colsample_bytree: float = 0.8,
             **kwargs):
    if not HAS_XGBOOST:
        raise ImportError("xgboost not installed")
    return XGBRegressor(n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        n_jobs=1,
                        random_state=SEED,
                        verbosity=0,
                        **kwargs)


def xgb_optuna_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }


XGB_SPEED = "slow"


# Each entry: (factory, optuna_space, speed_tag, recommended_features).
# `recommended_features` is one of {"bins300", "roi", "any"}; the evaluator uses it
# to pair each model with its best-fit representation.
BASE_MODEL_REGISTRY: dict[str, tuple[Callable, Callable, str, str]] = {
    "ridge":      (make_ridge,      ridge_optuna_space,      RIDGE_SPEED,      "bins300"),
    "lasso":      (make_lasso,      lasso_optuna_space,      LASSO_SPEED,      "bins300"),
    "elasticnet": (make_elasticnet, elasticnet_optuna_space, ELASTICNET_SPEED, "bins300"),
    "pls":        (make_pls,        pls_optuna_space,        PLS_SPEED,        "bins300"),
    "svr":        (make_svr,        svr_optuna_space,        SVR_SPEED,        "roi"),
    "gpr":        (make_gpr,        gpr_optuna_space,        GPR_SPEED,        "roi"),
    "knn":        (make_knn,        knn_optuna_space,        KNN_SPEED,        "roi"),
    "rf":         (make_rf,         rf_optuna_space,         RF_SPEED,         "roi"),
}
if HAS_XGBOOST:
    BASE_MODEL_REGISTRY["xgb"] = (make_xgb, xgb_optuna_space, XGB_SPEED, "roi")


def list_models() -> list[str]:
    return list(BASE_MODEL_REGISTRY.keys())


def get_model_factory(name: str) -> Callable:
    if name not in BASE_MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name}; available: {list_models()}")
    return BASE_MODEL_REGISTRY[name][0]


def get_optuna_space(name: str) -> Callable:
    if name not in BASE_MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name}; available: {list_models()}")
    return BASE_MODEL_REGISTRY[name][1]


def get_model_speed(name: str) -> str:
    return BASE_MODEL_REGISTRY[name][2]


def get_recommended_features(name: str) -> str:
    return BASE_MODEL_REGISTRY[name][3]
