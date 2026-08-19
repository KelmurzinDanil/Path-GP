# main.py
import sympy as sp
import numpy as np
import pandas as pd
import warnings
import random
import torch

try:
    from linear_operator.utils.warnings import NumericalWarning
    warnings.simplefilter("ignore", NumericalWarning)
except ImportError:
    pass

from generate.generate_dataset import *
from GP.config import (
    GPConfig, ModelConfig, TrainingConfig, ScalingConfig, LikelihoodConfig,
    Scale, RBF, Matern12, Matern32, Matern52, Linear, Polynomial, Periodic, RQ
)
from config import (
    PipelineConfig, BruteForceConfig, DimensionalAnalysisConfig,
    FriedmanSeparabilityConfig, CompositionalityConfig, SimplificationConfig,
    ResidualConfig
)
from symbolic.sm_context import SymbolicRegressionContext
from symbolic.pipeline_step import (
    BaselineGPStep,
    GPSimplificationStep
)
from residual_analysis import ResidualAnalyzer
from evaluation import resolve_expression, display_pipeline_results
from logger import setup_logger

logger = setup_logger("Main")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Установлен единый генератор случайных чисел: seed={seed}")


def build_base_config() -> PipelineConfig:
    return PipelineConfig(
        verbose=True,
        gp_config=GPConfig(
            scaling=ScalingConfig(enabled=True),
            model=ModelConfig(
                mean_type="constant",
                kernel=Scale(Linear()) + Scale(Matern32(ard=True)) + Scale(RQ(ard=True)),
                likelihood=LikelihoodConfig(extra_kwargs={})
            ),
            training=TrainingConfig(
                lr=0.1, epochs=80, early_stopping_patience=30,
                optimizer="lbfgs", loss_type="mll", verbose=False, cholesky_jitter=1e-6
            ),
            seed=42
        ),
        brute_force=BruteForceConfig(
            max_length=8, complexity_penalty=0.0001, optimize_constants=False,
            allowed_constants=[1.0, 2.0, 0.5],
            allowed_ops=["add", "sub", "mul", "div", "sqrt", "pow", "exp"],
            k_best=50
        ),
        friedman=FriedmanSeparabilityConfig(
            h2_threshold=0.01, grid_density=15, bg_samples=200, max_allowed_distance=1.2, min_contrast=1.5, bf_max_length=7
        ),
        compositionality=CompositionalityConfig(
            grid_density=15, n_slices=20, slice_spearman_threshold=0.92, template_spearman_threshold=0.90, bf_max_length=7
        ),
        simplification=SimplificationConfig(
            max_depth=8, local_bf_max_length=8, final_bf_max_length=8, base_k_sigma=3.0, loss_degradation_tolerance=0.5
        )
    )


if __name__ == "__main__":
    set_seed(42)

    df_base, reg_base, target_name = load_combined_teeth_invariants()
    base_context = SymbolicRegressionContext.create_initial_context(
        df=df_base, registry=reg_base, target_name=target_name
    )

    base_pipeline_config = build_base_config()

    base_pipeline = [
        BaselineGPStep(config=base_pipeline_config),
        GPSimplificationStep(config=base_pipeline_config)
    ]

    for step in base_pipeline:
        base_context = step.transform(base_context)

    display_pipeline_results(base_context, initial_df=df_base, original_target_name=target_name)

    