
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, Union, Type, Callable, Tuple, Sequence, List
from GP.config import GPConfig


@dataclass
class ResidualConfig:
    min_variance_threshold: float = 1e-8     
    noise_ratio_stop_threshold: float = 0.85  
    dcor_feature_threshold: float = 0.15      
    filter_unimportant_features: bool = False 
    top_k_features: Optional[int] = None      

@dataclass
class BruteForceConfig:
    max_length: int = 8
    optimize_constants: bool = False
    complexity_penalty: float = 0.05
    allowed_constants: List[float] = field(
        default_factory=lambda: [1.0, 2.0]
    )

    allowed_ops: List[str] = field(
        default_factory=lambda: [
            "add", "sub", "mul", "div",
            "pow", "sqrt", "log", "exp",
            "sin", "cos", "tan"
        ]
    )

    k_best: int = 50

    # Snap
    snap_rel_tol: float = 1.15   # было 1.05
    snap_abs_tol: float = 1e-2   # было 1e-6



@dataclass
class DimensionalAnalysisConfig:
    min_quality_score: float = 0.05
    max_redundancy_dcor: float = 0.50
    sparsity_penalty: float = 0.01
    max_integer_search: int = 3
    ste_l1_lambda: float = 0.04
    ste_epochs: int = 300
    ste_lr: float = 0.1

@dataclass
class FriedmanSeparabilityConfig:
    h2_threshold: float = 0.015
    grid_density: int = 15
    bg_samples: int = 200
    max_allowed_distance: float = 1.2
    min_contrast: float = 1.5
    bf_max_length: int = 5
    top_k_validate: int = 3

@dataclass
class CompositionalityConfig:
    grid_density: int = 15
    n_slices: int = 20
    slice_spearman_threshold: float = 0.92
    template_spearman_threshold: float = 0.90
    bf_max_length: int = 7

@dataclass
class SimplificationConfig:
    max_depth: int = 7
    local_bf_max_length: int = 5
    final_bf_max_length: int = 7
    base_k_sigma: float = 3.0
    k_sigma_multipliers: List[float] = field(default_factory=lambda: [1.0, 2.0])
    loss_degradation_tolerance: float = 0.55

@dataclass
class PipelineConfig:
    gp_config: GPConfig
    verbose: bool = True
    brute_force: BruteForceConfig = field(default_factory=BruteForceConfig)
    dim_analysis: DimensionalAnalysisConfig = field(default_factory=DimensionalAnalysisConfig)
    friedman: FriedmanSeparabilityConfig = field(default_factory=FriedmanSeparabilityConfig)
    compositionality: CompositionalityConfig = field(default_factory=CompositionalityConfig)
    simplification: SimplificationConfig = field(default_factory=SimplificationConfig)