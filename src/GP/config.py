from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, Union, Type, Callable, Tuple
import gpytorch
import torch

KernelType = Union[
    Literal[
        "rbf", 
        "matern_32", 
        "matern_52", 
        "rq", 
        "periodic", 
        "cosine", 
        "spectral_mixture",  
        "cauchy" 
    ], 
    Type[gpytorch.kernels.Kernel],
    gpytorch.kernels.Kernel
]

MeanType = Union[
    Literal["constant", "zero"],
    Type[gpytorch.means.Mean],
    gpytorch.means.Mean
]

OptimizerType = Union[
    Literal["adam", "sgd", "lbfgs"],
    Type[torch.optim.Optimizer],
    torch.optim.Optimizer
]

LossType = Union[
    Literal["mll", "loo"],
    Type[gpytorch.mlls.MarginalLogLikelihood],
    gpytorch.mlls.MarginalLogLikelihood
]


@dataclass
class KernelConfig:
    type: KernelType = "rbf"
    scale_kernel: bool = True
    ard: bool = True

    use_lengthscale_prior: bool = True
    lengthscale_prior_params: Tuple[float, float] = (3.0, 0.5)

    lengthscale_kernel: Optional[Any] = None 
    outputscale_kernel: Optional[Any] = None

@dataclass
class LikelihoodConfig:
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelConfig:
    mean_type: MeanType = "constant"
    kernel: KernelConfig = field(default_factory=KernelConfig)
    likelihood: LikelihoodConfig = field(default_factory=LikelihoodConfig)

@dataclass
class TrainingConfig:
    lr: float = 0.1
    epochs: int = 61
    early_stopping_patience: int = 15

    optimizer: OptimizerType = "adam"
    loss_type: LossType = "mll"

    verbose: bool = True

    cholesky_jitter: float = 1e-3
    fast_solves: bool = False      
    fast_log_prob: bool = False

    loss_modifier: Optional[Callable[[Any, Any, Any, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None

@dataclass
class ScalingConfig:
    enabled: bool = True

@dataclass
class GPConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    seed: int = 42