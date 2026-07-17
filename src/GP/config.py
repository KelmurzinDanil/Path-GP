from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, Union, Type, Callable
import gpytorch
import torch

KernelType = Union[
    Literal[
        "rbf", 
        "matern_32", 
        "matern_52", 
        "linear", 
        "rq", 
        "periodic", 
        "cosine", 
        "polynomial"
    ], 
    Type[gpytorch.kernels.Kernel],
    gpytorch.kernels.Kernel
]

MeanType = Union[
    Literal["constant", "linear", "zero"],
    Type[gpytorch.means.Mean],
    gpytorch.means.Mean
]

OptimizerType = Union[
    Literal["adam", "sgd", "lbfgs"],
    Type[torch.optim.Optimizer],
    torch.optim.Optimizer
]

LossType = Union[
    Literal["mll", "loo", "elbo"],
    Type[gpytorch.mlls.MarginalLogLikelihood],
    gpytorch.mlls.MarginalLogLikelihood
]

LikelihoodType = Union[
    Literal["gaussian", "student_t", "bernoulli"],
    Type[gpytorch.likelihoods.Likelihood],
    gpytorch.likelihoods.Likelihood
]

@dataclass
class KernelConfig:
    type: KernelType = "rbf"
    scale_kernel: bool = True
    ard: bool = False

    lengthscale_kernel: Optional[Any] = None 
    outputscale_kernel: Optional[Any] = None

@dataclass
class LikelihoodConfig:
    type: LikelihoodType = "gaussian"
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class ModelConfig:
    model_type: Literal["exact", "approximate"] = "exact"
    mean_type: MeanType = "constant"

    kernel: KernelConfig = field(default_factory=KernelConfig)
    likelihood: LikelihoodConfig = field(default_factory=LikelihoodConfig)

@dataclass
class TrainingConfig:
    lr: float = 0.1
    epochs: int = 100
    early_stopping_patience: int = 15

    optimizer: OptimizerType = "adam"
    loss_type: LossType = "mll"
    early_stopping_patience: Optional[int] = None

    verbose: bool = True

    loss_modifier: Optional[Callable[[Any, Any, Any, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None


@dataclass
class GPConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42