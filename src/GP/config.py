from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, Union, Type, Callable, Tuple, Sequence, List
import gpytorch
import torch

@dataclass
class KernelSpec:
    active_dims: Optional[Sequence[int]] = None

    def __add__(self, other: Union[KernelSpec, KernelType]) -> AddKernelConfig:
        other_spec = _ensure_kernel_spec(other)
        return AddKernelConfig(left=self, right=other_spec)

    def __radd__(self, other: Union[KernelSpec, KernelType]) -> AddKernelConfig:
        other_spec = _ensure_kernel_spec(other)
        return AddKernelConfig(left=other_spec, right=self)

    def __mul__(self, other: Union[KernelSpec, KernelType]) -> MulKernelConfig:
        other_spec = _ensure_kernel_spec(other)
        return MulKernelConfig(left=self, right=other_spec)

    def __rmul__(self, other: Union[KernelSpec, KernelType, float, int]) -> KernelSpec:
        if isinstance(other, (int, float)):
            return ScaleKernelConfig(base=self)
        other_spec = _ensure_kernel_spec(other)
        return MulKernelConfig(left=other_spec, right=self)

@dataclass
class AddKernelConfig(KernelSpec):
    left: KernelSpec = field(default_factory=lambda: RBF())
    right: KernelSpec = field(default_factory=lambda: RBF())

    def __repr__(self) -> str:
        return f"({self.left} + {self.right})"


@dataclass
class MulKernelConfig(KernelSpec):
    left: KernelSpec = field(default_factory=lambda: RBF())
    right: KernelSpec = field(default_factory=lambda: RBF())

    def __repr__(self) -> str:
        return f"({self.left} * {self.right})"


@dataclass
class ScaleKernelConfig(KernelSpec):
    base: KernelSpec = field(default_factory=lambda: RBF())
    outputscale_prior_params: Optional[Tuple[float, float]] = (2.0, 0.15)
    outputscale_constraint: Optional[Tuple[float, float]] = None

    def __repr__(self) -> str:
        return f"Scale({self.base})"


def Scale(
    kernel: KernelSpec,
    outputscale_prior_params: Optional[Tuple[float, float]] = (2.0, 0.15),
    outputscale_constraint: Optional[Tuple[float, float]] = None,
) -> ScaleKernelConfig:
    spec = _ensure_kernel_spec(kernel)
    return ScaleKernelConfig(
        base=spec,
        outputscale_prior_params=outputscale_prior_params,
        outputscale_constraint=outputscale_constraint
    )

@dataclass
class StationaryKernelSpec(KernelSpec):
    ard: bool = True
    use_lengthscale_prior: bool = True
    lengthscale_prior_params: Tuple[float, float] = (3.0, 0.5)
    lengthscale_bounds: Tuple[float, float] = (0.05, 10.0)


@dataclass
class RBF(StationaryKernelSpec):
    def __repr__(self) -> str:
        dims = f", dims={list(self.active_dims)}" if self.active_dims is not None else ""
        ard = f", ard={self.ard}" if not self.ard else ""
        return f"RBF({dims}{ard})".replace("(, ", "(")


@dataclass
class Matern12(StationaryKernelSpec):
    def __repr__(self) -> str:
        dims = f", dims={list(self.active_dims)}" if self.active_dims is not None else ""
        return f"Matern12({dims})".replace("(, ", "(")


@dataclass
class Matern32(StationaryKernelSpec):
    def __repr__(self) -> str:
        dims = f", dims={list(self.active_dims)}" if self.active_dims is not None else ""
        return f"Matern32({dims})".replace("(, ", "(")


@dataclass
class Matern52(StationaryKernelSpec):
    def __repr__(self) -> str:
        dims = f", dims={list(self.active_dims)}" if self.active_dims is not None else ""
        return f"Matern52({dims})".replace("(, ", "(")


@dataclass
class RQ(StationaryKernelSpec):
    alpha_prior_params: Optional[Tuple[float, float]] = None

    def __repr__(self) -> str:
        dims = f", dims={list(self.active_dims)}" if self.active_dims is not None else ""
        return f"RQ({dims})".replace("(, ", "(")


@dataclass
class Cauchy(StationaryKernelSpec):
    def __repr__(self) -> str:
        dims = f", dims={list(self.active_dims)}" if self.active_dims is not None else ""
        return f"Cauchy({dims})".replace("(, ", "(")


@dataclass
class Periodic(KernelSpec):
    ard: bool = False
    use_period_prior: bool = False
    period_prior_params: Tuple[float, float] = (3.0, 0.5)
    period_bounds: Tuple[float, float] = (0.05, 10.0)
    use_lengthscale_prior: bool = True
    lengthscale_prior_params: Tuple[float, float] = (3.0, 0.5)
    lengthscale_bounds: Tuple[float, float] = (0.05, 10.0)

    def __repr__(self) -> str:
        dims = f", dims={list(self.active_dims)}" if self.active_dims is not None else ""
        return f"Periodic({dims})".replace("(, ", "(")


@dataclass
class Cosine(KernelSpec):
    ard: bool = True
    use_period_prior: bool = False
    period_prior_params: Tuple[float, float] = (3.0, 0.5)
    period_bounds: Tuple[float, float] = (0.05, 10.0)

    def __repr__(self) -> str:
        dims = f", dims={list(self.active_dims)}" if self.active_dims is not None else ""
        return f"Cosine({dims})".replace("(, ", "(")


@dataclass
class Linear(KernelSpec):
    variance_prior_params: Optional[Tuple[float, float]] = None

    def __repr__(self) -> str:
        dims = f", dims={list(self.active_dims)}" if self.active_dims is not None else ""
        return f"Linear({dims})".replace("(, ", "(")


@dataclass
class Polynomial(KernelSpec):
    power: int = 2
    offset_prior_params: Optional[Tuple[float, float]] = None

    def __repr__(self) -> str:
        dims = f", dims={list(self.active_dims)}" if self.active_dims is not None else ""
        return f"Polynomial(power={self.power}{dims})"


@dataclass
class SpectralMixture(KernelSpec):
    num_mixtures: int = 4

    def __repr__(self) -> str:
        dims = f", dims={list(self.active_dims)}" if self.active_dims is not None else ""
        return f"SpectralMixture(mixtures={self.num_mixtures}{dims})"
    
KernelType = Union[
    Literal[
        "rbf",
        "matern_12",
        "matern_32",
        "matern_52",
        "rq",
        "periodic",
        "cosine",
        "spectral_mixture",
        "cauchy",
        "linear",
        "polynomial"
    ],
    Type[gpytorch.kernels.Kernel],
    gpytorch.kernels.Kernel,
    KernelSpec
]


@dataclass
class KernelConfig(KernelSpec):

    type: KernelType = "rbf"
    scale_kernel: bool = True
    ard: bool = True

    use_lengthscale_prior: bool = True
    lengthscale_prior_params: Tuple[float, float] = (3.0, 0.5)

    lengthscale_kernel: Optional[Any] = None
    outputscale_kernel: Optional[Any] = None

    polynomial_power: int = 4

    def to_spec(self) -> KernelSpec:
        if isinstance(self.type, KernelSpec):
            spec = self.type
        elif isinstance(self.type, str):
            t = self.type.lower()
            if t == "rbf":
                spec = RBF(ard=self.ard, use_lengthscale_prior=self.use_lengthscale_prior, lengthscale_prior_params=self.lengthscale_prior_params)
            elif t in ["matern_12", "matern12"]:
                spec = Matern12(ard=self.ard, use_lengthscale_prior=self.use_lengthscale_prior, lengthscale_prior_params=self.lengthscale_prior_params)
            elif t in ["matern_32", "matern32"]:
                spec = Matern32(ard=self.ard, use_lengthscale_prior=self.use_lengthscale_prior, lengthscale_prior_params=self.lengthscale_prior_params)
            elif t in ["matern_52", "matern52"]:
                spec = Matern52(ard=self.ard, use_lengthscale_prior=self.use_lengthscale_prior, lengthscale_prior_params=self.lengthscale_prior_params)
            elif t == "linear":
                spec = Linear()
            elif t == "polynomial":
                spec = Polynomial(power=self.polynomial_power)
            elif t == "periodic":
                spec = Periodic(ard=self.ard)
            elif t == "cosine":
                spec = Cosine(ard=self.ard)
            elif t == "rq":
                spec = RQ(ard=self.ard)
            elif t == "cauchy":
                spec = Cauchy(ard=self.ard)
            elif t in ["spectral_mixture", "sm"]:
                num_m = self.lengthscale_kernel if isinstance(self.lengthscale_kernel, int) else 4
                return SpectralMixture(num_mixtures=num_m, active_dims=self.active_dims)
            else:
                raise ValueError(f"Неизвестный тип ядра в legacy KernelConfig: {self.type}")
        else:
            return self

        spec.active_dims = self.active_dims
        if self.scale_kernel:
            return ScaleKernelConfig(base=spec)
        return spec


def _ensure_kernel_spec(kernel: Union[KernelSpec, KernelType, str]) -> KernelSpec:
    if isinstance(kernel, KernelConfig):
        return kernel.to_spec()
    if isinstance(kernel, KernelSpec):
        return kernel
    if isinstance(kernel, str):
        return KernelConfig(type=kernel).to_spec()
    legacy = KernelConfig(type=kernel)
    return legacy.to_spec()


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
class LikelihoodConfig:
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    mean_type: MeanType = "constant"
    kernel: Union[KernelSpec, KernelConfig, KernelType] = field(default_factory=KernelConfig)
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