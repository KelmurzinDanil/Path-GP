from __future__ import annotations
import torch
import gpytorch

from gpytorch.means import Mean, ConstantMean, LinearMean, ZeroMean
from gpytorch.kernels import (
    Kernel,
    RBFKernel,
    MaternKernel,
    LinearKernel,
    PolynomialKernel,
    PeriodicKernel,
    CosineKernel,
    RQKernel,
    SpectralMixtureKernel,
    ScaleKernel,
    AdditiveKernel,
    ProductKernel
)
from gpytorch.likelihoods import Likelihood, GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood

from typing import Optional, Any, Iterable, Union, Sequence
from .config import (
    ModelConfig,
    KernelConfig,
    TrainingConfig,
    KernelType,
    MeanType,
    KernelSpec,
    AddKernelConfig,
    MulKernelConfig,
    ScaleKernelConfig,
    StationaryKernelSpec,
    RBF,
    Matern12,
    Matern32,
    Matern52,
    RQ,
    Cauchy,
    Periodic,
    Cosine,
    Linear,
    Polynomial,
    SpectralMixture,
    _ensure_kernel_spec
)

class CauchyKernel(gpytorch.kernels.Kernel):
    """Ядро Коши с тяжелыми хвостами: k(r) = (1 + r^2 / l^2)^(-alpha)."""
    is_stationary = True
    has_lengthscale = True

    def __init__(
        self,
        ard_num_dims: Optional[int] = None,
        active_dims: Optional[Sequence[int]] = None,
        **kwargs
    ):
        super().__init__(has_lengthscale=True, ard_num_dims=ard_num_dims, active_dims=active_dims, **kwargs)
        self.register_parameter(
            name="raw_alpha",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_constraint("raw_alpha", gpytorch.constraints.Positive())

    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=self.raw_alpha.dtype, device=self.raw_alpha.device)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        dist_sq = self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params)
        return (1.0 + dist_sq).pow(-self.alpha)


def _apply_lengthscale_config(
    kernel: gpytorch.kernels.Kernel,
    spec: StationaryKernelSpec,
    effective_dim: Optional[int]
):
    if getattr(kernel, "has_lengthscale", False):
        low, high = getattr(spec, "lengthscale_bounds", (0.05, 10.0))
        kernel.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(low, high))

        if getattr(spec, "ard", False) and effective_dim is not None:
            kernel.lengthscale = torch.ones(*kernel.batch_shape, 1, effective_dim) * 1.0
        else:
            kernel.lengthscale = torch.tensor(1.0)

        if getattr(spec, "use_lengthscale_prior", False):
            conc, rate = spec.lengthscale_prior_params
            kernel.register_prior(
                "lengthscale_prior",
                gpytorch.priors.GammaPrior(concentration=conc, rate=rate),
                "lengthscale"
            )


def build_kernel_from_spec(
    spec: Union[KernelSpec, KernelConfig, KernelType],
    input_dim: Optional[int] = None
) -> gpytorch.kernels.Kernel:

    if isinstance(spec, gpytorch.kernels.Kernel):
        return spec

    if isinstance(spec, type) and issubclass(spec, gpytorch.kernels.Kernel):
        return spec(ard_num_dims=input_dim)

    spec = _ensure_kernel_spec(spec)

    if isinstance(spec, AddKernelConfig):
        left_k = build_kernel_from_spec(spec.left, input_dim)
        right_k = build_kernel_from_spec(spec.right, input_dim)
        return left_k + right_k

    if isinstance(spec, MulKernelConfig):
        left_k = build_kernel_from_spec(spec.left, input_dim)
        right_k = build_kernel_from_spec(spec.right, input_dim)
        return left_k * right_k

    if isinstance(spec, ScaleKernelConfig):
        base_k = build_kernel_from_spec(spec.base, input_dim)
        
        if isinstance(base_k, gpytorch.kernels.SpectralMixtureKernel):
            return base_k

        scale_k = ScaleKernel(base_k, active_dims=spec.active_dims)
        
        if spec.outputscale_prior_params is not None:
            conc, rate = spec.outputscale_prior_params
            scale_k.register_prior(
                "outputscale_prior",
                gpytorch.priors.GammaPrior(concentration=conc, rate=rate),
                "outputscale"
            )
        if spec.outputscale_constraint is not None:
            low, high = spec.outputscale_constraint
            scale_k.register_constraint("raw_outputscale", gpytorch.constraints.Interval(low, high))
        return scale_k

    active_dims = tuple(spec.active_dims) if spec.active_dims is not None else None
    if active_dims is not None:
        effective_dim = len(active_dims)
    else:
        effective_dim = input_dim

    ard_dims = effective_dim if getattr(spec, "ard", False) else None

    if isinstance(spec, RBF):
        kernel = RBFKernel(ard_num_dims=ard_dims, active_dims=active_dims)
        _apply_lengthscale_config(kernel, spec, effective_dim)
        return kernel

    if isinstance(spec, Matern12):
        kernel = MaternKernel(nu=0.5, ard_num_dims=ard_dims, active_dims=active_dims)
        _apply_lengthscale_config(kernel, spec, effective_dim)
        return kernel

    if isinstance(spec, Matern32):
        kernel = MaternKernel(nu=1.5, ard_num_dims=ard_dims, active_dims=active_dims)
        _apply_lengthscale_config(kernel, spec, effective_dim)
        return kernel

    if isinstance(spec, Matern52):
        kernel = MaternKernel(nu=2.5, ard_num_dims=ard_dims, active_dims=active_dims)
        _apply_lengthscale_config(kernel, spec, effective_dim)
        return kernel

    if isinstance(spec, RQ):
        kernel = RQKernel(ard_num_dims=ard_dims, active_dims=active_dims)
        _apply_lengthscale_config(kernel, spec, effective_dim)
        if spec.alpha_prior_params is not None:
            conc, rate = spec.alpha_prior_params
            kernel.register_prior(
                "alpha_prior",
                gpytorch.priors.GammaPrior(concentration=conc, rate=rate),
                "alpha"
            )
        return kernel

    if isinstance(spec, Cauchy):
        kernel = CauchyKernel(ard_num_dims=ard_dims, active_dims=active_dims)
        _apply_lengthscale_config(kernel, spec, effective_dim)
        return kernel

    if isinstance(spec, Linear):
        kernel = LinearKernel(active_dims=active_dims)
        if spec.variance_prior_params is not None:
            conc, rate = spec.variance_prior_params
            kernel.register_prior(
                "variance_prior",
                gpytorch.priors.GammaPrior(concentration=conc, rate=rate),
                "variance"
            )
        return kernel

    if isinstance(spec, Polynomial):
        kernel = PolynomialKernel(power=spec.power, active_dims=active_dims)
        if spec.offset_prior_params is not None:
            conc, rate = spec.offset_prior_params
            kernel.register_prior(
                "offset_prior",
                gpytorch.priors.GammaPrior(concentration=conc, rate=rate),
                "offset"
            )
        return kernel

    if isinstance(spec, Periodic):
        kernel = PeriodicKernel(active_dims=active_dims)
        low_p, high_p = spec.period_bounds
        kernel.register_constraint("raw_period_length", gpytorch.constraints.Interval(low_p, high_p))
        if spec.use_period_prior:
            conc, rate = spec.period_prior_params
            kernel.register_prior(
                "period_length_prior",
                gpytorch.priors.GammaPrior(concentration=conc, rate=rate),
                "period_length"
            )
        if spec.use_lengthscale_prior:
            conc, rate = spec.lengthscale_prior_params
            kernel.register_prior(
                "lengthscale_prior",
                gpytorch.priors.GammaPrior(concentration=conc, rate=rate),
                "lengthscale"
            )
        return kernel

    if isinstance(spec, Cosine):
        kernel = CosineKernel(ard_num_dims=ard_dims, active_dims=active_dims)
        low_p, high_p = spec.period_bounds
        kernel.register_constraint("raw_period_length", gpytorch.constraints.Interval(low_p, high_p))
        if spec.use_period_prior:
            conc, rate = spec.period_prior_params
            kernel.register_prior(
                "period_length_prior",
                gpytorch.priors.GammaPrior(concentration=conc, rate=rate),
                "period_length"
            )
        return kernel

    if isinstance(spec, SpectralMixture):
        num_dims_val = effective_dim if effective_dim is not None else 1
        kernel = SpectralMixtureKernel(
            num_mixtures=spec.num_mixtures,
            num_dims=num_dims_val,
            active_dims=active_dims
        )
        return kernel

    raise ValueError(f"Неподдерживаемая спецификация ядра: {spec}")


def get_kernel(
    config: Union[KernelConfig, KernelSpec, KernelType],
    input_dim: Optional[int] = None
) -> gpytorch.kernels.Kernel:
    return build_kernel_from_spec(config, input_dim=input_dim)


def get_mean(config: ModelConfig, input_dim: Optional[int] = None) -> gpytorch.means.Mean:
    mean_type = config.mean_type

    if isinstance(mean_type, gpytorch.means.Mean):
        return mean_type

    if isinstance(mean_type, type) and issubclass(mean_type, gpytorch.means.Mean):
        return mean_type()

    if mean_type == "zero":
        return gpytorch.means.ZeroMean()
    elif mean_type == "constant":
        return gpytorch.means.ConstantMean()
    else:
        raise ValueError(f"Неизвестный тип среднего (mean type): {mean_type}")


def get_likelihood(config: ModelConfig, train_y: torch.Tensor = None) -> gpytorch.likelihoods.Likelihood:
    lh_config = config.likelihood
    kwargs = lh_config.extra_kwargs

    likelihood = gpytorch.likelihoods.GaussianLikelihood(**kwargs)

    if train_y is not None:
        y_var = float(torch.var(train_y).item())
        if y_var < 1e-8:
            y_var = 1.0

        min_noise = 1e-6
        max_noise = max(0.05 * y_var, 1e-2)

        likelihood.noise_covar.register_constraint(
            "raw_noise",
            gpytorch.constraints.Interval(min_noise, max_noise)
        )
        init_noise = min(max(min_noise, 0.001 * y_var), max_noise * 0.9)
        likelihood.noise = torch.tensor(init_noise, dtype=train_y.dtype)
    else:
        likelihood.noise_covar.register_constraint(
            "raw_noise",
            gpytorch.constraints.Interval(1e-6, 0.05)
        )

    return likelihood


def get_loss_fn(
    model: gpytorch.models.GP,
    likelihood: gpytorch.likelihoods.Likelihood,
    training_config: TrainingConfig,
    num_data: Optional[int] = None
) -> gpytorch.mlls.MarginalLogLikelihood:

    loss_type = training_config.loss_type

    if isinstance(loss_type, gpytorch.mlls.MarginalLogLikelihood):
        return loss_type

    if isinstance(loss_type, type) and issubclass(loss_type, gpytorch.mlls.MarginalLogLikelihood):
        if num_data is not None:
            return loss_type(likelihood, model, num_data=num_data)
        return loss_type(likelihood, model)

    if loss_type == "mll":
        return gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    elif loss_type == "loo":
        return gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood, model)
    else:
        raise ValueError(f"Неизвестный тип функции потерь: {loss_type}")


def get_optimizer(
    model_parameters: Iterable[torch.nn.Parameter],
    config: TrainingConfig
) -> torch.optim.Optimizer:

    opt_type = config.optimizer
    lr = config.lr

    if isinstance(opt_type, torch.optim.Optimizer):
        return opt_type

    if isinstance(opt_type, type) and issubclass(opt_type, torch.optim.Optimizer):
        return opt_type(model_parameters, lr=lr)

    if opt_type == "adam":
        return torch.optim.Adam(model_parameters, lr=lr)
    elif opt_type == "sgd":
        return torch.optim.SGD(model_parameters, lr=lr)
    elif opt_type == "lbfgs":
        return torch.optim.LBFGS(model_parameters, lr=lr)
    else:
        raise ValueError(f"Неизвестный тип оптимизатора: {opt_type}")