from __future__ import annotations
import torch
import numpy as np
import pandas as pd
import gpytorch
from typing import Tuple, List, Optional
from .config import GPConfig
from .pipeline import GPRegressionPipeline

def format_kernel_tree(kernel: gpytorch.kernels.Kernel, feature_names: List[str]) -> str:
    if isinstance(kernel, gpytorch.kernels.AdditiveKernel):
        parts = [format_kernel_tree(k, feature_names) for k in kernel.kernels]
        return " + ".join(parts)

    if isinstance(kernel, gpytorch.kernels.ProductKernel):
        parts = [format_kernel_tree(k, feature_names) for k in kernel.kernels]
        return " * ".join(parts)

    if isinstance(kernel, gpytorch.kernels.ScaleKernel):
        base_str = format_kernel_tree(kernel.base_kernel, feature_names)
        return f"Scale({base_str})"

    dims_str = ""
    if kernel.active_dims is not None:
        active_names = [feature_names[i] for i in kernel.active_dims if i < len(feature_names)]
        dims_str = f"[{', '.join(active_names)}]"

    k_name = kernel.__class__.__name__.replace("Kernel", "")
    if isinstance(kernel, gpytorch.kernels.MaternKernel):
        nu_map = {0.5: "12", 1.5: "32", 2.5: "52"}
        k_name = f"Matern{nu_map.get(kernel.nu, str(kernel.nu))}"
    elif isinstance(kernel, gpytorch.kernels.PolynomialKernel):
        k_name = f"Polynomial(p={kernel.power})"
    elif isinstance(kernel, gpytorch.kernels.SpectralMixtureKernel):
        k_name = f"SpectralMixture(Q={kernel.num_mixtures})"

    return f"{k_name}({dims_str})" if dims_str else k_name


def format_kernel_summary(
    covar_module: gpytorch.kernels.Kernel,
    feature_names: List[str],
    scale_x_factor: Optional[torch.Tensor] = None,
    scale_y_factor: Optional[torch.Tensor] = None
) -> str:
    sx = scale_x_factor.cpu().numpy() if scale_x_factor is not None else np.ones(len(feature_names))
    sy = float(scale_y_factor.cpu().item()) if scale_y_factor is not None else 1.0

    tree_repr = format_kernel_tree(covar_module, feature_names)
    lines = [f"   Топология ядра: {tree_repr}"]

    for name, mod in covar_module.named_modules():
        if isinstance(mod, gpytorch.kernels.ScaleKernel):
            try:
                raw_os = float(mod.outputscale.item())
                phys_os = raw_os * (sy ** 2)
                lines.append(f"     -> [Scale] Outputscale (σ²): {raw_os:.4f} (в физ. ед.: {phys_os:.4e})")
            except Exception:
                pass

        if getattr(mod, "has_lengthscale", False) and hasattr(mod, "lengthscale") and mod.lengthscale is not None:
            try:
                ls = mod.lengthscale.squeeze().detach().cpu().numpy()
                active_indices = list(mod.active_dims) if mod.active_dims is not None else list(range(len(feature_names)))
                k_title = mod.__class__.__name__.replace("Kernel", "")

                if np.ndim(ls) == 0:
                    val_norm = float(ls)
                    lines.append(f"     -> [{k_title}] Изотропный Lengthscale: {val_norm:.4f}")
                else:
                    lines.append(f"     -> [{k_title}] ARD Lengthscales:")
                    for idx, dim_idx in enumerate(active_indices):
                        if idx < len(ls) and dim_idx < len(feature_names):
                            val_norm = float(ls[idx])
                            val_phys = val_norm * float(sx[dim_idx])
                            lines.append(f"          * {feature_names[dim_idx]:<12} : {val_phys:.4f} (норм: {val_norm:.4f})")
            except Exception:
                pass

        if isinstance(mod, gpytorch.kernels.PeriodicKernel):
            try:
                p_len = float(mod.period_length.squeeze().item())
                lines.append(f"     -> [Periodic] Period Length: {p_len:.4f}")
            except Exception:
                pass

    return "\n".join(lines)


def get_quick_kernel_info(covar_module: gpytorch.kernels.Kernel) -> str:
    info_parts = []
    for mod in covar_module.modules():
        if getattr(mod, "has_lengthscale", False) and hasattr(mod, "lengthscale") and mod.lengthscale is not None:
            try:
                ls = mod.lengthscale.squeeze().detach().cpu().numpy()
                if np.ndim(ls) == 0:
                    info_parts.append(f"ls: {ls.item():.3f}")
                else:
                    ls_formatted = ",".join([f"{x:.2f}" for x in ls])
                    info_parts.append(f"ls: [{ls_formatted}]")
            except Exception:
                pass
        if isinstance(mod, gpytorch.kernels.ScaleKernel):
            try:
                info_parts.append(f"os: {mod.outputscale.item():.3f}")
            except Exception:
                pass
    return " | ".join(info_parts[:4])

def train_gp_model(
    train_x: torch.Tensor, 
    train_y: torch.Tensor, 
    config: GPConfig
) -> Tuple[GPRegressionPipeline, List[float]]:
    pipeline = GPRegressionPipeline(config)
    loss_history = pipeline.fit(train_x, train_y)
    return pipeline, loss_history


def train_gp_on_dataframe(
    dataset: pd.DataFrame, 
    target_name: str, 
    config: GPConfig
) -> Tuple[GPRegressionPipeline, List[float]]:
    feature_cols = [col for col in dataset.columns if col != target_name]
    train_x = torch.tensor(dataset[feature_cols].values, dtype=torch.float64)
    train_y = torch.tensor(dataset[target_name].values, dtype=torch.float64)
    return train_gp_model(train_x, train_y, config)