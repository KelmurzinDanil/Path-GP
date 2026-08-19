# src/residual_analysis.py
from __future__ import annotations
import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from symbolic.sm_context import SymbolicRegressionContext
from get_pi_complex import PhysicalRegistry, PhysicalDimension
from evaluation import resolve_expression, evaluate_accuracy
from pi_optimizer import compute_dcor
from GP.utils import train_gp_on_dataframe
from config import PipelineConfig, ResidualConfig
from logger import setup_logger

logger = setup_logger("ResidualAnalysis")

@dataclass
class ResidualDiagnosticReport:
    mse_base: float
    r2_base: float
    residual_mean: float
    residual_std: float
    residual_variance: float
    is_deterministic: bool
    feature_importances: Dict[str, float]
    significant_features: List[str]
    suggested_action: str

class ResidualAnalyzer:
    def __init__(self, config: ResidualConfig = None, verbose: bool = True):
        self.config = config or ResidualConfig()
        self.verbose = verbose

    def compute_residual(
        self,
        final_context: SymbolicRegressionContext,
        initial_df: pd.DataFrame,
        original_target_name: str
    ) -> Tuple[np.ndarray, np.ndarray, sp.Expr]:

        gp_formula_pi = final_context.target_expr
        f_phys = sp.simplify(resolve_expression(gp_formula_pi, final_context.symbolic_mapping))
        
        y_true_phys = initial_df[original_target_name].values
        
        current_target = final_context.target_name
        if hasattr(final_context, 'anchor_expr') and final_context.anchor_expr is not None:
            anchor_vals = y_true_phys / (final_context.df[current_target].values + 1e-19)
        else:
            anchor_vals = 1.0

        free_syms = sorted(list(f_phys.free_symbols), key=lambda s: s.name)
        if not free_syms:
            y_pred_phys = np.full_like(y_true_phys, float(f_phys)) * anchor_vals
        else:
            f_compiled = sp.lambdify(free_syms, f_phys, 'numpy')
            args = [initial_df[s.name].values for s in free_syms]
            y_pred_raw = f_compiled(*args)
            if np.isscalar(y_pred_raw):
                y_pred_raw = np.full_like(y_true_phys, y_pred_raw)
            y_pred_phys = y_pred_raw * anchor_vals

        residual = y_true_phys - y_pred_phys
        return y_pred_phys, residual, f_phys

    def diagnose(
        self,
        X_df: pd.DataFrame,
        residual: np.ndarray,
        base_metrics: dict,
        gp_pipeline_config: Optional[PipelineConfig] = None
    ) -> ResidualDiagnosticReport:
        res_var = float(np.var(residual))
        res_std = float(np.std(residual))
        res_mean = float(np.mean(residual))

        if res_var < self.config.min_variance_threshold:
            return ResidualDiagnosticReport(
                mse_base=base_metrics['mse'],
                r2_base=base_metrics['r2'],
                residual_mean=res_mean,
                residual_std=res_std,
                residual_variance=res_var,
                is_deterministic=False,
                feature_importances={},
                significant_features=[],
                suggested_action="STOP: Базовая модель уже дает околонулевую ошибку."
            )

        feature_importances = {}
        for col in X_df.columns:
            dcor_val = compute_dcor(X_df[col].values, residual)
            feature_importances[col] = dcor_val

        feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

        significant_features = [
            col for col, score in feature_importances.items() 
            if score >= self.config.dcor_feature_threshold
        ]

        is_deterministic = True
        if gp_pipeline_config is not None:
            try:
                temp_df = X_df.copy()
                temp_df["_res_target"] = residual
                gp_pipe, _ = train_gp_on_dataframe(temp_df, "_res_target", gp_pipeline_config.gp_config)
                raw_noise = float(gp_pipe.likelihood.noise.item())
                sy = float(gp_pipe.scale_y_factor.item()) if gp_pipe.scale_y_factor is not None else 1.0
                phys_noise_var = raw_noise * (sy ** 2)
                
                noise_fraction = min(1.0, phys_noise_var / (res_var + 1e-12))
                if noise_fraction >= self.config.noise_ratio_stop_threshold:
                    is_deterministic = False
            except Exception:
                pass

        if not significant_features:
            is_deterministic = False
            suggested_action = "WARNING: Ни один признак не коррелирует с ошибкой."
        elif not is_deterministic:
            suggested_action = "WARNING: Высокий уровень случайного шума в остатке."
        else:
            suggested_action = f"RUN_RESIDUAL: Обнаружен явный сигнал в признаках {significant_features}."

        report = ResidualDiagnosticReport(
            mse_base=base_metrics['mse'],
            r2_base=base_metrics['r2'],
            residual_mean=res_mean,
            residual_std=res_std,
            residual_variance=res_var,
            is_deterministic=is_deterministic,
            feature_importances=feature_importances,
            significant_features=significant_features,
            suggested_action=suggested_action
        )

        if self.verbose:
            self._print_diagnostic_summary(report)

        return report

    def _print_diagnostic_summary(self, report: ResidualDiagnosticReport):
        log_msg = "\n" + "="*60 + "\n"
        log_msg += "ДИАГНОСТИКА СТРУКТУРЫ НЕВЯЗКИ (ERROR RESIDUAL ANALYSIS)\n"
        log_msg += "="*60 + "\n"
        log_msg += f"Базовая модель: MSE = {report.mse_base:.6e} | R² = {report.r2_base:.6f}\n"
        log_msg += f"Статистика ошибки: Mean = {report.residual_mean:.4e} | Std = {report.residual_std:.4e} | Var = {report.residual_variance:.4e}\n"
        log_msg += f"Детерминированность остатка: {'ДА (Есть физический сигнал)' if report.is_deterministic else 'НЕТ (Шум)'}\n\n"
        
        log_msg += "Влияние признаков на ошибку (Distance Correlation):\n"
        for feat, score in report.feature_importances.items():
            mark = "★ ВАЖНЫЙ" if feat in report.significant_features else "  фоновый"
            log_msg += f"   - {feat:<15} : dcor = {score:.4f}  [{mark}]\n"
            
        log_msg += f"\nРекомендация: {report.suggested_action}\n"
        log_msg += "="*60
        logger.info(log_msg)

    def create_residual_context(
        self,
        residual_values: np.ndarray,
        original_registry: PhysicalRegistry,
        original_target_name: str,
        custom_df: Optional[pd.DataFrame] = None,
        custom_registry: Optional[PhysicalRegistry] = None,
        diagnostic_report: Optional[ResidualDiagnosticReport] = None
    ) -> SymbolicRegressionContext:
        
        res_target_name = f"{original_target_name}_residual"
        
        if custom_df is not None:
            target_cols_to_drop = [c for c in custom_df.columns if c in [original_target_name, res_target_name]]
            res_df = custom_df.drop(columns=target_cols_to_drop).copy()
            res_df[res_target_name] = residual_values
            
            reg = custom_registry if custom_registry is not None else original_registry
            res_registry = PhysicalRegistry()
            for col in res_df.columns:
                if col != res_target_name:
                    res_registry.register(col, reg.get_dim(col).vector)
        else:
            raise ValueError("Не передан DataFrame для невязки.")

        target_dim = original_registry.get_dim(original_target_name).vector
        res_registry.register(res_target_name, target_dim)

        return SymbolicRegressionContext.create_initial_context(
            df=res_df,
            registry=res_registry,
            target_name=res_target_name
        )

    def reoptimize_combined_formula(
        self, 
        f_combined: sp.Expr, 
        eval_df: pd.DataFrame, 
        y_true: np.ndarray
    ) -> sp.Expr:

        free_syms = sorted(list(f_combined.free_symbols), key=lambda s: s.name)
        if not free_syms:
            return f_combined

        alpha = sp.Symbol('alpha_opt')
        beta = sp.Symbol('beta_opt')
        scaled_expr = alpha * f_combined + beta

        f_eval = sp.lambdify([alpha, beta] + free_syms, scaled_expr, 'numpy')
        args = [eval_df[s.name].values for s in free_syms]

        def loss_fn(p):
            a_val, b_val = p[0], p[1]
            try:
                pred = f_eval(a_val, b_val, *args)
                if np.isscalar(pred): pred = np.full_like(y_true, pred)
                return np.mean((pred - y_true)**2)
            except Exception:
                return 1e10

        res = minimize(loss_fn, [1.0, 0.0], method='Nelder-Mead')
        if res.success:
            a_opt, b_opt = res.x[0], res.x[1]
            if abs(a_opt - 1.0) > 1e-4 or abs(b_opt) > 1e-4:
                return sp.simplify(a_opt * f_combined + b_opt)

        return f_combined

    def combine_and_evaluate(
        self,
        f_base_phys: sp.Expr,
        f_res_phys: sp.Expr,
        initial_df: pd.DataFrame,
        original_target_name: str,
        eval_df: Optional[pd.DataFrame] = None
    ) -> dict:

        f_total_raw = sp.simplify(f_base_phys + f_res_phys)
        
        full_df = initial_df.copy()
        if eval_df is not None:
            for col in eval_df.columns:
                if col not in full_df.columns:
                    full_df[col] = eval_df[col]

        f_total = self.reoptimize_combined_formula(f_total_raw, full_df, full_df[original_target_name].values)

        y_true = full_df[original_target_name].values
        free_syms = sorted(list(f_total.free_symbols), key=lambda s: s.name)
        
        if not free_syms:
            y_pred = np.full_like(y_true, float(f_total))
        else:
            f_compiled = sp.lambdify(free_syms, f_total, 'numpy')
            args = [full_df[s.name].values for s in free_syms]
            y_pred = f_compiled(*args)
            if np.isscalar(y_pred):
                y_pred = np.full_like(y_true, y_pred)

        metrics = evaluate_accuracy(
            sp.Symbol("dummy"), 
            pd.DataFrame({"dummy": y_pred}), 
            y_true
        )

        log_msg = "\n" + "="*60 + "\n"
        log_msg += "ИТОГОВЫЙ СИНТЕЗ: БАЗОВАЯ МОДЕЛЬ + МОДЕЛЬ НЕВЯЗКИ\n"
        log_msg += "="*60 + "\n"
        log_msg += f"1. Базовая формула:      f_base  = {f_base_phys}\n"
        log_msg += f"2. Формула невязки:      e_model = {f_res_phys}\n"
        log_msg += f"3. Полная модель:        f_total = {f_total}\n\n"
        log_msg += f"Итоговая точность (Полная модель):\n"
        log_msg += f"   MSE:  {metrics['mse']:.6e} | R²:   {metrics['r2']:.6f}\n"
        log_msg += f"   MRE:  {metrics['mre']:.4f}%   | MdRE: {metrics['mdre']:.4f}%\n"
        log_msg += "="*60
        logger.info(log_msg)

        return {
            "f_base": f_base_phys,
            "f_residual": f_res_phys,
            "f_total": f_total,
            "metrics": metrics
        }