from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Any, Callable
import os
import json
import optuna
import torch
import sympy as sp
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import gpytorch

from scipy.stats import spearmanr
from .sm_context import SymbolicRegressionContext, DimensionalityEvaluator
from get_pi_complex import DimensionalError, PhysicalRegistry
from GP.config import GPConfig, ModelConfig, KernelConfig, TrainingConfig
from GP.pipeline import GPRegressionPipeline
from function_analysis.simplify import *
from function_analysis.py_brute_force import BruteForceRunner

simplifiers = [
    AdditiveSeparabilitySimplifier(), 
    MultiplicationSeparabilitySimplifier(),
    TranslationalSymmetrySimplifier(),
    AdditionSymmetrySimplifier(),
    LargeScaleSymmetrySimplifier(),
    MultiplySymmetrySimplifier(),
    CompositionalitySimplifier(),
    GeneralizedSymmetrySimplifier(),
    GeneralAdditiveSeparabilitySimplifier()
]

def eval_gp_r2(pipeline: GPRegressionPipeline, dataset: pd.DataFrame, target_name: str) -> float:
    feature_cols = [col for col in dataset.columns if col != target_name]
    x_tensor = torch.tensor(dataset[feature_cols].values, dtype=torch.float64)
    y_true = dataset[target_name].values
    
    pipeline.model.eval()
    pipeline.likelihood.eval()
    with torch.no_grad():
        pred_dist = pipeline.predict(x_tensor)
        y_pred = pred_dist.mean.numpy()
        
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))
    return max(r2, -1.0)

def svd_polynomial(df_slice: pd.DataFrame, group_cols: list, registry: PhysicalRegistry) -> Optional[sp.Expr]:
    if len(group_cols) != 2:
        return None

    u_name, v_name = group_cols[0], group_cols[1]
    u = df_slice[u_name].values
    v = df_slice[v_name].values
    Y = df_slice["target"].values

    s_u, s_v = sp.Symbol(u_name), sp.Symbol(v_name)

    templates = [
        (s_u - s_v, u - v),                     # Разность / Сдвиг
        (s_u + s_v, u + v),                     # Линейное сложение
        (s_u**2 + s_v**2, u**2 + v**2),         # Сумма квадратов
        (s_u**2 - s_v**2, u**2 - v**2),         # Разность квадратов
        (s_u * s_v, u * v),                     # Произведение
        (s_u / s_v, u / (v + 1e-9)),            # Отношение
        ((s_u - s_v)**2, (u - v)**2),           # Квадрат разности
        ((s_u + s_v)**2, (u + v)**2)            # Квадрат суммы
    ]

    y_targets = [
        Y,                                      # Прямой таргет
        1.0 / (Y + 1e-9),                       # Обратный таргет
        np.log(np.abs(Y) + 1e-9)                # Логарифмический таргет
    ]

    for expr, t_vals in templates:
        try:
            DimensionalityEvaluator.evaluate(expr, registry)
        except (DimensionalError, KeyError):
            continue

        for y_t in y_targets:
            try:
                if np.std(t_vals) < 1e-9 or np.std(y_t) < 1e-9:
                    continue

                corr = np.corrcoef(t_vals, y_t)[0, 1]
                r2 = float(corr) ** 2 

                if not np.isnan(r2) and r2 > 0.998:
                    print(f"-> [Tier 1 SVD Filter] Найдена точная физическая структура: {expr} (R2 = {r2:.6f})")
                    return expr
            except Exception:
                continue

    return None

def find_best_h_for_group(gp_model, context: 'SymbolicRegressionContext', group_cols: list, num_test_points=None, local_bf_max_length=6,
                          optimize_constants=True, allowed_constants=[1.0, 2.0],
                          allowed_ops=["add", "sub", "mul", "div", "sin", "cos", "exp", "log"], k_best=50) -> sp.Expr:
    all_features = context.get_active_features()

    pts_all = torch.tensor(context.df[all_features].values, dtype=torch.float64)

    gp_model.model.eval()
    gp_model.likelihood.eval()

    with torch.no_grad():
        predictions = gp_model.predict(pts_all)
        Y_slice = predictions.mean.numpy()

    df_slice_data_list = {col: context.df[col].values for col in group_cols}
    df_slice_data_list["target"] = Y_slice
    df_slice = pd.DataFrame(df_slice_data_list)

    fast_h = svd_polynomial(df_slice, group_cols, context.registry)
    if fast_h is not None:
        return fast_h
    
    registry_slice = PhysicalRegistry()
    registry_slice.register("target", context.registry.get_dim(context.target_name).vector)    
    
    mapping_slice = {}
    for col_name in group_cols:
        registry_slice.register(col_name, context.registry.get_dim(col_name).vector)
        mapping_slice[col_name] = sp.Symbol(col_name)

    slice_context = SymbolicRegressionContext(
        df=df_slice,
        registry=registry_slice,
        target_name="target",
        symbolic_mapping=mapping_slice,
        target_expr=sp.Symbol("target")
    )

    runner = BruteForceRunner(
        max_length=local_bf_max_length,
        optimize_constants=optimize_constants, 
        allowed_constants=allowed_constants,
        allowed_ops=allowed_ops,
        k_best=k_best
    )

    print(f"[Generalized Symmetry] Запуск C++ brute-force для группы {group_cols} на реальных данных...")

    def evaluate_expr_mse(expr: sp.Expr, df_s: pd.DataFrame) -> float:
        try:
            symbols = sorted(list(expr.free_symbols), key=lambda s: s.name)
            f_compiled = sp.lambdify(symbols, expr, 'numpy')
            args = [df_s[s.name].values for s in symbols]
            y_pred = f_compiled(*args)
            if np.isscalar(y_pred):
                y_pred = np.full(len(df_s), y_pred)
            y_true = df_s["target"].values
            return float(np.mean((y_true - y_pred) ** 2))
        except Exception:
            return float('inf')
        
    candidates = runner.run_top_candidates(slice_context, top_k=15, require_all_vars=True)

    if not candidates:
        print("[Generalized Symmetry] Brute-force не нашел явного выражения. Отмена упрощения.")
        return None

    best_expr = None
    best_val_mse = float('inf')
    X_val = df_slice[group_cols].values
    Y_val = df_slice["target"].values

    for candidate in candidates:
        val_mse = evaluate_expr_mse(candidate, df_slice)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_expr = candidate

    if best_expr is not None:
        print(f"-> Найдена многомерная внутренняя функция связи из Топ-{len(candidates)}: {best_expr} (MSE на срезе: {best_val_mse:.6e})")
        return best_expr
    else:
        print("[Generalized Symmetry] Brute-force не нашел подходящей формулы. Отмена упрощения.")
        return None
    

class PipelineStep(ABC):
    @abstractmethod
    def transform(self, context: 'SymbolicRegressionContext') -> 'SymbolicRegressionContext':
        pass

class DimensionalAnalysisStep(PipelineStep):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def transform(self, context: 'SymbolicRegressionContext') -> 'SymbolicRegressionContext':
        if self.verbose:
            print("\n" + "="*60)
            print("[ЗАПУСК] Теорема Пи Букингема")
            print("="*60)

        active_features = context.get_active_features()
        target_name = context.target_name
        n_features = len(active_features)

        if n_features == 0:
            raise ValueError("Нет доступных признаков для проведения размерного анализа.")

        dim_vectors = [context.registry.get_dim(name).vector for name in active_features]
        target_dim_vector = context.registry.get_dim(target_name).vector

        D = sp.Matrix(np.column_stack(dim_vectors))
        a_Q = sp.Matrix(target_dim_vector)

        augmented_matrix = D.row_join(a_Q)
        rref_matrix, pivots = augmented_matrix.rref()

        if n_features in pivots:
            raise DimensionalError(
                f"Размерность целевой переменной '{target_name}' "
                f"не может быть представлена через размерности текущих признаков {active_features}."
            )

        valid_pivots = [p for p in pivots if p < n_features]
        free_vars = [j for j in range(n_features) if j not in valid_pivots]

        c_particular = sp.Matrix.zeros(n_features, 1)
        for i, p_col in enumerate(valid_pivots):
            c_particular[p_col] = rref_matrix[i, n_features]

        nullspace_vectors = []
        for free_idx in free_vars:
            v = sp.Matrix.zeros(n_features, 1)
            v[free_idx] = 1
            for i, p_col in enumerate(valid_pivots):
                v[p_col] = -rref_matrix[i, free_idx]
            nullspace_vectors.append(v)

        if self.verbose:
            self._print_solutions_summary(active_features, c_particular, nullspace_vectors)

        new_df_data = {}

        def evaluate_vector_values(vector: sp.Matrix) -> np.ndarray:
            result = np.ones(len(context.df))

            for idx, power in enumerate(vector):
                if power != 0:
                    col_vals = context.df[active_features[idx]].values
                    result *= col_vals ** float(power)

            result = np.nan_to_num(result, nan=1.0, posinf=1e5, neginf=-1e5)
            return result
        
        for i, vec in enumerate(nullspace_vectors, start=1):
            new_df_data[f"Pi_{i}"] = evaluate_vector_values(vec)

        anchor_values = evaluate_vector_values(c_particular)
        target_values = context.df[target_name].values
        new_df_data["target"] = target_values / (anchor_values + 1e-19)

        new_df = pd.DataFrame(new_df_data)

        new_registry = PhysicalRegistry()
        zero_dim_vector = [0.0] * len(target_dim_vector)

        for col in new_df.columns:
            new_registry.register(col, zero_dim_vector)

        def vector_to_sympy_expression(vector: sp.Matrix) -> sp.Expr:
            expr = sp.Integer(1)
            for idx, power in enumerate(vector):
                if power != 0:
                    clean_power = sp.nsimplify(power)
                    current_symbol = context.symbolic_mapping[active_features[idx]]
                    expr *= current_symbol ** clean_power
            return expr
        
        new_symbolic_mapping = {}
        for i, vec in enumerate(nullspace_vectors, start=1):
            new_symbolic_mapping[f"Pi_{i}"] = vector_to_sympy_expression(vec)

        anchor_expr = vector_to_sympy_expression(c_particular)
        new_target_expr = context.target_expr / anchor_expr

        new_context = SymbolicRegressionContext(
            df=new_df,
            registry=new_registry,
            target_name="target",
            symbolic_mapping=new_symbolic_mapping,
            target_expr=new_target_expr
        )

        if self.verbose:
            print(f"Размерный анализ завершен. Сформировано {len(nullspace_vectors)} безразмерных комплексов.")
            print("="*60 + "\n")

        return new_context
    
    
    def _print_solutions_summary(
        self, 
        features: list, 
        c_particular: sp.Matrix, 
        nullspace_vectors: list
    ):
        def format_vector(vector):
            parts = []
            for name, power in zip(features, vector):
                if power != 0:
                    clean_power = sp.nsimplify(power)
                    parts.append(f"{name}^{clean_power}" if clean_power != 1 else name)
            return " · ".join(parts) if parts else "1"

        print(f"Размерный якорь: {format_vector(c_particular)}")
        if nullspace_vectors:
            print("Полученные безразмерные Пи-группы:")
            for i, vec in enumerate(nullspace_vectors, start=1):
                print(f"  Pi_{i} = {format_vector(vec)}")
        else:
            print("Безразмерные Пи-группы не обнаружены (система жестко определена).")


class GPSimplificationStep(PipelineStep):
    def __init__(self, 
                gp_config: 'GPConfig',
                max_depth: int = None, 
                local_bf_max_length: int = 6, 
                final_bf_max_length: int = 8, 
                k_sigma_multipliers: list = [1.0, 1.5, 2.0],
                base_k_sigma: float = 3.0,
                loss_degradation_tolerance: float = 0.35,
                verbose: bool = True,
                optimize_constants: bool = True, 
                allowed_constants: list = [1.0, 2.0],            
                allowed_ops: list = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"],
                k_best: int = 50):
        self.gp_config = gp_config
        self.max_depth = max_depth
        self.local_bf_max_length = local_bf_max_length
        self.final_bf_max_length = final_bf_max_length 
        self.loss_degradation_tolerance = loss_degradation_tolerance
        self.verbose = verbose
        self.optimize_constants = optimize_constants
        self.allowed_constants = allowed_constants
        self.allowed_ops = allowed_ops
        self.base_k_sigma = base_k_sigma
        self.k_sigma_multipliers = k_sigma_multipliers
        self.k_best = k_best

    def transform(self, context: 'SymbolicRegressionContext') -> 'SymbolicRegressionContext':
        if self.verbose:
            print("\n" + "="*60)
            print("[GPSimplificationStep][ЗАПУСК] Шаг GP декомпозиции")
            print(f"Активные переменные на входе: {context.get_active_features()}")
            print("="*60)
        
        working_context = context.copy()

        import gpytorch
        with gpytorch.settings.fast_computations(solves=False, log_prob=False), gpytorch.settings.cholesky_jitter(1e-3):
            final_formula = self._recursive_solve(working_context, current_depth=0)
        
        working_context.target_expr = final_formula

        if self.verbose:
            print("\n" + "="*60)
            print("GP декомпозиция завершена.")
            print(f"Полученная структура: {final_formula}")
            print("="*60 + "\n")

        return working_context


    def combine_formulas(self, formula_A, formula_B, split_type: str) -> sp.Expr:
        expr_A = sp.sympify(formula_A)
        expr_B = sp.sympify(formula_B)
        
        sum_expr = expr_A + expr_B
        
        if split_type == "Additive Separability":
            return sum_expr
        elif split_type == "Multiplicative Separability":
            return expr_A * expr_B
        elif split_type == "General Additive (Linear)":
            return sum_expr
        elif split_type == "General Additive (Log)":
            return sp.exp(sum_expr)
        elif split_type == "General Additive (Sqrt)":
            return sum_expr**2
        elif split_type == "General Additive (Square)":
            return sp.sqrt(sum_expr)
        elif split_type == "General Additive (Sin)":
            return sp.sin(sum_expr)
        elif split_type == "General Additive (Tan)":
            return sp.tan(sum_expr)
        else:
            raise ValueError(f"Неизвестный тип склейки: {split_type}")
    
    def train_gp(self, dataset: pd.DataFrame, target_name: str, config: GPConfig) -> GPRegressionPipeline:
        feature_cols = [col for col in dataset.columns if col != target_name]
        train_x = torch.tensor(dataset[feature_cols].values, dtype=torch.float64)
        train_y = torch.tensor(dataset[target_name].values, dtype=torch.float64)
        pipeline = GPRegressionPipeline(config)
        loss_history = pipeline.fit(train_x, train_y)
        return pipeline, loss_history

    def _brute_force_symbolic_search(self, context: 'SymbolicRegressionContext') -> sp.Expr:
        feature_names = context.get_active_features()
        run_length = self.local_bf_max_length

        runner = BruteForceRunner(
            max_length=run_length,
            optimize_constants=self.optimize_constants,
            allowed_constants=self.allowed_constants,
            allowed_ops=self.allowed_ops,
            k_best=self.k_best
        )

        if self.verbose:
            print(f"[GPSimplificationStep][BF Search] Запуск brute-force для {feature_names} (Адаптивная длина: {run_length})...")

        best_expr, best_mse = runner.run(context)
        if best_expr is not None:
            if self.verbose:
                print(f"[BF Search] Найдено явное выражение: {best_expr} (MSE: {best_mse:.6e})")
            return best_expr
        
        if self.verbose:
            print("[BF Search] Корректных формул не найдено. Возврат символьной заглушки.")
        
        if len(feature_names) == 1:
            x_name = feature_names[0]
            phi = sp.Function(f"Phi_{x_name}")
            return phi(sp.Symbol(x_name))
        else:
            phi = sp.Function("Phi_remaining")
            return phi(*[sp.Symbol(name) for name in feature_names])
        
    def _final_brute_force_search(self, context: 'SymbolicRegressionContext') -> sp.Expr:
        feature_names = context.get_active_features()
        run_length = self.final_bf_max_length
        
        runner = BruteForceRunner(
            max_length=run_length,
            optimize_constants=self.optimize_constants,
            allowed_constants=self.allowed_constants,
            allowed_ops=self.allowed_ops
        )

        if self.verbose:
            print(f"[GPSimplificationStep][Финальный BF] Запуск для {feature_names} (Макс. длина: {self.final_bf_max_length})...")

        best_expr, best_mse = runner.run(context)
        if best_expr is not None:
            if self.verbose:
                print(f"[Финальный BF] Найдено итоговое выражение: {best_expr} (MSE: {best_mse:.6e})")
            return best_expr
        
        if self.verbose:
            print("[Финальный BF] Формул не найдено. Возврат символьной заглушки.")
        
        if len(feature_names) == 1:
            x_name = feature_names[0]
            phi = sp.Function(f"Phi_{x_name}")
            return phi(sp.Symbol(x_name))
        else:
            phi = sp.Function("Phi_remaining")
            return phi(*[sp.Symbol(name) for name in feature_names])
        
    def _recursive_solve(self, context: 'SymbolicRegressionContext', current_depth: int) -> sp.Expr:
        feature_names = context.get_active_features()

        if len(feature_names) == 1:
            return self._final_brute_force_search(context)

        if self.max_depth is not None and current_depth >= self.max_depth:
            return self._final_brute_force_search(context)
        
        if self.verbose:
            print(f"\n--- [Глубина {current_depth}] Обучение GP для переменных: {feature_names} ---")

        active_simplifiers = [
            GeneralAdditiveSeparabilitySimplifier(
                            num_test_points=None,
                            bf_max_length=self.local_bf_max_length,
                            optimize_constants=self.optimize_constants, 
                            allowed_constants=self.allowed_constants,
                            allowed_ops=self.allowed_ops
            ),
            AdditiveSeparabilitySimplifier(num_test_points=None), 
            MultiplicationSeparabilitySimplifier(num_test_points=None),
            TranslationalSymmetrySimplifier(num_test_points=None),
            AdditionSymmetrySimplifier(num_test_points=None),
            LargeScaleSymmetrySimplifier(num_test_points=None),
            MultiplySymmetrySimplifier(num_test_points=None),
            CompositionalitySimplifier(
                num_test_points=None,
                bf_max_length=self.local_bf_max_length,
                optimize_constants=self.optimize_constants, 
                allowed_constants=self.allowed_constants,
                allowed_ops=self.allowed_ops,
                k_best=self.k_best
            ),
            # GeneralizedSymmetrySimplifier(num_test_points=None)
        ]
        gp_model, loss_history = self.train_gp(context.df, context.target_name, self.gp_config)
        current_loss = loss_history[-1] if loss_history else 1e5

        active_multipliers = [1.0] if len(feature_names) <= 2 else self.k_sigma_multipliers

        for mult in active_multipliers:
            current_k_sigma = self.base_k_sigma * mult
            rejected_candidates = []

            for simplifier in active_simplifiers:
                success, result = simplifier.try_simplify(gp_model, context, k_sigma=current_k_sigma)
                if not success:
                    continue

                if self.verbose:
                    print(f"[Найдена гипотеза]: {simplifier.name} -> {result}")

                if simplifier.name in ["Additive Separability", "Multiplicative Separability", "General Additive Separability"]:
                    if simplifier.name == "General Additive Separability":
                        groups, g_inv, trans_type = result
                        mapping = {
                            "linear": "General Additive (Linear)", "log": "General Additive (Log)",
                            "pow_2": "General Additive (Sqrt)", "pow_half": "General Additive (Square)",
                            "sin": "General Additive (Sin)", "tan": "General Additive (Tan)"
                        }
                        split_name = mapping.get(trans_type, "General Additive (Linear)")
                        split_res = (groups, g_inv)
                    else:
                        split_name = simplifier.name
                        split_res = result

                    context_A, context_B = self._split_context(context, split_res, gp_model, split_name)
                    
                    _, loss_h_A = self.train_gp(context_A.df, context_A.target_name, self.gp_config)
                    _, loss_h_B = self.train_gp(context_B.df, context_B.target_name, self.gp_config)
                    
                    loss_A = loss_h_A[-1] if loss_h_A else 1e5
                    loss_B = loss_h_B[-1] if loss_h_B else 1e5
                    max_child_loss = max(loss_A, loss_B)
                    delta_loss = max_child_loss - current_loss

                    if delta_loss <= self.loss_degradation_tolerance:
                        if self.verbose:
                            print(f"[Принято {split_name}]: Delta Loss={delta_loss:.4f} <= {self.loss_degradation_tolerance}")
                        formula_A = self._recursive_solve(context_A, current_depth + 1)
                        formula_B = self._recursive_solve(context_B, current_depth + 1)
                        return self.combine_formulas(formula_A, formula_B, split_type=split_name)
                    else:
                        if self.verbose:
                            print(f"[Откат {split_name}]: Деградация лосса +{delta_loss:.4f} > {self.loss_degradation_tolerance}")
                        rejected_candidates.append({
                            'delta_loss': delta_loss,
                            'type': 'split',
                            'split_name': split_name,
                            'context_A': context_A,
                            'context_B': context_B
                        })

                else:
                    mutated_context = self._collapse_context_variables(context, result, simplifier.name, gp_model)
                    if mutated_context is None:
                        continue

                    _, test_loss_h = self.train_gp(mutated_context.df, mutated_context.target_name, self.gp_config)
                    test_loss = test_loss_h[-1] if test_loss_h else 1e5
                    delta_loss = test_loss - current_loss

                    if delta_loss <= self.loss_degradation_tolerance:
                        if self.verbose:
                            print(f"[Принято {simplifier.name}]: Delta Loss={delta_loss:.4f} <= {self.loss_degradation_tolerance}")
                        return self._recursive_solve(mutated_context, current_depth + 1)
                    else:
                        if self.verbose:
                            print(f"[Откат {simplifier.name}]: Деградация лосса +{delta_loss:.4f} > {self.loss_degradation_tolerance}")
                        rejected_candidates.append({
                            'delta_loss': delta_loss,
                            'type': 'collapse',
                            'mutated_context': mutated_context
                        })

            if rejected_candidates:
                best_cand = min(rejected_candidates, key=lambda x: x['delta_loss'])
                if best_cand['delta_loss'] <= 0.8:
                    if self.verbose:
                        print(f"\nПрименяем лучшую из отвергнутых гипотез с минимальной деградацией (Delta Loss: +{best_cand['delta_loss']:.4f})")

                    if best_cand['type'] == 'split':
                        formula_A = self._recursive_solve(best_cand['context_A'], current_depth + 1)
                        formula_B = self._recursive_solve(best_cand['context_B'], current_depth + 1)
                        return self.combine_formulas(formula_A, formula_B, split_type=best_cand['split_name'])
                    else:
                        return self._recursive_solve(best_cand['mutated_context'], current_depth + 1)

        if self.verbose:
            print(f"[Рекурсия] Переход к финальному BF.")
        return self._final_brute_force_search(context)

    def _collapse_context_variables(
        self, 
        context: 'SymbolicRegressionContext', 
        result, 
        split_type: str, 
        gp_model
    ) -> 'SymbolicRegressionContext':
        new_context = context.copy()
        df_mutated = new_context.df

        if split_type in ["Translational Symmetry", "LargeScale Symmetry", "Multiply Symmetry", "Addition Symmetry"]:
            if isinstance(result[0], list):
                group = next(g for g in result if len(g) >= 2)
            else:
                group = result

            x1_name, x2_name = group[0], group[1]
            s1, s2 = sp.Symbol(x1_name), sp.Symbol(x2_name)

            if split_type == "Translational Symmetry":
                new_col_name = f"({x1_name}_minus_{x2_name})"
                h_expr = s1 - s2
                df_mutated[new_col_name] = df_mutated[x1_name] - df_mutated[x2_name]
            
            elif split_type == "Addition Symmetry": 
                new_col_name = f"({x1_name}_plus_{x2_name})"
                h_expr = s1 + s2
                df_mutated[new_col_name] = df_mutated[x1_name] + df_mutated[x2_name]

            elif split_type == "LargeScale Symmetry":
                new_col_name = f"({x1_name}_div_{x2_name})"
                h_expr = s1 / s2
                df_mutated[new_col_name] = df_mutated[x1_name] / (df_mutated[x2_name] + 1e-19)
                
            elif split_type == "Multiply Symmetry":
                new_col_name = f"({x1_name}_mul_{x2_name})"
                h_expr = s1 * s2
                df_mutated[new_col_name] = df_mutated[x1_name] * df_mutated[x2_name]

            df_mutated.drop(columns=[x1_name, x2_name], inplace=True)

            new_dim = DimensionalityEvaluator.evaluate(h_expr, context.registry)
            new_context.register_mutation([x1_name, x2_name], new_col_name, h_expr, new_dim)

        elif split_type == "Compositionality":
            if isinstance(result, tuple):
                h_expr = result[0]
            else:
                h_expr = result

            symbols = sorted(list(h_expr.free_symbols), key=lambda s: s.name)
            involved_vars = [sym.name for sym in symbols]
            f_h = sp.lambdify(symbols, h_expr, 'numpy')
            args = [df_mutated[name].values for name in involved_vars]
            new_col_name = f"({str(h_expr)})"
            df_mutated[new_col_name] = f_h(*args)
            df_mutated.drop(columns=involved_vars, inplace=True)

            new_dim = DimensionalityEvaluator.evaluate(h_expr, context.registry)
            new_context.register_mutation(involved_vars, new_col_name, h_expr, new_dim)

        elif split_type == "Generalized Symmetry":
            group = result 
            if self.verbose:
                print(f"Запуск локального поиска формулы связи для группы {group}...")
            
            h_expr = find_best_h_for_group(
                gp_model, context, group, 
                local_bf_max_length=self.local_bf_max_length, 
                optimize_constants=self.optimize_constants, 
                allowed_constants=self.allowed_constants,
                allowed_ops=self.allowed_ops,
                k_best=self.k_best 
            )
            if h_expr is None:
                return None
            
            return self._collapse_context_variables(context, h_expr, "Compositionality", gp_model)

        return new_context


    def _split_context(
        self, 
        context: 'SymbolicRegressionContext', 
        groups, 
        gp_model, 
        split_type: str
    ) -> Tuple['SymbolicRegressionContext', 'SymbolicRegressionContext']:
        feature_cols = context.get_active_features()

        is_gas = split_type.startswith("General Additive")
        if is_gas:
            groups, g_inv_expr = groups

        group_A = groups[0]
        group_B = []
        for g in groups[1:]:
            group_B.extend(g)
        
        target_name = context.target_name
        y_original = context.df[target_name].values
        n_samples = len(context.df)

        pts_A = torch.tensor(context.df[feature_cols].values, dtype=torch.float64)
        for col_name in group_B:
            col_idx = feature_cols.index(col_name)
            median_val = context.df[col_name].median()
            pts_A[:, col_idx] = torch.full((n_samples,), median_val)

        gp_model.model.eval()
        gp_model.likelihood.eval()

        with torch.no_grad():
            predictions = gp_model.predict(pts_A)
            y_A = predictions.mean.numpy()

        original_target_dim = context.registry.get_dim(target_name)
        zero_dim_vector = [0.0] * len(original_target_dim.vector)

        if is_gas:
            registry_temp = PhysicalRegistry()
            registry_temp.register(target_name, original_target_dim.vector)
            dim_target_A = DimensionalityEvaluator.evaluate(g_inv_expr, registry_temp)
            dim_target_B = dim_target_A
        elif split_type == "Multiplicative Separability":
            dim_target_A = original_target_dim
            dim_target_B = PhysicalDimension(zero_dim_vector)
        elif split_type == "Additive Separability":
            dim_target_A = original_target_dim
            dim_target_B = original_target_dim

        shift_const = np.median(y_A) if split_type == "Additive Separability" else 0.0

        df_A = context.df[group_A].copy()
        if is_gas:
            target_sym = sp.Symbol(target_name)
            f_g_inv = sp.lambdify([target_sym], g_inv_expr, 'numpy')
            df_A[target_name] = f_g_inv(y_A)
        elif split_type == "Additive Separability":
            df_A[target_name] = y_A - shift_const
        else:
            df_A[target_name] = y_A

        registry_A = PhysicalRegistry()
        registry_A.register(target_name, dim_target_A.vector)
        for col in group_A:
            registry_A.register(col, context.registry.get_dim(col).vector)

        mapping_A = {col: context.symbolic_mapping[col] for col in group_A}
        context_A = SymbolicRegressionContext(df_A, registry_A, target_name, mapping_A, context.target_expr)

        df_B = context.df[group_B].copy()
        if is_gas:
            target_sym = sp.Symbol(target_name)
            f_g_inv = sp.lambdify([target_sym], g_inv_expr, 'numpy')
            df_B[target_name] = f_g_inv(y_original) - f_g_inv(y_A)
        elif split_type == "Additive Separability":
            df_B[target_name] = (y_original - y_A) + shift_const
        elif split_type == "Multiplicative Separability":
            y_A_safe = np.copysign(np.maximum(np.abs(y_A), 1e-4), y_A)
            df_B[target_name] = np.clip(y_original / y_A_safe, -1e5, 1e5)

        registry_B = PhysicalRegistry()
        registry_B.register(target_name, dim_target_B.vector)
        for col in group_B:
            registry_B.register(col, context.registry.get_dim(col).vector)

        mapping_B = {col: context.symbolic_mapping[col] for col in group_B}
        context_B = SymbolicRegressionContext(df_B, registry_B, target_name, mapping_B, context.target_expr)

        return context_A, context_B
        
class SymmetryPreprocessingStep(PipelineStep):
    def __init__(self, gp_config: 'GPConfig', verbose: bool = True,
                 optimize_constants: bool = True, allowed_constants: list = [1.0, 2.0],
                 allowed_ops: list = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"],
                 active_simplifiers: list = ["translational", "addition", "largescale", "multiply", "generalized"]):
        self.gp_config = gp_config
        self.verbose = verbose
        self.optimize_constants = optimize_constants
        self.allowed_constants = allowed_constants
        self.allowed_ops = allowed_ops
        self.active_simplifiers = active_simplifiers

    def train_gp(self, dataset: pd.DataFrame, target_name: str, config: GPConfig) -> GPRegressionPipeline:
        feature_cols = [col for col in dataset.columns if col != target_name]
        train_x = torch.tensor(dataset[feature_cols].values, dtype=torch.float64)
        train_y = torch.tensor(dataset[target_name].values, dtype=torch.float64)
        pipeline = GPRegressionPipeline(config)
        loss_history = pipeline.fit(train_x, train_y)
        return pipeline, loss_history

    def transform(self, context: 'SymbolicRegressionContext') -> 'SymbolicRegressionContext':
        if self.verbose:
            print("\n" + "="*60)
            print("[SymmetryPreprocessingStep][ЗАПУСК] Перманентное сжатие признаков")
            print(f"Переменные на входе: {context.get_active_features()}")
            print("="*60)

        working_context = context.copy()

        simplifier_map = {
            "translational": lambda: TranslationalSymmetrySimplifier(),
            "addition": lambda: AdditionSymmetrySimplifier(),
            "largescale": lambda: LargeScaleSymmetrySimplifier(),
            "multiply": lambda: MultiplySymmetrySimplifier(),
            "generalized": lambda: GeneralizedSymmetrySimplifier(),
            "compositionality": lambda: CompositionalitySimplifier(
                optimize_constants=self.optimize_constants,
                allowed_constants=self.allowed_constants,
                allowed_ops=self.allowed_ops
            )
        }

        active_simplifiers = []
        for key in self.active_simplifiers:
            if key in simplifier_map:
                active_simplifiers.append(simplifier_map[key]())
            else:
                print(f"[Warning] Unknown simplifier key for preprocessing: '{key}'")

        while len(working_context.get_active_features()) > 1:
            gp_model, loss_history = self.train_gp(working_context.df, working_context.target_name, self.gp_config)
            current_loss = loss_history[-1] if loss_history else 1e5
            current_noise = gp_model.likelihood.noise.item()

            found_any = False

            for simplifier in active_simplifiers:
                success, result = simplifier.try_simplify(gp_model, working_context)
                if success:
                    if self.verbose:
                        print(f"[Предобработка] Сжатие по симметрии: {simplifier.name} -> {result}")
                    
                    collapsed = self._collapse_context_variables(
                        working_context, result, simplifier.name, gp_model
                    )
                    if collapsed is not None:
                        test_gp, test_loss_history = self.train_gp(
                            collapsed.df, collapsed.target_name, self.gp_config
                        )
                        test_loss = test_loss_history[-1] if test_loss_history else 1e5
                        test_noise = test_gp.likelihood.noise.item()
                        if test_loss > current_loss + 0.35:
                            if self.verbose:
                                print(f"[Предобработка] ОТКАТ: Сжатие по {simplifier.name} отклонено. "
                                    f"Модель деградировала (Loss: {current_loss:.4f} -> {test_loss:.4f}, "
                                    f"Noise: {current_noise:.4f} -> {test_noise:.4f})")
                            continue
                        working_context = collapsed
                        found_any = True
                        break  
                    else:
                        if self.verbose:
                            print(f"[Предобработка] Свертка по {simplifier.name} вернула None. Переход к следующему упростителю.")

            if not found_any:
                break  

        if self.verbose:
            print(f"Сжатие завершено. Переменные на выходе: {working_context.get_active_features()}")
            print("="*60 + "\n")

        return working_context

    def _collapse_context_variables(self, context: 'SymbolicRegressionContext', result, 
                                    split_type: str, gp_model) -> 'SymbolicRegressionContext':
        new_context = context.copy()
        df_mutated = new_context.df

        if split_type in ["Translational Symmetry", "LargeScale Symmetry", "Multiply Symmetry", "Addition Symmetry"]:
            if isinstance(result[0], list):
                group = next(g for g in result if len(g) >= 2)
            else:
                group = result

            x1_name, x2_name = group[0], group[1]
            s1, s2 = sp.Symbol(x1_name), sp.Symbol(x2_name)

            if split_type == "Translational Symmetry":
                new_col_name = f"({x1_name}_minus_{x2_name})"
                h_expr = s1 - s2
                df_mutated[new_col_name] = df_mutated[x1_name] - df_mutated[x2_name]
            elif split_type == "Addition Symmetry": 
                new_col_name = f"({x1_name}_plus_{x2_name})"
                h_expr = s1 + s2
                df_mutated[new_col_name] = df_mutated[x1_name] + df_mutated[x2_name]
            elif split_type == "LargeScale Symmetry":
                new_col_name = f"({x1_name}_div_{x2_name})"
                h_expr = s1 / s2
                df_mutated[new_col_name] = df_mutated[x1_name] / (df_mutated[x2_name] + 1e-19)
            elif split_type == "Multiply Symmetry":
                new_col_name = f"({x1_name}_mul_{x2_name})"
                h_expr = s1 * s2
                df_mutated[new_col_name] = df_mutated[x1_name] * df_mutated[x2_name]

            df_mutated.drop(columns=[x1_name, x2_name], inplace=True)

            new_dim = DimensionalityEvaluator.evaluate(h_expr, context.registry)
            new_context.register_mutation([x1_name, x2_name], new_col_name, h_expr, new_dim)

        elif split_type == "Compositionality":
            if isinstance(result, tuple):
                h_expr = result[0]
            else:
                h_expr = result

            symbols = sorted(list(h_expr.free_symbols), key=lambda s: s.name)
            involved_vars = [sym.name for sym in symbols]
            f_h = sp.lambdify(symbols, h_expr, 'numpy')
            args = [df_mutated[name].values for name in involved_vars]
            new_col_name = f"({str(h_expr)})"
            df_mutated[new_col_name] = f_h(*args)
            df_mutated.drop(columns=involved_vars, inplace=True)

            new_dim = DimensionalityEvaluator.evaluate(h_expr, context.registry)
            new_context.register_mutation(involved_vars, new_col_name, h_expr, new_dim)

        elif split_type == "Generalized Symmetry":
            group = result 
            if self.verbose:
                print(f"Запуск локального поиска формулы связи для группы {group}...")
            
            h_expr = find_best_h_for_group(
                gp_model, context, group, 
                local_bf_max_length=self.local_bf_max_length, 
                optimize_constants=self.optimize_constants, 
                allowed_constants=self.allowed_constants,
                allowed_ops=self.allowed_ops,
                k_best=self.k_best 
            )
            if h_expr is None:
                return None
            
            return self._collapse_context_variables(context, h_expr, "Compositionality", gp_model)

        return new_context
        
        
class GPHyperparameterTuningStep(PipelineStep):
    def __init__(self, gp_config: GPConfig, cache_path: str = "gp_best_params.json",
                 force_tune: bool = False, n_trials: int = 40, subsample_size: Optional[int] = None,
                 gamma: float = 0.01, verbose: bool = True,
                 fixed_optimizer: Optional[Any] = None,
                 fixed_kernel_type: Optional[Any] = None,
                 fixed_mean_type: Optional[Any] = None,
                 fixed_loss_type: Optional[Any] = None,
                 fixed_lr: Optional[float] = None,
                 loss_modifier: Optional[Callable] = None):
        self.gp_config = gp_config
        self.cache_path = cache_path
        self.force_tune = force_tune
        self.n_trials = n_trials
        self.subsample_size = subsample_size
        self.gamma = gamma
        self.verbose = verbose

        self.fixed_optimizer = fixed_optimizer
        self.fixed_kernel_type = fixed_kernel_type
        self.fixed_mean_type = fixed_mean_type
        self.fixed_loss_type = fixed_loss_type
        self.fixed_lr = fixed_lr
        self.loss_modifier = loss_modifier

    def transform(self, context: 'SymbolicRegressionContext') -> 'SymbolicRegressionContext':
        if not self.force_tune and os.path.exists(self.cache_path):
            if self.verbose:
                print(f"[Tuning] Найден кэш параметров в '{self.cache_path}'. Загрузка...")
            self._load_and_apply_params()
            return context
        
        if self.verbose:
            print("\n" + "="*60)
            print("[Tuning][ЗАПУСК] Поиск оптимальных гиперпараметров GP (Optuna)")
            print(f"Зафиксированные параметры (без подбора):")
            if self.fixed_optimizer is not None: print(f"  - optimizer: {self.fixed_optimizer}")
            if self.fixed_kernel_type is not None: print(f"  - kernel_type: {self.fixed_kernel_type}")
            if self.fixed_mean_type is not None: print(f"  - mean_type: {self.fixed_mean_type}")
            if self.fixed_loss_type is not None: print(f"  - loss_type: {self.fixed_loss_type}")
            if self.fixed_lr is not None: print(f"  - lr: {self.fixed_lr}")
            if self.loss_modifier is not None: print(f"  - loss_modifier: {self.loss_modifier.__name__ if hasattr(self.loss_modifier, '__name__') else 'Custom Callable'}")
            print("="*60)

        feature_cols = context.get_active_features()
        if self.subsample_size is not None and len(context.df) > self.subsample_size:
            df_sub = context.df.sample(n=self.subsample_size, random_state=42)
        else:
            df_sub = context.df

        X_np = df_sub[feature_cols].values
        Y_np = df_sub[context.target_name].values

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler, 
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=45)
        )
        study.optimize(lambda trial: self._objective(trial, X_np, Y_np), n_trials=self.n_trials)

        best_params = study.best_params.copy()
        
        if self.fixed_optimizer is not None: best_params["optimizer"] = self.fixed_optimizer
        if self.fixed_kernel_type is not None: best_params["kernel_type"] = self.fixed_kernel_type
        if self.fixed_mean_type is not None: best_params["mean_type"] = self.fixed_mean_type
        if self.fixed_loss_type is not None: best_params["loss_type"] = self.fixed_loss_type
        if self.fixed_lr is not None: best_params["lr"] = self.fixed_lr

        if self.verbose:
            print("\n" + "="*50)
            print("[Tuning] Оптимизация завершена!")
            print(f"Лучший скор (CV 1-R2 + Penalty): {study.best_value:.5f}")
            print("Итоговые подобранные параметры:")
            for k, v in best_params.items():
                print(f"  {k}: {v}")
            print("="*50 + "\n")

        json_save_dict = {}
        for k, v in best_params.items():
            if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                json_save_dict[k] = v
            else:
                json_save_dict[k] = str(v)

        with open(self.cache_path, "w") as f:
            json.dump(json_save_dict, f, indent=4)

        self._apply_params(best_params)
        return context

    def _objective(self, trial, X_np, Y_np) -> float:
        optimizer_name = self.fixed_optimizer if self.fixed_optimizer is not None \
            else trial.suggest_categorical("optimizer", ["adam", "lbfgs"])
            
        kernel_type = self.fixed_kernel_type if self.fixed_kernel_type is not None \
            else trial.suggest_categorical("kernel_type", [
                "rbf", "matern_32", "matern_52", "rq", "periodic", "cosine", "spectral_mixture", "cauchy"
            ])
            
        mean_type = self.fixed_mean_type if self.fixed_mean_type is not None \
            else trial.suggest_categorical("mean_type", ["constant", "zero"])
            
        loss_type = self.fixed_loss_type if self.fixed_loss_type is not None \
            else trial.suggest_categorical("loss_type", ["mll", "loo"])

        if self.fixed_lr is not None:
            lr = self.fixed_lr
        elif optimizer_name == "adam":
            lr = trial.suggest_float("adam_lr", 1e-3, 0.1, log=True)
        else:
            lr = 1.0

        trial_config = GPConfig(
            model=ModelConfig(
                mean_type=mean_type,
                kernel=KernelConfig(type=kernel_type, scale_kernel=True, ard=True)
            ),
            training=TrainingConfig(
                lr=lr,
                epochs=500, 
                early_stopping_patience=100,
                optimizer=optimizer_name,
                loss_type=loss_type,
                verbose=False,
                loss_modifier=self.loss_modifier 
            )
        )

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_np)):
            X_tr, X_val = torch.tensor(X_np[train_idx], dtype=torch.float64), torch.tensor(X_np[val_idx], dtype=torch.float64)
            Y_tr, Y_val = torch.tensor(Y_np[train_idx], dtype=torch.float64), torch.tensor(Y_np[val_idx], dtype=torch.float64)

            pipeline = GPRegressionPipeline(trial_config)
            try:
                current_trial = trial if fold_idx == 0 else None
                pipeline.fit(X_tr, Y_tr, trial=current_trial)
                
                pipeline.model.eval()
                pipeline.likelihood.eval()
                with torch.no_grad():
                    pred_dist = pipeline.predict(X_val)
                    pred_mean = pred_dist.mean.numpy()
            except optuna.TrialPruned:
                raise
            except Exception:
                return 1e5
            
            with torch.no_grad():
                y_val_np = Y_val.numpy()
                ss_res = np.sum((y_val_np - pred_mean) ** 2)
                ss_tot = np.sum((y_val_np - np.mean(y_val_np)) ** 2)
                r2 = 1.0 - (ss_res / (ss_tot + 1e-9))
                r2 = max(r2, -2.0)

            x_grid = X_tr.clone().detach().requires_grad_(True)

            try:
                mean_grid = pipeline.model(x_grid).mean

                grad_outputs = torch.ones_like(mean_grid)
                grads = torch.autograd.grad(
                    outputs=mean_grid,
                    inputs=x_grid,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True
                )[0]

                hessian_sum = 0.0
                for d in range(x_grid.shape[1]):
                    grad_d = grads[:, d]
                    grad_grad_d = torch.autograd.grad(
                        outputs=grad_d,
                        inputs=x_grid,
                        grad_outputs=torch.ones_like(grad_d),
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    if grad_grad_d is not None:
                        hessian_sum += torch.mean(grad_grad_d[:, d] ** 2)
                penalty = hessian_sum.item()
            except Exception:
                penalty = 1e3

            score = (1.0 - r2) + self.gamma * penalty
            
            if np.isnan(score) or np.isinf(score):
                return 1e5
                
            scores.append(score)

        return float(np.mean(scores))  
     
    def _apply_params(self, params: dict):
        optimizer_name = params["optimizer"]
        self.gp_config.training.optimizer = optimizer_name
        
        if optimizer_name == "adam":
            self.gp_config.training.lr = params.get("adam_lr", params.get("lr", 0.05))
            self.gp_config.training.early_stopping_patience = params.get(
                "adam_patience", params.get("early_stopping_patience", 200)
            )
        else:
            self.gp_config.training.lr = 1.0
            self.gp_config.training.early_stopping_patience = params.get(
                "lbfgs_patience", params.get("early_stopping_patience", 200)
            )
            
        self.gp_config.model.kernel.type = params["kernel_type"]
        self.gp_config.model.mean_type = params["mean_type"] 
        self.gp_config.training.loss_type = params["loss_type"]

    def _load_and_apply_params(self):
        with open(self.cache_path, "r") as f:
            params = json.load(f)
        self._apply_params(params)

class BaselineGPStep(PipelineStep):
    def __init__(self, gp_config: GPConfig, verbose: bool = True):
        self.gp_config = gp_config
        self.verbose = verbose
        self.baseline_metrics: Optional[dict] = None

    def transform(self, context: 'SymbolicRegressionContext') -> 'SymbolicRegressionContext':
        if self.verbose:
            print("\n" + "="*60)
            print("[BaselineGPStep][ЗАПУСК] Холостой запуск GP (Оценка базовой точности)")
            print("="*60)

        feature_cols = context.get_active_features()
        target_name = context.target_name

        train_x = torch.tensor(context.df[feature_cols].values, dtype=torch.float64)
        train_y = torch.tensor(context.df[target_name].values, dtype=torch.float64)

        pipeline = GPRegressionPipeline(self.gp_config)
        pipeline.fit(train_x, train_y)

        pipeline.model.eval()
        pipeline.likelihood.eval()

        with torch.no_grad():
            pred_dist = pipeline.predict(train_x)
            y_pred = pred_dist.mean.numpy()

        y_true = train_y.numpy()

        denom = np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)
        relative_errors = (np.abs(y_true - y_pred) / np.abs(denom)) * 100.0

        spearman_res = spearmanr(y_true, y_pred)
        spearman_val = float(spearman_res.statistic if hasattr(spearman_res, 'statistic') else spearman_res[0])
        if np.isnan(spearman_val):
            spearman_val = 0.0

        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mre = float(np.mean(relative_errors))
        mdre = float(np.median(relative_errors))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))

        self.baseline_metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "spearman": spearman_val,
            "mre": mre,
            "mdre": mdre
        }

        if self.verbose:
            print(f"[Базовая GP модель ({target_name})]:")
            print(f"   MSE       : {mse:.6e}")
            print(f"   RMSE      : {rmse:.6e}")
            print(f"   MAE       : {mae:.6e}")
            print(f"   R²        : {r2:.6f}")
            print(f"   Spearman  : {spearman_val:.6f}")
            print(f"   Средняя относительная ошибка (MRE) : {mre:.4f}%")
            print(f"   Медианная отн. ошибка (Median RE)  : {mdre:.4f}%")
            print("="*60 + "\n")

        return context


def run_baseline_gp(df: pd.DataFrame, target_name: str, gp_config: GPConfig, verbose: bool = True) -> dict:
    feature_cols = [col for col in df.columns if col != target_name]
    train_x = torch.tensor(df[feature_cols].values, dtype=torch.float64)
    train_y = torch.tensor(df[target_name].values, dtype=torch.float64)

    pipeline = GPRegressionPipeline(gp_config)
    pipeline.fit(train_x, train_y)

    pipeline.model.eval()
    pipeline.likelihood.eval()

    with torch.no_grad():
        pred_dist = pipeline.predict(train_x)
        y_pred = pred_dist.mean.numpy()

    y_true = train_y.numpy()
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)
    relative_errors = (np.abs(y_true - y_pred) / np.abs(denom)) * 100.0

    spearman_res = spearmanr(y_true, y_pred)
    spearman_val = float(spearman_res.statistic if hasattr(spearman_res, 'statistic') else spearman_res[0])
    if np.isnan(spearman_val):
        spearman_val = 0.0

    metrics = {
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "r2": float(1.0 - (np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12))),
        "spearman": spearman_val,
        "mre": float(np.mean(relative_errors)),
        "mdre": float(np.median(relative_errors))
    }

    if verbose:
        print("\n" + "="*60)
        print(f"[Холостой запуск GP] Оценка базовой точности для '{target_name}':")
        print(f"   MSE       : {metrics['mse']:.6e}")
        print(f"   RMSE      : {metrics['rmse']:.6e}")
        print(f"   MAE       : {metrics['mae']:.6e}")
        print(f"   R²        : {metrics['r2']:.6f}")
        print(f"   Spearman  : {metrics['spearman']:.6f}")
        print(f"   MRE (%)   : {metrics['mre']:.4f}%")
        print(f"   MdRE (%)  : {metrics['mdre']:.4f}%")
        print("="*60 + "\n")

    return metrics