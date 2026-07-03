from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np
import pandas as pd
import sympy as sp
import torch

from .sm_context import SymbolicRegressionContext, DimensionalityEvaluator
from get_pi_complex import DimensionalError, PhysicalRegistry
from GP.config import GPConfig, ModelConfig, KernelConfig, TrainingConfig
from GP.pipeline import GPRegressionPipeline
from function_analysis.simplify import *

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

def find_best_h_for_group(gp_model, dataset: pd.DataFrame, group_cols: list, num_test_points=100) -> sp.Expr:
    all_features = [col for col in dataset.columns if col != 'target']
    n_features = len(all_features)
    n_samples = len(dataset)
    
    n_points = min(num_test_points, n_samples)
    sampled_indices = np.random.choice(n_samples, n_points, replace=False)
    sampled_data = dataset.iloc[sampled_indices]
    
    # Формируем точки, заменяя неактивные переменные медианами (отличный шаг!)
    pts = torch.tensor(dataset[all_features].values, dtype=torch.float32)[sampled_indices]
    remaining_cols = [col for col in all_features if col not in group_cols]
    for col_name in remaining_cols:
        col_idx = all_features.index(col_name)
        median_val = dataset[col_name].median()
        pts[:, col_idx] = median_val 
        
    pts.requires_grad_(True)
    
    gp_model.model.eval()
    gp_model.likelihood.eval()
    y = gp_model.likelihood(gp_model.model(pts)).mean
    grads_all = torch.autograd.grad(y.sum(), pts)[0] 
    
    symbols = [sp.Symbol(name) for name in group_cols]
    
    best_error = 1.0
    best_candidate = None
    
    for idx1, idx2 in combinations(range(len(group_cols)), 2):
        col_name1, col_name2 = group_cols[idx1], group_cols[idx2]
        s1, s2 = symbols[idx1], symbols[idx2]
        
        global_idx1 = all_features.index(col_name1)
        global_idx2 = all_features.index(col_name2)
        
        grads_target_pair = grads_all[:, [global_idx1, global_idx2]]
        norms_target_pair = torch.norm(grads_target_pair, dim=1, keepdim=True) + 1e-9
        v_target_pair = grads_target_pair / norms_target_pair
        
        candidates = [
            s1 + s2,
            s1 - s2, 
            s1 * s2, 
            s1 / s2,

            s1**2 + s2**2, 
            s1**2 - s2**2,
            s1 / (s2**2 + 1e-9), 
            s2 / (s1**2 + 1e-9),

            sp.sin(s1 - s2), 
            sp.sin(s1 + s2), 
            sp.cos(s1 - s2), 
            sp.cos(s1 + s2),

            s1 * sp.exp(s2), 
            s2 * sp.exp(s1), 
            s1 * sp.exp(-s2), 
            s2 * sp.exp(-s1),

            sp.sin(s1) * sp.exp(s2), 
            sp.sin(s2) * sp.exp(s1), 
            sp.sin(s1 - s2) * sp.exp(s1),

            sp.log(s1 + 1e-9) - sp.log(s2 + 1e-9), 
            sp.log(s1 + 1e-9) + sp.log(s2 + 1e-9),
        ]
        
        group_data_np = sampled_data[[col_name1, col_name2]].values
        args = [group_data_np[:, 0], group_data_np[:, 1]]
        
        for h_expr in candidates:
            grad_h_sym = [sp.diff(h_expr, s1), sp.diff(h_expr, s2)]
            
            grad_arrays = []
            for g_expr in grad_h_sym:
                f_g = sp.lambdify([s1, s2], g_expr, 'numpy')
                vals = f_g(*args)
                if isinstance(vals, (int, float, np.integer, np.floating)):
                    vals = np.full(n_points, vals)
                grad_arrays.append(vals)
                
            grads_cand = np.stack(grad_arrays, axis=1)
            grads_cand_tensor = torch.tensor(grads_cand, dtype=torch.float32)
            
            norms_cand = torch.norm(grads_cand_tensor, dim=1, keepdim=True) + 1e-9
            v_cand = grads_cand_tensor / norms_cand
            
            cos_sim = torch.sum(v_target_pair * v_cand, dim=1)
            error = torch.mean(1.0 - cos_sim**2).item()
            
            if error < best_error:
                best_error = error
                best_candidate = h_expr
                
    print(f"-> Лучшая найденная внутренняя функция связи: {best_candidate} (Ошибка косинуса: {best_error:.6f})")
    return best_candidate

class PipelineStep(ABC):
    @abstractmethod
    def transform(self, context: 'SymbolicRegressionContext') -> 'SymbolicRegressionContext':
        """
        Выполняет преобразование контекста и возвращает новый измененный контекст.
        
        Аргументы:
            context: Текущий контекст состояния SymbolicRegressionContext.
            
        Возвращает:
            SymbolicRegressionContext: Новый (или глубоко модифицированный) контекст.
        """
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
                    if power != 0:
                        col_vals = context.df[active_features[idx]].values
                        result *= col_vals ** float(power)

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
    def __init__(self, gp_config: 'GPConfig', max_depth: int = None, verbose: bool = True):
        self.gp_config = gp_config
        self.max_depth = max_depth
        self.verbose = verbose

    def transform(self, context: 'SymbolicRegressionContext') -> 'SymbolicRegressionContext':
        if self.verbose:
            print("\n" + "="*60)
            print("ЗАПУСК: Шаг GP декомпозиции")
            print(f"Активные переменные на входе: {context.get_active_features()}")
            print("="*60)
        
        working_context = context.copy()
        final_formula = self._recursive_solve(working_context, current_depth=0)

        resolved_formula = final_formula.subs(context.symbolic_mapping)

        new_context = context.copy()
        new_context.target_expr = resolved_formula
        
        if self.verbose:
            print("\n" + "="*60)
            print("GP декомпозиция завершена.")
            print(f"Полученная структура: {resolved_formula}")
            print("="*60 + "\n")

        return new_context


    def combine_formulas(self, formula_A, formula_B, split_type: str) -> sp.Expr:
        expr_A = sp.sympify(formula_A)
        expr_B = sp.sympify(formula_B)
        
        if split_type == "Additive Separability" or split_type == "General Additive Separability":
            # f = g(A) + h(B)
            return expr_A + expr_B
            
        elif split_type == "Multiplicative Separability":
            # f = g(A) * h(B)
            return expr_A * expr_B
            
        else:
            raise ValueError(f"Неизвестный тип склейки: {split_type}")
    
    def train_gp(self, dataset: pd.DataFrame, config: GPConfig) -> GPRegressionPipeline:
        """
        Обучает ваш собственный GPRegressionPipeline на текущем датасете.
        """
        feature_cols = [col for col in dataset.columns if col != 'target']
        n_features = len(feature_cols)
        
        train_x = torch.tensor(dataset[feature_cols].values, dtype=torch.float32)
        train_y = torch.tensor(dataset['target'].values, dtype=torch.float32)
        
        pipeline = GPRegressionPipeline(config)
        pipeline.fit(train_x, train_y)
    
        return pipeline

    def _brute_force_symbolic_search(self, context: 'SymbolicRegressionContext') -> sp.Expr:
        feature_names = context.get_active_features()
        if len(feature_names) == 1:
            x_name = feature_names[0]
            phi = sp.Function(f"Phi_{x_name}")
            return phi(sp.Symbol(x_name))
        else:
            phi = sp.Function("Phi_multivariate")
            return phi(*[sp.Symbol(name) for name in feature_names])
        
    def _recursive_solve(self, context: 'SymbolicRegressionContext', current_depth: int) -> sp.Expr:
        feature_names = context.get_active_features()

        if len(feature_names) == 1:
            return self._brute_force_symbolic_search(context)

        if self.max_depth is not None and current_depth >= self.max_depth:
            if self.verbose:
                print(f"Достигнут лимит глубины рекурсии ({self.max_depth}). Остановка.")
            phi = sp.Function("Phi_remaining")
            return phi(*[sp.Symbol(name) for name in feature_names])
        
        if self.verbose:
            print(f"\n--- [Глубина {current_depth}] Обучение GP для переменных: {feature_names} ---")

        gp_model = self.train_gp(context.df, self.gp_config)
        for simplifier in simplifiers:
            success, result = simplifier.try_simplify(gp_model, context.df)
            if success:
                if self.verbose:
                    print(f"[Сработало упрощение]: {simplifier.name} -> {result}")
                if simplifier.name in ["Additive Separability", "Multiplicative Separability", "General Additive Separability"]:
                    context_A, context_B = self._split_context(context, result, gp_model, simplifier.name)

                    formula_A = self._recursive_solve(context_A, current_depth + 1)
                    formula_B = self._recursive_solve(context_B, current_depth + 1)
                    
                    return self.combine_formulas(formula_A, formula_B, split_type=simplifier.name)
                else:
                    mutated_context = self._collapse_context_variables(context, result, simplifier.name, gp_model)
                    return self._recursive_solve(mutated_context, current_depth + 1)

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
            h_expr = find_best_h_for_group(gp_model, context.df, group) 
            return self._collapse_context_variables(context, h_expr, "Compositionality", gp_model)

        return new_context


    def _split_context(
        self, 
        context: 'SymbolicRegressionContext', 
        groups, 
        gp_model, 
        split_type: str
    ) -> Tuple['SymbolicRegressionContext', 'SymbolicRegressionContext']:
        """Разбивает контекст на два независимых подконтекста для параллельного решения."""
        feature_cols = context.get_active_features()
        group_A = groups[0]
        group_B = []
        for g in groups[1:]:
            group_B.extend(g)

        y_original = context.df['target'].values
        n_samples = len(context.df)

        pts_A = torch.tensor(context.df[feature_cols].values, dtype=torch.float32)
        for col_name in group_B:
            col_idx = feature_cols.index(col_name)
            median_val = context.df[col_name].median()
            pts_A[:, col_idx] = torch.full((n_samples,), median_val)

        gp_model.model.eval()
        gp_model.likelihood.eval()

        with torch.no_grad():
            predictions = gp_model.predict(pts_A)
            y_A = predictions.mean.numpy()

        df_A = context.df[group_A].copy()
        df_A['target'] = y_A
        registry_A = PhysicalRegistry()
        registry_A.register("target", context.registry.get_dim(context.target_name).vector)

        for col in group_A:
            registry_A.register(col, context.registry.get_dim(col).vector)

        mapping_A = {col: context.symbolic_mapping[col] for col in group_A}
        context_A = SymbolicRegressionContext(df_A, registry_A, "target", mapping_A, context.target_expr)

        df_B = context.df[group_B].copy()
        if split_type in ["Additive Separability", "General Additive Separability"]:
            df_B['target'] = y_original - y_A
        elif split_type == "Multiplicative Separability":
            y_A_safe = np.copysign(np.maximum(np.abs(y_A), 1e-4), y_A)
    
            df_B['target'] = np.clip(y_original / y_A_safe, -1e5, 1e5)

        registry_B = PhysicalRegistry()
        registry_B.register("target", context.registry.get_dim(context.target_name).vector)
        for col in group_B:
            registry_B.register(col, context.registry.get_dim(col).vector)

        mapping_B = {col: context.symbolic_mapping[col] for col in group_B}
        context_B = SymbolicRegressionContext(df_B, registry_B, "target", mapping_B, context.target_expr)

        return context_A, context_B
        

        
        

