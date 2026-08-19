import copy
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict
import torch
import sympy as sp
import numpy as np
import pandas as pd
import gpytorch
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr
from .sm_context import SymbolicRegressionContext, DimensionalityEvaluator
from get_pi_complex import DimensionalError, PhysicalRegistry, PhysicalDimension, DimensionalProjector
from GP.utils import train_gp_on_dataframe, train_gp_model
from function_analysis._simplify import (
    AdditiveSeparabilitySimplifier,
    MultiplicationSeparabilitySimplifier,
    CompositionalitySimplifier,
    GeneralAdditiveSeparabilitySimplifier
)

from function_analysis.py_brute_force import BruteForceRunner
from pi_optimizer import STEAnchorOptimizer, SequentialPiOptimizer, compute_dcor
from config import PipelineConfig
from logger import setup_logger

logger = setup_logger("Pipeline")

class PipelineStep(ABC):
    @abstractmethod
    def transform(self, context: 'SymbolicRegressionContext') -> 'SymbolicRegressionContext':
        pass

class BaseDimensionalAnalysisStep(PipelineStep, ABC):
    def __init__(self, config: PipelineConfig):
        self.pipeline_config = config
        self.verbose = config.verbose

    def _extract_dimensional_matrices(self, context: 'SymbolicRegressionContext'):
        active_features = context.get_active_features()
        target_name = context.target_name
        n_features = len(active_features)
        if n_features == 0: raise ValueError("Нет признаков для размерного анализа.")

        dim_vectors = [context.registry.get_dim(name).vector for name in active_features]
        target_dim_vector = context.registry.get_dim(target_name).vector
        D = sp.Matrix(np.column_stack(dim_vectors))
        a_Q = sp.Matrix(target_dim_vector)

        augmented_matrix = D.row_join(a_Q)
        rref_matrix, pivots = augmented_matrix.rref()

        if n_features in pivots:
            raise DimensionalError(f"Размерность '{target_name}' невыразима через {active_features}.")

        valid_pivots = [p for p in pivots if p < n_features]
        c_particular = sp.Matrix.zeros(n_features, 1)
        for i, p_col in enumerate(valid_pivots):
            c_particular[p_col] = rref_matrix[i, n_features]

        nullspace_sym = D.nullspace()
        V_canonical = np.column_stack([np.array(v, dtype=float).flatten() for v in nullspace_sym]) if nullspace_sym else np.empty((n_features, 0))

        return active_features, target_name, D, target_dim_vector, c_particular, V_canonical

    def _vector_to_sympy_expression(self, context: 'SymbolicRegressionContext', active_features: List[str], vector) -> sp.Expr:
        expr = sp.Integer(1)
        for idx, power in enumerate(vector):
            if power != 0:
                clean_power = sp.nsimplify(power)
                expr *= context.symbolic_mapping[active_features[idx]] ** clean_power
        return expr

    def _evaluate_vector_values(self, context: 'SymbolicRegressionContext', active_features: List[str], vector) -> np.ndarray:
        result = np.ones(len(context.df))
        for idx, power in enumerate(vector):
            if power != 0:
                result *= context.df[active_features[idx]].values ** float(power)
        return np.nan_to_num(result, nan=1.0, posinf=1e5, neginf=-1e5)

    def _create_dimensionless_context(self, context: 'SymbolicRegressionContext', active_features: List[str], anchor_vector: sp.Matrix, pi_vectors: List[sp.Matrix], target_dimless_values: np.ndarray, target_dim_vector: List[float]) -> 'SymbolicRegressionContext':
        new_df_data = {}
        for i, vec in enumerate(pi_vectors, start=1):
            new_df_data[f"Pi_{i}"] = self._evaluate_vector_values(context, active_features, vec)
        new_df_data["target"] = target_dimless_values
        new_df = pd.DataFrame(new_df_data)

        new_registry = PhysicalRegistry()
        zero_dim_vector = [0.0] * len(target_dim_vector)
        for col in new_df.columns:
            new_registry.register(col, zero_dim_vector)

        new_symbolic_mapping = {}
        for i, vec in enumerate(pi_vectors, start=1):
            new_symbolic_mapping[f"Pi_{i}"] = self._vector_to_sympy_expression(context, active_features, vec)

        anchor_expr = self._vector_to_sympy_expression(context, active_features, anchor_vector)
        new_target_expr = context.target_expr / anchor_expr

        new_context = SymbolicRegressionContext(
            df=new_df, registry=new_registry, target_name="target",
            symbolic_mapping=new_symbolic_mapping, target_expr=new_target_expr
        )
        new_context.anchor_expr = anchor_expr  
        return new_context


class DimensionalAnalysisStep(BaseDimensionalAnalysisStep):
    def transform(self, context: 'SymbolicRegressionContext') -> 'SymbolicRegressionContext':
        active_features, target_name, D, target_dim_vector, c_particular, V_canonical = self._extract_dimensional_matrices(context)

        if self.verbose: logger.info("\n=== [ЗАПУСК] Теорема Пи Букингема (Классическая) ===")

        nullspace_vectors = [sp.Matrix(V_canonical[:, col]) for col in range(V_canonical.shape[1])]
        anchor_values = self._evaluate_vector_values(context, active_features, c_particular)
        target_dimless = context.df[target_name].values / (anchor_values + 1e-19)

        if self.verbose:
            anchor_expr = self._vector_to_sympy_expression(context, active_features, c_particular)
            logger.info(f"Размерный якорь: {context.registry.format_expr_inline(anchor_expr)}")
            logger.info(f"Сформировано {len(nullspace_vectors)} безразмерных комплексов.\n")

        return self._create_dimensionless_context(context, active_features, c_particular, nullspace_vectors, target_dimless, target_dim_vector)


class OptDimensionalAnalysisStep(BaseDimensionalAnalysisStep):
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.config = config.dim_analysis

    def transform(self, context: 'SymbolicRegressionContext') -> 'SymbolicRegressionContext':
        active_features, target_name, D, target_dim_vector, c_particular, V_canonical = self._extract_dimensional_matrices(context)

        if self.verbose: logger.info("\n=== [ЗАПУСК] Data-Driven Теорема Пи (STE Якорь) ===")

        X_data = context.df[active_features].values
        Y_data = context.df[target_name].values
        log_X = np.log(np.where(X_data <= 0, 1e-9, X_data))
        log_Y = np.log(np.where(Y_data <= 0, 1e-9, Y_data))

        c_rref_powers = np.array(c_particular, dtype=float).flatten()
        ste_opt = STEAnchorOptimizer(c_particular=c_rref_powers, V=V_canonical, l1_lambda=self.config.ste_l1_lambda)
        c_ste_matrix = ste_opt.fit(log_X, log_Y, epochs=self.config.ste_epochs, lr=self.config.ste_lr)

        anchor_values = self._evaluate_vector_values(context, active_features, c_ste_matrix)
        target_dimless = Y_data / (anchor_values + 1e-19)

        pi_opt = SequentialPiOptimizer(max_integer_search=self.config.max_integer_search, max_redundancy_dcor=self.config.max_redundancy_dcor)
        opt_sympy_vectors = pi_opt.optimize(log_X, target_dimless, V_canonical, c_anchor_vector=c_ste_matrix)

        if self.verbose:
            self._print_optimization_summary(context, active_features, c_particular, c_ste_matrix, opt_sympy_vectors, V_canonical.shape[1], log_Y, target_dimless)

        return self._create_dimensionless_context(context, active_features, c_ste_matrix, opt_sympy_vectors, target_dimless, target_dim_vector)

    def _print_optimization_summary(self, context, active_features, c_rref, c_ste, opt_vectors, k_total, log_Y, target_dimless):
        rref_str = context.registry.format_expr_inline(self._vector_to_sympy_expression(context, active_features, c_rref))
        ste_str = context.registry.format_expr_inline(self._vector_to_sympy_expression(context, active_features, c_ste))

        msg = f"\nРазмерный Якорь (RREF): Anchor_RREF = {rref_str}\n"
        msg += f"Размерный Якорь (STE): Anchor_STE  = {ste_str}\n"
        msg += f"Сжатие Y (std): {np.std(log_Y):.4f} ---> {np.std(np.log(np.where(target_dimless<=0, 1e-9, target_dimless))):.4f}\n"
        msg += f"Полученные Пи-группы (Теоретическое ядро k={k_total} | Отобрано: {len(opt_vectors)}):\n"
        
        if opt_vectors:
            for i, vec in enumerate(opt_vectors, start=1):
                pi_str = context.registry.format_expr_inline(self._vector_to_sympy_expression(context, active_features, vec))
                score = compute_dcor(self._evaluate_vector_values(context, active_features, vec), target_dimless)
                msg += f"  Pi_{i} = {pi_str:<30} | dcor = {score:.4f}\n"
        else:
            msg += "  [ПРЕДУПРЕЖДЕНИЕ] Ни один Пи-комплекс не прошел порог качества.\n"
        
        logger.info(msg)


class GPSimplificationStep(PipelineStep):
    def __init__(self, config: PipelineConfig):
        self.pipeline_config = config
        self.config = config.simplification
        self.bf_config = config.brute_force
        self.verbose = config.verbose

    def transform(self, context):
        if self.verbose:
            logger.info(
                f"\n=== [ЗАПУСК] Шаг GP декомпозиции ===\n"
                f"Активные переменные: {context.get_active_features()}"
            )

        working_context = context.copy()

        with gpytorch.settings.fast_computations(
            solves=False,
            log_prob=False
        ), gpytorch.settings.cholesky_jitter(1e-3):

            final_formula = self._recursive_solve(
                working_context,
                current_depth=0
            )

        working_context.target_expr = final_formula

        if self.verbose:
            logger.info(
                f"GP декомпозиция завершена.\n"
                f"Полученная структура: {final_formula}\n"
                f"====================="
            )

        return working_context

    def _brute_force_symbolic_search(self, context: 'SymbolicRegressionContext', override_length: int) -> sp.Expr:
        feature_names = context.get_active_features()
        local_bf_config = copy.copy(self.bf_config)
        local_bf_config.max_length = override_length
        runner = BruteForceRunner(local_bf_config)

        if self.verbose: logger.info(f"[BF Search] Запуск для {feature_names} (Длина: {override_length})...")
        best_expr, best_mse = runner.run(context)
        
        if best_expr is not None:
            if self.verbose: logger.info(f"[BF Search] Найдено выражение: {best_expr} (MSE: {best_mse:.6e})")
            return best_expr
        
        if self.verbose: logger.info("[BF Search] Формул не найдено. Возврат символьной заглушки.")
        if len(feature_names) == 1: return sp.Function(f"Phi_{feature_names[0]}")(sp.Symbol(feature_names[0]))
        else: return sp.Function("Phi_remaining")(*[sp.Symbol(name) for name in feature_names])


    def _predict_values(self, model, X_eval: np.ndarray) -> np.ndarray:
        model.model.eval()
        model.likelihood.eval()
        with torch.no_grad():
            pred = model.predict(torch.tensor(X_eval, dtype=torch.float64)).mean
        return pred.detach().cpu().numpy()

    def _compute_split_numerical(
        self,
        context: 'SymbolicRegressionContext',
        groups: List[List[str]],
        gp_model,
        split_type: str,
        g_inv_expr: sp.Expr = None,
    ):
        feature_names = context.get_active_features()
        target_name = context.target_name
        X_mat = context.df[feature_names].values
        Y_val = context.df[target_name].values

        group_A = groups[0]
        group_B = [col for g in groups[1:] for col in g]        

        idx_A = [feature_names.index(col) for col in group_A]
        idx_B = [feature_names.index(col) for col in group_B] if group_B else []

        N = len(X_mat)


        if split_type == "General Additive Separability":
            if g_inv_expr is None:
                raise ValueError("Для General Additive Separability необходимо передать g_inv_expr.")

            t_sym = sp.Symbol(target_name, real=True)
            f_ginv = sp.lambdify(t_sym, g_inv_expr, 'numpy')
            
            try:
                Z_val = f_ginv(Y_val)
                if np.isscalar(Z_val):
                    Z_val = np.full(N, float(Z_val))
                Z_val = np.asarray(Z_val, dtype=float)
            except Exception as e:
                raise RuntimeError(f"Ошибка вычисления g^-1(Y) для выражения {g_inv_expr}: {e}")

            if not np.all(np.isfinite(Z_val)) or np.std(Z_val) < 1e-9:
                raise ValueError("Трансформация g^-1(Y) вернула невалидный сигнал (NaN, Inf или константу).")

            cfg = self.pipeline_config.gp_config
            eval_model, _ = train_gp_model(
                torch.tensor(X_mat, dtype=torch.float64),
                torch.tensor(Z_val, dtype=torch.float64),
                cfg
            )
            working_target = Z_val
        else:
            eval_model = gp_model
            working_target = Y_val

        X_center = np.median(X_mat, axis=0, keepdims=True)
        S_0 = float(self._predict_values(eval_model, X_center)[0])
        if abs(S_0) < 1e-7:
            S_0 = 1e-7 * np.sign(S_0 + 1e-12)

        X_eval_A = X_mat.copy()
        if idx_B:
            for j in idx_B:
                X_eval_A[:, j] = X_center[0, j]
        S_A = self._predict_values(eval_model, X_eval_A)

        if group_B:
            X_eval_B = X_mat.copy()
            for j in idx_A:
                X_eval_B[:, j] = X_center[0, j]
            S_B = self._predict_values(eval_model, X_eval_B)
        else:
            S_B = None

        if split_type == "Multiplicative Separability":
            y_A = S_A
            y_B = (S_B / S_0) if group_B else np.ones(N)

        else:
            if group_B:
                min_S_B = float(np.min(S_B))
                c_B = S_0 - min_S_B
                y_A = S_A - c_B
                y_B = working_target - y_A
            else:
                y_A = S_A
                y_B = np.zeros(N)

        df_A = pd.DataFrame(X_mat[:, idx_A], columns=group_A)
        df_A[target_name] = y_A

        if group_B:
            df_B = pd.DataFrame(X_mat[:, idx_B], columns=group_B)
            df_B[target_name] = y_B
        else:
            df_B = pd.DataFrame(columns=group_B)
            df_B[target_name] = y_B

        return df_A, df_B, group_A, group_B

    def combine_formulas(
        self,
        context: 'SymbolicRegressionContext',
        f_A,
        f_B,
        split_type: str,
        g_forward=None,
        g_inv_expr=None,
        c_shift: float = 0.0,
    ) -> sp.Expr:
        from function_analysis.snap import snap_number

        e_A = sp.sympify(f_A)
        e_B = sp.sympify(f_B)
        
        Y_val = context.df[context.target_name].values
        N = len(Y_val)

        def eval_sym(expr: sp.Expr) -> np.ndarray:
            syms = sorted(list(expr.free_symbols), key=lambda s: s.name)
            if not syms:
                return np.full(N, float(expr))
            fn = sp.lambdify(syms, expr, 'numpy')
            args = [context.df[s.name].values for s in syms]
            res = fn(*args)
            if np.isscalar(res):
                res = np.full(N, float(res))
            return np.asarray(res, dtype=float)

        val_A = eval_sym(e_A)
        val_B = eval_sym(e_B)

        if split_type == "Additive Separability":
            A_mat = np.vstack([val_A, val_B, np.ones(N)]).T
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A_mat, Y_val, rcond=None)
                alpha_A, alpha_B, beta = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
            except Exception:
                alpha_A, alpha_B, beta = 1.0, 1.0, 0.0

            if abs(beta) / (np.ptp(Y_val) + 1e-9) < 1e-3:
                beta = 0.0

            snap_A = snap_number(alpha_A)
            snap_B = snap_number(alpha_B)
            snap_b = snap_number(beta)

            combined = snap_A * e_A + snap_B * e_B + snap_b
            return sp.simplify(combined)

        if split_type == "Multiplicative Separability":
            prod_val = val_A * val_B
            A_mat = np.vstack([prod_val, np.ones(N)]).T
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A_mat, Y_val, rcond=None)
                alpha, beta = float(coeffs[0]), float(coeffs[1])
            except Exception:
                alpha, beta = 1.0, 0.0

            if abs(beta) / (np.ptp(Y_val) + 1e-9) < 1e-3:
                beta = 0.0

            snap_alpha = snap_number(alpha)
            snap_beta = snap_number(beta)

            combined = snap_alpha * (e_A * e_B) + snap_beta
            return sp.simplify(combined)

        if split_type == "General Additive Separability":
            if g_inv_expr is not None:
                t_sym = sp.Symbol(context.target_name, real=True)
                fn_ginv = sp.lambdify(t_sym, g_inv_expr, 'numpy')
                Z_val = np.asarray(fn_ginv(Y_val), dtype=float)
            else:
                Z_val = Y_val

            A_mat = np.vstack([val_A, val_B, np.ones(N)]).T
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A_mat, Z_val, rcond=None)
                alpha_A, alpha_B, beta = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
            except Exception:
                alpha_A, alpha_B, beta = 1.0, 1.0, 0.0

            if abs(beta) / (np.ptp(Z_val) + 1e-9) < 1e-3:
                beta = 0.0

            snap_A = snap_number(alpha_A)
            snap_B = snap_number(alpha_B)
            snap_b = snap_number(beta)

            s_expr = snap_A * e_A + snap_B * e_B + snap_b

            if g_forward is not None and list(g_forward.free_symbols):
                target_var = list(g_forward.free_symbols)[0]
                return sp.simplify(g_forward.subs(target_var, s_expr))
            return sp.simplify(s_expr)

        raise ValueError(f"Unknown split: {split_type}")

    def _compute_split_symbolic(
        self, 
        context: 'SymbolicRegressionContext', 
        group_A: List[str], 
        group_B: List[str], 
        split_type: str, 
        g_inv_expr: sp.Expr = None
    ):
        target_name = context.target_name
        original_target_dim = context.registry.get_dim(target_name)
        zero_dim = PhysicalDimension([0.0] * len(original_target_dim.vector))

        if split_type == "General Additive Separability":
            try:
                reg_temp = PhysicalRegistry()
                reg_temp.register(target_name, original_target_dim.vector)
                dim_target_A = DimensionalityEvaluator.evaluate(g_inv_expr, reg_temp)
            except DimensionalError: 
                dim_target_A = zero_dim
            dim_target_B = dim_target_A

        elif split_type == "Multiplicative Separability":
            M_A = context.registry.build_matrix_numpy(group_A)
            M_B = context.registry.build_matrix_numpy(group_B) if group_B else np.zeros((len(original_target_dim.vector), 1))
            dim_Y = original_target_dim.vector

            D_A_vec, D_B_vec = DimensionalProjector.resolve_multiplicative_split(dim_Y, M_A, M_B)

            if self.verbose:
                logger.info(
                    f"[DimensionalProjector] Мультипликативное распределение размерностей:\n"
                    f"  -> Исходный таргет {target_name}: {original_target_dim}\n"
                    f"  -> Блок A {group_A} таргет: PhysicalDimension({list(D_A_vec)})\n"
                    f"  -> Блок B {group_B} таргет: PhysicalDimension({list(D_B_vec)})"
                )

            dim_target_A = PhysicalDimension(D_A_vec)
            dim_target_B = PhysicalDimension(D_B_vec)
        else:
            dim_target_A, dim_target_B = original_target_dim, original_target_dim

        reg_A, reg_B = PhysicalRegistry(), PhysicalRegistry()
        reg_A.register(target_name, dim_target_A.vector)
        reg_B.register(target_name, dim_target_B.vector)

        for col in group_A: 
            reg_A.register(col, context.registry.get_dim(col).vector)
        for col in group_B: 
            reg_B.register(col, context.registry.get_dim(col).vector)

        map_A = {c: context.symbolic_mapping[c] for c in group_A}
        map_B = {c: context.symbolic_mapping[c] for c in group_B}

        return reg_A, reg_B, map_A, map_B

    def _split_context(
        self,
        context,
        groups,
        gp_model,
        split_type,
        g_inv_expr=None,
    ):
        df_A, df_B, grp_A, grp_B = self._compute_split_numerical(
            context,
            groups,
            gp_model,
            split_type,
            g_inv_expr=g_inv_expr,
        )

        reg_A, reg_B, map_A, map_B = self._compute_split_symbolic(
            context,
            grp_A,
            grp_B,
            split_type,
            g_inv_expr=g_inv_expr,
        )

        return (
            SymbolicRegressionContext(
                df_A,
                reg_A,
                context.target_name,
                map_A,
                context.target_expr,
            ),
            SymbolicRegressionContext(
                df_B,
                reg_B,
                context.target_name,
                map_B,
                context.target_expr,
            ),
        )

    def _compute_collapse_numerical(self, context: 'SymbolicRegressionContext', h_expr: sp.Expr):
        df_mut = context.df.copy()
        symbols = sorted(list(h_expr.free_symbols), key=lambda s: s.name)
        vars_in = [s.name for s in symbols]
        
        f_h = sp.lambdify(symbols, h_expr, 'numpy')
        new_col = f"({str(h_expr)})"
        df_mut[new_col] = f_h(*[df_mut[n].values for n in vars_in])
        df_mut.drop(columns=vars_in, inplace=True)
        return df_mut, new_col, vars_in

    def _collapse_context_variables(self, context: 'SymbolicRegressionContext', result):
        h_expr = result[0] if isinstance(result, tuple) else result
        new_context = context.copy()
        
        df_mut, new_col, vars_in = self._compute_collapse_numerical(context, h_expr)
        new_dim = DimensionalityEvaluator.evaluate(h_expr, context.registry)
        
        new_context.df = df_mut
        new_context.register_mutation(vars_in, new_col, h_expr, new_dim)
        return new_context, new_col, h_expr

    def _evaluate_and_apply_simplifier(self, simplifier, context, gp_model, k_sig, cur_loss, depth):
        success, res = simplifier.try_simplify(gp_model, context, k_sigma=k_sig)
        if not success: return False, None, None

        if self.verbose: logger.info(f"[{simplifier.name}] Выдвинута гипотеза: {res}")

        if simplifier.name in [
            "Additive Separability",
            "Multiplicative Separability",
            "General Additive Separability"
        ]:


            if simplifier.name == "General Additive Separability":
                groups = res[0]
                g_fwd = res[1]
                g_inv = res[2]
            else:
                groups = res
                g_fwd = None
                g_inv = None

            if not isinstance(groups, (list, tuple)) or len(groups) < 2:
                if self.verbose:
                    logger.warning(
                        f"[{simplifier.name}] Некорректный split: {groups}"
                    )
                return False, None, None

            if any(not isinstance(group, (list, tuple)) for group in groups):
                if self.verbose:
                    logger.warning(
                        f"[{simplifier.name}] Некорректная структура групп: {groups}"
                    )
                return False, None, None

            ctx_A, ctx_B = self._split_context(
                context,
                groups,
                gp_model,
                simplifier.name,
                g_inv_expr=g_inv,
            )

            _, h_A = train_gp_on_dataframe(ctx_A.df, ctx_A.target_name, self.pipeline_config.gp_config)
            _, h_B = train_gp_on_dataframe(ctx_B.df, ctx_B.target_name, self.pipeline_config.gp_config)
            max_loss = max(h_A[-1] if h_A else 1e5, h_B[-1] if h_B else 1e5)
            d_loss = max_loss - cur_loss

            if self.verbose:
                logger.info(f"[{simplifier.name}] Сравнение лоссов: Original={cur_loss:.4f} | Блок A={h_A[-1] if h_A else 1e5:.4f} | Блок B={h_B[-1] if h_B else 1e5:.4f} | Деградация: {d_loss:.4f}")

            if d_loss <= self.config.loss_degradation_tolerance:
                if self.verbose: logger.info(f"[{simplifier.name}] ПРИНЯТО (Деградация {d_loss:.4f} <= {self.config.loss_degradation_tolerance})")
                form_A = self._recursive_solve(ctx_A, depth + 1)
                form_B = self._recursive_solve(ctx_B, depth + 1)
                context.symbolic_mapping.update(ctx_A.symbolic_mapping)
                context.symbolic_mapping.update(ctx_B.symbolic_mapping)
                return True, self.combine_formulas(context, form_A, form_B, simplifier.name, g_fwd, g_inv_expr=g_inv), None
            else:
                if self.verbose: logger.info(f"[{simplifier.name}] ОТКЛОНЕНО (Деградация лосса: +{d_loss:.4f} > Порог: {self.config.loss_degradation_tolerance})")
                return False, None, {
                    'delta_loss': d_loss, 
                    'type': 'split', 
                    'split_name': simplifier.name, 
                    'context_A': ctx_A, 
                    'context_B': ctx_B, 
                    'g_forward': g_fwd,
                    'c_shift': 0.0
                }
        else:
            ctx_mut, col, h_loc = self._collapse_context_variables(context, res)
            _, h_mut = train_gp_on_dataframe(ctx_mut.df, ctx_mut.target_name, self.pipeline_config.gp_config)
            d_loss = (h_mut[-1] if h_mut else 1e5) - cur_loss

            if self.verbose: logger.info(f"[{simplifier.name}] Сравнение лоссов: Original={cur_loss:.4f} | Мутация={h_mut[-1] if h_mut else 1e5:.4f} | Деградация: {d_loss:.4f}")

            if d_loss <= self.config.loss_degradation_tolerance:
                if self.verbose: logger.info(f"[{simplifier.name}] ПРИНЯТО (Деградация {d_loss:.4f} <= {self.config.loss_degradation_tolerance})")
                return True, self._recursive_solve(ctx_mut, depth + 1).subs(sp.Symbol(col), h_loc), None
            else:
                if self.verbose: logger.info(f"[{simplifier.name}] ОТКЛОНЕНО (Деградация лосса: +{d_loss:.4f} > Порог: {self.config.loss_degradation_tolerance})")
                return False, None, {'delta_loss': d_loss, 'type': 'collapse', 'mutated_context': ctx_mut, 'new_col': col, 'h_expr_loc': h_loc, 'split_name': simplifier.name}

    def _recursive_solve(self, context: 'SymbolicRegressionContext', current_depth: int) -> sp.Expr:
        feats = context.get_active_features()
        if len(feats) == 1 or (self.config.max_depth is not None and current_depth >= self.config.max_depth):
            return self._brute_force_symbolic_search(context, self.config.final_bf_max_length)
        
        if self.verbose: logger.info(f"\n--- [Глубина {current_depth}] Обучение GP для переменных: {feats} ---")
        gp_model, hist = train_gp_on_dataframe(context.df, context.target_name, self.pipeline_config.gp_config)
        cur_loss = hist[-1] if hist else 1e5

        simplifiers = [
            AdditiveSeparabilitySimplifier(self.pipeline_config), 
            MultiplicationSeparabilitySimplifier(self.pipeline_config),
            GeneralAdditiveSeparabilitySimplifier(self.pipeline_config),
            CompositionalitySimplifier(self.pipeline_config)
        ]
        
        rej = []
        for mult in ([1.0] if len(feats) <= 2 else self.config.k_sigma_multipliers):
            k_sig = self.config.base_k_sigma * mult
            for s in simplifiers:
                success, form, rej_data = self._evaluate_and_apply_simplifier(s, context, gp_model, k_sig, cur_loss, current_depth)
                if success: return form
                if rej_data: rej.append(rej_data)

        if rej:
            best = min(rej, key=lambda x: x['delta_loss'])
            if best['delta_loss'] <= 0.8:
                if self.verbose: logger.info(f"\nПрименяем лучший fallback (Delta Loss: +{best['delta_loss']:.4f})")
                if best['type'] == 'split':
                    return self.combine_formulas(
                        context,
                        self._recursive_solve(best['context_A'], current_depth + 1),
                        self._recursive_solve(best['context_B'], current_depth + 1),
                        best['split_name'],
                        best.get('g_forward')
                    )
                else:
                    return self._recursive_solve(best['mutated_context'], current_depth + 1).subs(sp.Symbol(best['new_col']), best['h_expr_loc'])

        if self.verbose: logger.info(f"[Рекурсия] Переход к финальному BF.")
        return self._brute_force_symbolic_search(context, self.config.final_bf_max_length)
    
class BaselineGPStep(PipelineStep):
    def __init__(self, config: PipelineConfig):
        self.gp_config = config.gp_config
        self.verbose = config.verbose

    def transform(self, context: 'SymbolicRegressionContext') -> 'SymbolicRegressionContext':
        if self.verbose: 
            logger.info("\n=== [ЗАПУСК] Холостой запуск GP (Оценка базовой точности) ===")
            
        pipeline, _ = train_gp_on_dataframe(context.df, context.target_name, self.gp_config)
        features = context.get_active_features()
        train_x = torch.tensor(context.df[features].values, dtype=torch.float64)
        pipeline.model.eval()
        pipeline.likelihood.eval()

        with torch.no_grad():
            y_pred = pipeline.predict(train_x).mean.cpu().numpy()

        y_true = context.df[context.target_name].values
        mse = float(np.mean((y_true - y_pred) ** 2))
        r2 = float(1.0 - (np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12)))

        if self.verbose:
            log_msg = f"\n[Базовая GP модель ({context.target_name})]:\n"
            log_msg += f"   Точность: MSE = {mse:.6e} | R² = {r2:.6f}\n"

            try:
                raw_noise = float(pipeline.likelihood.noise.item())
                sy = float(pipeline.scale_y_factor.item()) if pipeline.scale_y_factor is not None else 1.0
                phys_noise_std = np.sqrt(raw_noise) * sy
                log_msg += f"   Шум измерения (Noise Variance): {raw_noise:.4e} (std в физ. ед.: {phys_noise_std:.4f})\n"
            except Exception:
                pass

            try:
                from GP.utils import format_kernel_summary
                kernel_summary = format_kernel_summary(
                    pipeline.model.covar_module,
                    features,
                    scale_x_factor=pipeline.scale_x_factor,
                    scale_y_factor=pipeline.scale_y_factor
                )
                log_msg += kernel_summary + "\n"
            except Exception as e:
                log_msg += f"   (Не удалось извлечь параметры ядра: {e})\n"

            log_msg += "===================================\n"
            logger.info(log_msg)

        return context