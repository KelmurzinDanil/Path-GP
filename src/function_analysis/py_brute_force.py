import sympy as sp
import fast_symbolic 
import numpy as np
import traceback
from scipy.stats import linregress
from typing import List, Optional, Tuple
from fractions import Fraction
from symbolic.sm_context import SymbolicRegressionContext, DimensionalityEvaluator
from get_pi_complex import DimensionalError, PhysicalDimension
from config import BruteForceConfig
from logger import setup_logger
from function_analysis.snap import snap_and_reoptimize
logger = setup_logger("BRF")
    
def map_symbols_to_features(expr: sp.Expr, active_features: List[str]) -> sp.Expr:
    """Универсальная функция маппинга переменных x_0, x_1 на реальные имена."""
    sub_dict = {}
    for sym in expr.free_symbols:
        if sym.name.startswith('x_'):
            try:
                idx = int(sym.name.split('_')[1])
                if idx < len(active_features):
                    sub_dict[sym] = sp.Symbol(active_features[idx])
            except ValueError:
                pass
    return expr.subs(sub_dict)

def rpn_to_sympy(rpn_str: str) -> sp.Expr:
    tokens = rpn_str.strip().split()
    if not tokens: return None
    
    stack = []
    for t in tokens:
        if t.startswith('x'):
            try:
                idx = int(t[1:])
                stack.append(sp.Symbol(f'x_{idx}'))
            except ValueError:
                return None 
        elif t in ['+', '-', '*', '/', '^']:
            if len(stack) < 2: return None
            b = stack.pop()
            a = stack.pop()
            if t == '+': stack.append(a + b)
            elif t == '-': stack.append(a - b)
            elif t == '*': stack.append(a * b)
            elif t == '/': stack.append(a / b)
            elif t == '^': stack.append(a ** b)
        elif  t in ['sin', 'cos', 'tan', 'abs', 'sqrt', 'log', 'exp']:
            if len(stack) < 1: return None 
            arg = stack.pop()
            if t == 'sin': stack.append(sp.sin(arg))
            elif t == 'cos': stack.append(sp.cos(arg))
            elif t == 'tan': stack.append(sp.tan(arg))
            elif t == 'abs': stack.append(sp.Abs(arg))
            elif t == 'sqrt': stack.append(sp.sqrt(arg))
            elif t == 'log': stack.append(sp.log(arg))
            elif t == 'exp': stack.append(sp.exp(arg))
        else:
            try:
                val = float(t)
                if val.is_integer():
                    stack.append(sp.Integer(int(val)))
                else:
                    stack.append(sp.Float(val))
            except ValueError:
                return None
            
    if len(stack) != 1: return None
    return stack[0]

class BruteForceRunner:
    def __init__(self, config: BruteForceConfig):
        self.config = config

    def get_valid_candidates(
        self, 
        context: SymbolicRegressionContext, 
        X_np: np.ndarray, 
        Y_np: np.ndarray, 
        active_features: List[str],
        target_dim: Optional[PhysicalDimension] = None,
        require_all_vars: bool = False
    ) -> List[dict]:
        """Общий метод для получения, парсинга и валидации размерностей кандидатов."""
        raw_candidates = fast_symbolic.run_brute_force(
            X_np.tolist(), Y_np.tolist(), self.config.max_length, 
            self.config.allowed_constants, self.config.optimize_constants,
            self.config.allowed_ops, self.config.k_best
        )   
        
        valid_candidates = []
        for record in raw_candidates:
            expr = rpn_to_sympy(record.expression)
            if expr is None: continue

            mapped_expr = map_symbols_to_features(expr, active_features)
            
            if require_all_vars:
                found_vars = {sym.name for sym in mapped_expr.free_symbols}
                if not all(feat in found_vars for feat in active_features):
                    continue

            try:
                candidate_dim = DimensionalityEvaluator.evaluate(mapped_expr, context.registry)
                if target_dim is not None and candidate_dim != target_dim:
                    continue
            except (DimensionalError, NotImplementedError):
                continue

            valid_candidates.append({
                'expr_unmapped': expr,        
                'expr_mapped': mapped_expr,   
                'raw_complexity': record.complexity,
                'raw_mse': record.mse
            })
        return valid_candidates
    
    def _eval_expr_on_data(
        self,
        expr: sp.Expr,
        X_np: np.ndarray,
        active_features: List[str],
    ) -> np.ndarray:
        free_syms = sorted(list(expr.free_symbols), key=lambda s: s.name)
        f_fn = sp.lambdify(free_syms, expr, 'numpy')
        args = []
        for sym in free_syms:
            if sym.name in active_features:
                args.append(X_np[:, active_features.index(sym.name)])
            elif sym.name.startswith('x_'):
                idx = int(sym.name.split('_')[1])
                args.append(X_np[:, idx])
        vals = np.asarray(f_fn(*args), dtype=float)
        return vals

    def _compute_mse(
        self,
        expr: sp.Expr,
        X_np: np.ndarray,
        Y_np: np.ndarray,
        active_features: List[str],
    ) -> float:
        pred = self._eval_expr_on_data(expr, X_np, active_features)
        return float(np.mean((pred - Y_np) ** 2))       

    def run(self, context: SymbolicRegressionContext, data: Optional[Tuple[np.ndarray, np.ndarray]] = None, target_dim_check: bool = True, require_all_vars: bool = False) -> Tuple[Optional[sp.Expr], float]:
        active_features = context.get_active_features()
        if data is not None:
            X_np, Y_np = data
        else:
            X_np = context.df[active_features].values
            Y_np = context.df[context.target_name].values

        target_dim = None
        if target_dim_check:
            try:
                target_dim = context.registry.get_dim(context.target_name)
            except KeyError:
                sample_dim = list(context.registry.variables.values())[0].dimension
                target_dim = PhysicalDimension.dimensionless_like(sample_dim)

        valid_candidates = self.get_valid_candidates(context, X_np, Y_np, active_features, target_dim, require_all_vars)

        if not valid_candidates:
            return None, float('inf')

        ranked_candidates = []
        for cand in valid_candidates:
            try:
                pred = self._eval_expr_on_data(cand["expr_unmapped"], X_np, active_features)
                if not np.all(np.isfinite(pred)) or np.std(pred) < 1e-12:
                    continue
                
                slope, intercept, r_val, _, _ = linregress(pred, Y_np)


                if np.isnan(r_val) or abs(r_val) < 1e-6:
                    continue
                
                r2 = r_val ** 2
                unexplained_var = max(1e-15, 1.0 - r2)
                
                final_score = np.log(unexplained_var) + 0.05 * cand["raw_complexity"]

                ranked_candidates.append((final_score, cand, unexplained_var))
            except Exception:
                continue

        if not ranked_candidates:
            return None, float("inf")

        ranked_candidates.sort(key=lambda item: item[0])


        logger.info(f"[{active_features}] Топ-10 кандидатов BRF:")
        for s, c, uv in ranked_candidates[:10]:
            logger.info(f"  -> {str(c['expr_mapped']):<25} | 1-R²: {uv:.2e} | comp: {c['raw_complexity']} | score: {s:.3f}")

        best_cand = ranked_candidates[0][1]
        best_expr = best_cand["expr_mapped"]
        best_unmapped = best_cand["expr_unmapped"]

        pred = self._eval_expr_on_data(best_unmapped, X_np, active_features)

        slope, intercept, r_val, _, _ = linregress(pred, Y_np)
        if np.isnan(slope): slope = 1.0
        if np.isnan(intercept): intercept = 0.0

        from function_analysis.snap import snap_number
        snap_slope = snap_number(slope)
        snap_intercept = snap_number(intercept)

        calibrated_unmapped = snap_slope * best_unmapped + snap_intercept

        snapped_expr = snap_and_reoptimize(
            calibrated_unmapped,
            X_np,
            Y_np,
            rel_tol=self.config.snap_rel_tol,
            abs_tol=self.config.snap_abs_tol,
        )
        final_expr = map_symbols_to_features(snapped_expr, active_features)
        final_expr = sp.simplify(final_expr)

        final_mse = self._compute_mse(final_expr, X_np, Y_np, active_features)

        logger.info(
            f"[BRF] Форма: {best_expr} (1-R²: {ranked_candidates[0][2]:.2e}) | "
            f"Итог: {final_expr} | MSE={final_mse:.6e}"
        )

        return final_expr, final_mse