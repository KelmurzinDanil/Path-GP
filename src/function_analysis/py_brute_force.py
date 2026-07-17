import sympy as sp
import fast_symbolic 
import numpy as np
import traceback


from typing import List, Optional, Tuple, Dict

from symbolic.sm_context import SymbolicRegressionContext, DimensionalityEvaluator
from function_analysis.snap import scipy_refit, snap_and_reoptimize, eval_sympy_mse
from get_pi_complex import DimensionalError, PhysicalDimension

class BruteForceRunner:
    def __init__(self, max_length: int = 8, complexity_penalty: float = 0.05,
                 optimize_constants: bool = True, allowed_constants: List[float] = [1.0, 2.0],
                 allowed_ops: List[str] = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"]):
        self.max_length = max_length
        self.complexity_penalty = complexity_penalty
        self.optimize_constants = optimize_constants
        self.allowed_constants = allowed_constants
        self.allowed_ops = allowed_ops

    def run(self, context: SymbolicRegressionContext, data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[Optional[sp.Expr], float]:
        if data is not None:
            X_np, Y_np = data
        else:
            active_features = context.get_active_features()
            X_np = context.df[active_features].values
            Y_np = context.df[context.target_name].values

        active_features = context.get_active_features()
        print(f"[BF] Запуск перебора C++ для {len(active_features)} переменных. Max length: {self.max_length}")

        raw_candidates = fast_symbolic.run_brute_force(
                    X_np.tolist(), 
                    Y_np.tolist(), 
                    self.max_length, 
                    self.allowed_constants, 
                    self.optimize_constants,
                    self.allowed_ops
                )   
             
        if not raw_candidates:
            print("[BF] C++ движок не вернул кандидатов.")
            return None, float('inf')
        
        valid_candidates = []

        try:
            target_dim = context.registry.get_dim(context.target_name)
        except KeyError:
            sample_dim = list(context.registry.variables.values())[0].dimension
            target_dim = PhysicalDimension.dimensionless_like(sample_dim)

        print(f"[BF] Получено кандидатов от C++: {len(raw_candidates)}. Начинаем валидацию...")
        
        for record in raw_candidates:
            expr = rpn_to_sympy(record.expression)
            if expr is None:
                continue

            mapped_expr = self._map_symbols_to_features(expr, active_features)
            try:
                candidate_dim = DimensionalityEvaluator.evaluate(mapped_expr, context.registry)

                if candidate_dim != target_dim:
                    continue
            except DimensionalError:
                continue
            except NotImplementedError:
                print("Неподдерживаемая SymPy-нода")
                continue

            valid_candidates.append({
                'expr_unmapped': expr,        
                'expr_mapped': mapped_expr,   
                'raw_complexity': record.complexity,
                'raw_mse': record.mse
            })

        print(f"[BF] Физическую валидацию прошли: {len(valid_candidates)} из {len(raw_candidates)}")
        if not valid_candidates:
            return None, float('inf')
        
        best_expr = None
        best_score = float('inf')
        best_mse = float('inf')
        
        for cand in valid_candidates:
            if self.optimize_constants:
                try:
                    opt_expr_unmapped, fit_mse = scipy_refit(cand['expr_unmapped'], X_np, Y_np, start_params=None)
                    snapped_expr_unmapped = snap_and_reoptimize(opt_expr_unmapped, X_np, Y_np)
                    
                    snapped_expr = self._map_symbols_to_features(snapped_expr_unmapped, active_features)
                    final_mse = eval_sympy_mse(snapped_expr_unmapped, X_np, Y_np)

                except Exception as e:
                    print(f"[ERROR] Ошибка при оптимизации или притягивании констант: {e}")
                    traceback.print_exc() 
                    snapped_expr = cand['expr_mapped']
                    final_mse = cand['raw_mse']
            else:
                snapped_expr = cand['expr_mapped']
                final_mse = cand['raw_mse']
                
            complexity = cand['raw_complexity']
            score = np.log(final_mse + 1e-12) + self.complexity_penalty * complexity

            if score < best_score:
                best_score = score
                best_expr = snapped_expr
                best_mse = final_mse

        return best_expr, best_mse
    
    def _map_symbols_to_features(self, expr: sp.Expr, active_features: List[str]) -> sp.Expr:
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
    if not tokens:
        return None
    
    stack = []

    for t in tokens:
        if t.startswith('x'):
            try:
                idx = int(t[1:])
                stack.append(sp.Symbol(f'x_{idx}'))
            except ValueError:
                return None 
        elif t in ['+', '-', '*', '/', '^']:
            if len(stack) < 2:
                return None

            b = stack.pop()
            a = stack.pop()

            if t == '+': stack.append(a + b)
            elif t == '-': stack.append(a - b)
            elif t == '*': stack.append(a * b)
            elif t == '/': stack.append(a / b)
            elif t == '^': stack.append(a ** b)
        elif  t in ['sin', 'cos', 'tan', 'abs', 'sqrt', 'log', 'exp']:
            if len(stack) < 1:
                return None 
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
            
    if len(stack) != 1:
        return None
        
    return stack[0]

def primary_clean(expr: sp.Expr) -> sp.Expr:
    if expr is None:
        return None
    return sp.simplify(expr)

