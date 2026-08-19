import sympy as sp
import numpy as np
from scipy.optimize import minimize
from fractions import Fraction

def eval_sympy_mse(expr: sp.Expr, X_np: np.ndarray, Y_np: np.ndarray) -> float:
    symbols = sorted([s for s in expr.free_symbols if s.name.startswith('x_')], key=lambda s: int(s.name.split('_')[1]))
    if not symbols:
        try:
            val = float(expr)
            return float(np.mean((val - Y_np) ** 2))
        except Exception:
            return float('inf')
    f = sp.lambdify(symbols, expr, 'numpy')
    try:
        args = [X_np[:, int(s.name.split('_')[1])] for s in symbols]
        pred = f(*args)
        if np.isscalar(pred):
            pred = np.full_like(Y_np, pred)
        return float(np.mean((pred - Y_np) ** 2))
    except Exception:
        return float('inf')

def snap_number(val: float, abs_zero_tol: float = 1e-4) -> sp.Expr:
    if abs(val) < abs_zero_tol:
        return sp.Integer(0)
    
    round_int = int(np.round(val))
    if abs(val - round_int) < 1e-4:
        return sp.Integer(round_int)
    
    frac = Fraction(val).limit_denominator(12)
    if abs(float(frac) - val) < 1e-4:
        return sp.Rational(frac.numerator, frac.denominator)
        
    for base in [2, 3, 5]:
        sq = np.sqrt(base)
        for multiplier in [1.0, 0.5, 2.0, 1.0/3.0]:
            target = multiplier * sq
            if abs(abs(val) - target) < 1e-4:
                sign = 1 if val > 0 else -1
                num_frac = Fraction(multiplier).limit_denominator(6)
                return sign * sp.Rational(num_frac.numerator, num_frac.denominator) * sp.sqrt(base)
                
    return sp.Float(val)

def snap_and_reoptimize(expr: sp.Expr, X_np: np.ndarray, Y_np: np.ndarray, rel_tol: float = 1.05, abs_tol: float = 1e-6) -> sp.Expr:
    param_symbols = []
    initial_values = []
    counter = 0

    def replace_float_node(node):
        nonlocal counter
        if isinstance(node, sp.Float):
            symbol = sp.Symbol(f'p_{counter}')
            param_symbols.append(symbol)
            initial_values.append(float(node))
            counter += 1
            return symbol
        return node

    param_expr = expr.replace(lambda x: isinstance(x, sp.Float), replace_float_node)

    if not param_symbols:
        return sp.simplify(expr)

    x_symbols = sorted(
        [s for s in param_expr.free_symbols if s.name.startswith('x_')],
        key=lambda s: int(s.name.split('_')[1])
    )

    var_y = float(np.var(Y_np))
    if var_y < 1e-12:
        return sp.simplify(expr)

    base_mse = eval_sympy_mse(expr, X_np, Y_np)
    allowed_mse = base_mse * rel_tol + (var_y * 1e-4)

    def reoptimize_free_params(active_expr, free_symbols, free_initials):
        if not free_symbols:
            return active_expr, eval_sympy_mse(active_expr, X_np, Y_np), []

        all_symbols = x_symbols + free_symbols
        f_compiled = sp.lambdify(all_symbols, active_expr, 'numpy')

        def loss(p_vals):
            try:
                x_args = [X_np[:, int(s.name.split('_')[1])] for s in x_symbols]
                pred = f_compiled(*(x_args + list(p_vals)))
                if np.isscalar(pred):
                    pred = np.full_like(Y_np, pred)
                return np.mean((pred - Y_np) ** 2)
            except Exception:
                return 1e10

        res = minimize(loss, free_initials, method='Nelder-Mead')
        optimized_expr = active_expr.subs(dict(zip(free_symbols, res.x)))
        return optimized_expr, res.fun, list(res.x)

    active_expr = param_expr
    active_symbols = list(param_symbols)
    active_values = list(initial_values)

    i = 0
    while i < len(active_symbols):
        target_symbol = active_symbols[i]
        val = active_values[i]

        hypotheses = []
        
        frac = Fraction(val).limit_denominator(8)
        if frac.denominator > 1:
            hypotheses.append(sp.Rational(frac.numerator, frac.denominator))

        if abs(val) >= 0.4:
            hypotheses.append(sp.Integer(int(np.round(val))))
            
        if abs(val) < 0.05:
            hypotheses.append(sp.Integer(0))

        snap_successful = False
        for hyp in hypotheses:
            temp_expr = active_expr.subs(target_symbol, hyp)
            
            if x_symbols and not any(s in temp_expr.free_symbols for s in x_symbols):
                continue
                
            remaining_symbols = [s for s in active_symbols if s != target_symbol]
            remaining_initials = [active_values[j] for j in range(len(active_symbols)) if j != i]
            
            test_expr, test_mse, opt_remaining_vals = reoptimize_free_params(
                temp_expr, remaining_symbols, remaining_initials
            )
            
            if test_mse <= allowed_mse:
                active_expr = test_expr
                active_symbols = remaining_symbols
                active_values = opt_remaining_vals
                snap_successful = True
                break

        if snap_successful:
            i = 0
            continue
        else:
            active_expr, _, active_values = reoptimize_free_params(
                active_expr, active_symbols, active_values
            )
            i += 1

    return sp.simplify(active_expr)