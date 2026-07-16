import sympy as sp
import numpy as np
from scipy.optimize import minimize
from fractions import Fraction


def eval_sympy_mse(expr, X_np, Y_np):
    symbols = sorted([s for s in expr.free_symbols if s.name.startswith('x_')], key=lambda s: int(s.name.split('_')[1]))
    if not symbols:
        try:
            val = float(expr)
            return np.mean((val - Y_np) ** 2)
        except:
            return float('inf')
    f = sp.lambdify(symbols, expr, 'numpy')
    try:
        args = [X_np[:, int(s.name.split('_')[1])] for s in symbols]
        pred = f(*args)
        if np.isscalar(pred):
            pred = np.full_like(Y_np, pred)
        return np.mean((pred - Y_np) ** 2)
    except Exception:
        return float('inf')
    
def scipy_refit(expr, X_np, Y_np, start_params):
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
        return node

    param_expr = expr.replace(lambda x: isinstance(x, sp.Float), replace_float_node)

    if not param_symbols:
        return expr, eval_sympy_mse(expr, X_np, Y_np)
    
    if start_params is None:
        start_params = initial_values

    x_symbols = sorted([s for s in param_expr.free_symbols if s.name.startswith('x_')], key=lambda s: int(s.name.split('_')[1]))
    all_symbols = x_symbols + param_symbols
    f_compiled = sp.lambdify(all_symbols, param_expr, 'numpy')

    def loss(p_vals):
        try:
            x_args = [X_np[:, int(s.name.split('_')[1])] for s in x_symbols]
            pred = f_compiled(*(x_args + list(p_vals)))
            if np.isscalar(pred):
                pred = np.full_like(Y_np, pred)
            return np.mean((pred - Y_np) ** 2)
        except:
            return 1e10
        
    res = minimize(loss, start_params, method='Nelder-Mead')
    optimized_expr = param_expr.subs(dict(zip(param_symbols, res.x)))
    return optimized_expr, res.fun

def snap_and_reoptimize(expr, X_np, Y_np, rel_tol=1.05, abs_tol=1e-6):
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

    current_params = list(initial_values)
    base_mse = eval_sympy_mse(expr, X_np, Y_np)
    allowed_error = max(base_mse * rel_tol, base_mse + abs_tol)

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
            except:
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

        # Zero-snap
        if abs(val) < 0.1:
            hypotheses.append(sp.Integer(0))

        # Integer-snap
        hypotheses.append(sp.Integer(round(val)))

        # Rational-snap 
        frac = Fraction(val).limit_denominator(10)
        if frac.denominator > 1 and frac.denominator <= 5: 
            hypotheses.append(sp.Rational(frac.numerator, frac.denominator))

        snap_successful = False
        for hyp in hypotheses:
            temp_expr = active_expr.subs(target_symbol, hyp)
            remaining_symbols = [s for s in active_symbols if s != target_symbol]
            remaining_initials = [active_values[j] for j in range(len(active_symbols)) if j != i]
            test_expr, test_mse, opt_remaining_vals = reoptimize_free_params(
                temp_expr, remaining_symbols, remaining_initials
            )
            if test_mse <= allowed_error:
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
            
    return active_expr