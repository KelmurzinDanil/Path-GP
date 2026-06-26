from function_analysis.recursive_apply_simplifications import recursive_solve
from generate.generate_dataset import *
from GP.config import GPConfig, ModelConfig, KernelConfig, TrainingConfig
from GP.pipeline import GPRegressionPipeline
import sympy as sp

def parse_collapsed_name(name_str: str) -> sp.Expr:
    """
    Рекурсивно парсит имена свернутых колонок типа '((Pi_3_minus_Pi_4)_div_Pi_2)'
    обратно в чистые математические выражения SymPy.
    """
    if name_str.startswith('(') and name_str.endswith(')'):
        name_str = name_str[1:-1]
    
    if "_minus_" in name_str:
        parts = name_str.split("_minus_")
        return parse_collapsed_name(parts[0]) - parse_collapsed_name(parts[1])
    elif "_div_" in name_str:
        parts = name_str.split("_div_")
        return parse_collapsed_name(parts[0]) / parse_collapsed_name(parts[1])
    elif "_mul_" in name_str:
        parts = name_str.split("_mul_")
        return parse_collapsed_name(parts[0]) * parse_collapsed_name(parts[1])
    else:
        return sp.Symbol(name_str)

def resolve_collapsed_symbols(expr: sp.Expr) -> sp.Expr:
    """
    Находит в выражении все свернутые символы и заменяет их на раскрытые формулы.
    """
    syms = expr.free_symbols
    sub_dict = {}
    for sym in syms:
        if "_minus_" in sym.name or "_div_" in sym.name or "_mul_" in sym.name:
            sub_dict[sym] = parse_collapsed_name(sym.name)
    return expr.subs(sub_dict)

def display_final_physics_formula(formula: sp.Expr, reg: PhysicalRegistry, target_name: str, 
                                  c_particular: sp.Matrix, nullspace_vectors: list[sp.Matrix]):
    """
    Выводит два красивых варианта формулы: безразмерный (якорь + Phi) и полностью раскрытый физический.
    """
    names = [name for name in reg.get_all_variables() if name != target_name]
    
    resolved_formula = resolve_collapsed_symbols(formula)

    anchor_expr = reg.vector_to_formula(c_particular, names)
    
    pi_subs = {}
    for i, vec in enumerate(nullspace_vectors, start=1):
        pi_subs[sp.Symbol(f"Pi_{i}")] = reg.vector_to_formula(vec, names)
        
    print("\n" + "="*60)

    print("\n1. В безразмерных П-комплексах (Пи-группах):")
    anchor_str = reg.format_expr_inline(anchor_expr)
    resolved_formula_str = reg.format_expr_inline(resolved_formula)
    print(f"   {target_name} = {anchor_str} · {resolved_formula_str}")

    print("\n2. Полное раскрытие через исходные физические переменные:")
    
    physical_inner_expr = resolved_formula.subs(pi_subs)
    
    full_physical_expr = anchor_expr * physical_inner_expr
    
    simplified_physical_expr = sp.simplify(full_physical_expr)
    
    final_physics_str = reg.format_expr_inline(simplified_physical_expr)
    print(f"   {target_name} = {final_physics_str}")
    print("\n" + "="*60 + "\n")



if __name__ == "__main__":

    df, reg, target_name, c_part, nullspace = generate_boltzmann_density_formula(1000)

    config = GPConfig(
        model=ModelConfig(
            mean_type="constant",
            kernel=KernelConfig(
                type="matern_52",    
                scale_kernel=True,   
                ard=True             
            )
        ),
        training=TrainingConfig(
            lr=0.02,                 
            epochs=1000,    
            early_stopping_patience=15,          
            optimizer="adam",
            loss_type = "mll",
            verbose=False     
        ),
    )
    formula = recursive_solve(df, config)

    display_final_physics_formula(
        formula=formula,
        reg=reg,
        target_name=target_name,
        c_particular=c_part,
        nullspace_vectors=nullspace
    )