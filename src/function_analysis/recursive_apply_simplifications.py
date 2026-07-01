from .simplify import *
from GP.config import GPConfig, ModelConfig, KernelConfig, TrainingConfig
from GP.pipeline import GPRegressionPipeline


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

def do_task_split(dataset: pd.DataFrame, groups, gp_model, split_type: str):
    feature_cols = [col for col in dataset.columns if col != 'target']
    group_A = groups[0]
    group_B = []
    for g in groups[1:]:
        group_B.extend(g)

    y_original = dataset['target'].values
    n_samples = len(dataset)

    pts_A = torch.tensor(dataset[feature_cols].values, dtype=torch.float32)

    for col_name in group_B:
        col_idx = feature_cols.index(col_name)
        median_val = dataset[col_name].median()
        pts_A[:, col_idx] = torch.full((n_samples,), median_val)

    # gp_model.eval()
    with torch.no_grad():
        y_A = gp_model.predict(pts_A).mean.numpy()

    df_A = dataset[group_A].copy()
    df_A['target'] = y_A

    df_B = dataset[group_B].copy()

    if split_type == "Additive Separability" or split_type == "General Additive Separability":
        df_B['target'] = y_original - y_A
    elif split_type == "Multiplicative Separability":
        df_B['target'] = y_original / (y_A + 1e-9)
        
    return df_A, df_B

def do_variable_collapse(dataset: pd.DataFrame, result, gp_model, split_type: str):
    df_mutated = dataset.copy()

    if split_type in ["Translational Symmetry", "LargeScale Symmetry", "Multiply Symmetry", "Addition Symmetry"]:
        if isinstance(result[0], list):
            group = next(g for g in result if len(g) >= 2)
        else:
            group = result
            
        x1_name, x2_name = group[0], group[1]

        if split_type == "Translational Symmetry":
            new_col_name = f"({x1_name}_minus_{x2_name})"
            df_mutated[new_col_name] = df_mutated[x1_name] - df_mutated[x2_name]
        
        elif split_type == "Addition Symmetry": 
                new_col_name = f"({x1_name}_plus_{x2_name})"
                df_mutated[new_col_name] = df_mutated[x1_name] + df_mutated[x2_name]

        elif split_type == "LargeScale Symmetry":
            new_col_name = f"({x1_name}_div_{x2_name})"
            df_mutated[new_col_name] = df_mutated[x1_name] / (df_mutated[x2_name] + 1e-9)
            
        elif split_type == "Multiply Symmetry":
            new_col_name = f"({x1_name}_mul_{x2_name})"
            df_mutated[new_col_name] = df_mutated[x1_name] * df_mutated[x2_name]

        df_mutated.drop(columns=[x1_name, x2_name], inplace=True)

    elif split_type == "Compositionality":
        h_expr = result

        symbols = sorted(list(h_expr.free_symbols), key=lambda s: s.name)
        involved_vars = [sym.name for sym in symbols]
        f_h = sp.lambdify(symbols, h_expr, 'numpy')

        args = [df_mutated[name].values for name in involved_vars]
        new_col_name = f"({str(h_expr)})"
        df_mutated[new_col_name] = f_h(*args)

        df_mutated.drop(columns=involved_vars, inplace=True)

    elif split_type == "Generalized Symmetry":
        group = result 
        print(f"Запуск локального поиска формулы связи для группы {group}...")
        h_expr = find_best_h_for_group(gp_model, dataset, group) 
        
        return do_variable_collapse(dataset, h_expr, gp_model, "Compositionality")

    return df_mutated

def combine_formulas(formula_A, formula_B, split_type: str) -> sp.Expr:
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

def brute_force_symbolic_search(dataset: pd.DataFrame) -> sp.Expr:
    feature_names = [col for col in dataset.columns if col != 'target']
    x_name = feature_names[0]
    
    phi = sp.Function(f"Phi_{x_name}")
    
    return phi(sp.Symbol(x_name))

def train_gp(dataset: pd.DataFrame, config: GPConfig) -> GPRegressionPipeline:
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

def recursive_solve(dataset: pd.DataFrame, config: GPConfig):
    feature_names = [col for col in dataset.columns if col != 'target']
    
    if len(feature_names) == 1:
        return brute_force_symbolic_search(dataset)

    print(f"\n--- Обучение вашего GP для {len(feature_names)} переменных: {feature_names} ---")
    gp_model = train_gp(dataset, config)

    for simplifier in simplifiers:
        success, result = simplifier.try_simplify(gp_model, dataset)
        if success:
            print(f"Сработал метод: {simplifier.name}")

            if simplifier.name in ["Additive Separability", "Multiplicative Separability"]:
                dataset_A, dataset_B = do_task_split(dataset, result, gp_model, simplifier.name)
                formula_A = recursive_solve(dataset_A, config)
                formula_B = recursive_solve(dataset_B, config)
                return combine_formulas(formula_A, formula_B, split_type=simplifier.name)
                
            else:
                mutated_dataset = do_variable_collapse(dataset, result, gp_model, split_type=simplifier.name)
                return recursive_solve(mutated_dataset, config)
                
    return brute_force_symbolic_search(dataset)
