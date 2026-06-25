import pandas as pd
import numpy as np
import torch
import sympy as sp
from itertools import combinations

from function_analysis.recursive_apply_simplifications import recursive_solve
import warnings
from gpytorch.utils.warnings import NumericalWarning

warnings.filterwarnings("ignore", category=NumericalWarning)

class MockGPRegressionPipeline:
    """
    Эмулирует идеально обученный GPRegressionPipeline для функции:
    f = sin(x1 - x2) * exp(x3) + x4^2
    Это позволяет протестировать всю цепочку рекурсии и градиентов без обучения.
    """
    def __init__(self, feature_cols):
        self.feature_cols = [col for col in feature_cols if col != 'target']

    def predict(self, x):
        # x имеет размерность (batch_size, n_features)
        # Нам нужно понять, какие переменные сейчас активны в датасете
        vals = {}
        for i, col in enumerate(self.feature_cols):
            if col == 'x1': vals['x1'] = x[:, i]
            elif col == 'x2': vals['x2'] = x[:, i]
            elif col == 'x3': vals['x3'] = x[:, i]
            elif col == 'x4': vals['x4'] = x[:, i]
            elif col == '(x1_minus_x2)': vals['U'] = x[:, i]

        # Вычисляем истинное значение в зависимости от того, какие колонки остались
        if 'x1' in vals and 'x2' in vals and 'x3' in vals and 'x4' in vals:
            # Исходное состояние (4D)
            y = torch.sin(vals['x1'] - vals['x2']) * torch.exp(vals['x3']) + vals['x4']**2
        elif 'U' in vals and 'x3' in vals:
            # После сворачивания x1 и x2 (2D)
            y = torch.sin(vals['U']) * torch.exp(vals['x3'])
        elif 'x1' in vals and 'x2' in vals and 'x3' in vals:
            # Ветка А после аддитивного разделения (3D)
            y = torch.sin(vals['x1'] - vals['x2']) * torch.exp(vals['x3'])
        elif 'U' in vals:
            # Ветка А1 после мультипликативного разделения (1D)
            y = torch.sin(vals['U'])
        elif 'x3' in vals:
            # Ветка А2 после мультипликативного разделения (1D)
            y = torch.exp(vals['x3'])
        elif 'x4' in vals:
            # Ветка Б после аддитивного разделения (1D)
            y = vals['x4']**2
        else:
            raise ValueError(f"Неизвестная комбинация колонок: {self.feature_cols}")

        class Output:
            pass
        out = Output()
        out.mean = y
        return out

# Подменяем функцию обучения на возврат нашей идеальной Mock-модели для теста
def train_gp(dataset: pd.DataFrame) -> MockGPRegressionPipeline:
    feature_cols = [col for col in dataset.columns if col != 'target']
    return MockGPRegressionPipeline(feature_cols)

# =====================================================================
# ЗАПУСК ИНТЕГРАЦИОННОГО ТЕСТА
# =====================================================================
if __name__ == "__main__":
    print("=== ГЕНЕРАЦИЯ СИНТЕТИЧЕСКОГО ДАТАСЕТА ===")
    np.random.seed(42)
    n_samples = 500
    

    # Генерируем случайные входы для 4 переменных
    x1 = np.random.uniform(-1, 1, n_samples)
    x2 = np.random.uniform(-1, 1, n_samples)
    x3 = np.random.uniform(-1, 1, n_samples)
    x4 = np.random.uniform(-1, 1, n_samples)
    
    # Вычисляем истинный таргет
    y = np.sin(x1 - x2) * np.exp(x3) + x4**2
    
    # Собираем DataFrame
    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'target': y
    })
    
    print(f"Форма исходного датасета: {df.shape}")
    print(f"Колонки: {df.columns.tolist()}")
    
    print("\n=== ЗАПУСК РЕКУРСИВНОГО РЕШАТЕЛЯ ===")
    final_formula = recursive_solve(df)
    
    print("\n=== ОКОНЧАТЕЛЬНЫЙ РЕЗУЛЬТАТ ===")
    print(f"Итоговая восстановленная структура формулы:")
    sp.pprint(final_formula)