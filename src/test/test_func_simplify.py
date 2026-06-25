import pytest
import pandas as pd
import torch
import numpy as np
import sympy as sp

from function_analysis.simplify import (AdditiveSeparabilitySimplifier,
                                        MultiplicationSeparabilitySimplifier,
                                        GeneralAdditiveSeparabilitySimplifier,
                                        TranslationalSymmetrySimplifier,
                                        LargeScaleSymmetrySimplifier,
                                        MultiplySymmetrySimplifier,
                                        GeneralizedSymmetrySimplifier,
                                        CompositionalitySimplifier)

class DummyGPModel:
    """
    Фиктивная модель, которая оборачивает обычную функцию PyTorch,
    имитируя интерфейс реальной GP-модели.
    """
    def __init__(self, pytorch_func):
        self.pytorch_func = pytorch_func

    def eval(self):
        pass

    def __call__(self, x):
        class GPResult:
            def __init__(self, mean):
                self.mean = mean
        
        return GPResult(self.pytorch_func(x))
    
def test_additive_separability_success():
    def func(x):
        return torch.sin(x[:, 0]) + x[:, 1] + x[:, 2]**2
    
    gp_model = DummyGPModel(func)

    np.random.seed(42)
    data = np.random.uniform(1, 5, size=(100, 3))
    df = pd.DataFrame(data, columns=['x1', 'x2', 'x3'])

    simplifier = AdditiveSeparabilitySimplifier(threshold=1e-4, num_test_points=10)

    success, groups = simplifier.try_simplify(gp_model, df)

    print("--------------------------------------------------")
    print(f"Результаты {simplifier.name}:{success}, {groups}")
    assert success is True
    assert len(groups) == 3

def test_additive_separability_fail():
    def func(x):
        return x[:, 0] * x[:, 1]

    gp_model = DummyGPModel(func)

    np.random.seed(42)
    data = np.random.uniform(1, 5, size=(100, 2))
    df = pd.DataFrame(data, columns=['x1', 'x2'])

    simplifier = AdditiveSeparabilitySimplifier(threshold=1e-4, num_test_points=10)
    success, groups = simplifier.try_simplify(gp_model, df)

    print("--------------------------------------------------")
    print(f"Результаты {simplifier.name}:{success}, {groups}")
    assert success is False

def test_multiplication_separability_success():
    def func(x):
        return torch.sin(x[:, 0]) * torch.cos(x[:, 1])

    gp_model = DummyGPModel(func)

    np.random.seed(42)
    data = np.random.uniform(1, 5, size=(100, 2))
    df = pd.DataFrame(data, columns=['x1', 'x2'])

    simplifier = MultiplicationSeparabilitySimplifier(threshold=1e-4, num_test_points=10)

    success, groups = simplifier.try_simplify(gp_model, df)

    print("--------------------------------------------------")
    print(f"Результаты {simplifier.name}:{success}, {groups}")
    assert success is True
    assert len(groups) == 2

def test_multiplication_separability_fail():
    def func(x):
        return torch.sin(x[:, 0]) * torch.cos(x[:, 1]) * torch.exp(x[:, 2]) + torch.exp(x[:, 3])

    gp_model = DummyGPModel(func)

    np.random.seed(42)
    data = np.random.uniform(1, 5, size=(100, 4))
    df = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'x4'])

    simplifier = MultiplicationSeparabilitySimplifier(threshold=1e-4, num_test_points=10)

    success, groups = simplifier.try_simplify(gp_model, df)

    print("--------------------------------------------------")
    print(f"Результаты {simplifier.name}:{success}, {groups}")
    assert success is False

def test_general_additive_separability_success():
    def func(x):
        return torch.sin(x[:, 0]**2 + x[:, 1])

    gp_model = DummyGPModel(func)

    np.random.seed(42)
    data = np.random.uniform(1, 5, size=(100, 2))
    df = pd.DataFrame(data, columns=['x1', 'x2'])

    simplifier = GeneralAdditiveSeparabilitySimplifier(threshold=1e-3, num_test_points=10)
    success, groups = simplifier.try_simplify(gp_model, df)

    print("--------------------------------------------------")
    print(f"Результаты {simplifier.name}:{success}, {groups}")

    assert success is True
    assert groups == [['x1'], ['x2']]

def test_general_additive_separability_fail():
    def func(x):
        return torch.log(x[:, 0]**2 * x[:, 1] + x[:, 0] + 1e-5)

    gp_model = DummyGPModel(func)

    np.random.seed(42)
    data = np.random.uniform(1, 5, size=(100, 2))
    df = pd.DataFrame(data, columns=['x1', 'x2'])

    simplifier = GeneralAdditiveSeparabilitySimplifier(threshold=1e-3, num_test_points=10)
    success, groups = simplifier.try_simplify(gp_model, df)

    print("--------------------------------------------------")
    print(f"Результаты {simplifier.name}:{success}, {groups}")
    
    assert success is False

def test_translational_symmetry():
    # f(x1, x2, x3) = sin(x1 - x2) + x3
    def func(x):
        return torch.sin(x[:, 0] - x[:, 1]) + x[:, 2]

    gp_model = DummyGPModel(func)

    np.random.seed(42)
    data = np.random.uniform(1, 5, size=(100, 3))
    df = pd.DataFrame(data, columns=['x1', 'x2', 'x3'])

    simplifier = TranslationalSymmetrySimplifier(threshold=1e-2, num_test_points=15)
    success, groups = simplifier.try_simplify(gp_model, df)

    print(f"\nРезультат {simplifier.name}: {success}, {groups}")
    assert success is True
    assert ['x1', 'x2'] in groups

def test_large_scale_symmetry():
    # f(x1, x2, x3) = exp(x1 / x2) + x3^2
    def func(x):
        return torch.exp(x[:, 0] / x[:, 1]) + x[:, 2]**2

    gp_model = DummyGPModel(func)

    np.random.seed(42)
    # Важно: генерируем строго положительные числа (>0), чтобы избежать деления на ноль
    data = np.random.uniform(1, 5, size=(100, 3))
    df = pd.DataFrame(data, columns=['x1', 'x2', 'x3'])

    simplifier = LargeScaleSymmetrySimplifier(threshold=1e-2, num_test_points=15)
    success, groups = simplifier.try_simplify(gp_model, df)

    print(f"Результат {simplifier.name}: {success}, {groups}")
    assert success is True
    assert ['x1', 'x2'] in groups


def test_multiply_symmetry():
    # f(x1, x2, x3) = cos(x1 * x2) + x3
    def func(x):
        return torch.cos(x[:, 0] * x[:, 1]) + x[:, 2]

    gp_model = DummyGPModel(func)

    np.random.seed(42)
    data = np.random.uniform(1, 5, size=(100, 3))
    df = pd.DataFrame(data, columns=['x1', 'x2', 'x3'])

    simplifier = MultiplySymmetrySimplifier(threshold=1e-2, num_test_points=15)
    success, groups = simplifier.try_simplify(gp_model, df)

    print(f"Результат {simplifier.name}: {success}, {groups}")
    assert success is True
    assert ['x1', 'x2'] in groups

def test_generalized_symmetry_success():
    # f(x1, x2, x3) = (x1^2 + x2^2) * x3
    # Симметрия присутствует для группы [x1, x2]
    def func(x):
        return (x[:, 0]**2 + x[:, 1]**2) * x[:, 2]

    gp_model = DummyGPModel(func)

    np.random.seed(42)
    data = np.random.uniform(1, 5, size=(100, 3))
    df = pd.DataFrame(data, columns=['x1', 'x2', 'x3'])

    simplifier = GeneralizedSymmetrySimplifier(
        threshold=1e-2, 
        num_test_points=5, 
        num_samples_m=50, 
        max_group_size=2
    )
    
    success, groups = simplifier.try_simplify(gp_model, df)

    print(f"\nРезультат Generalized Symmetry (Success): {success}, {groups}")
    
    assert success is True
    assert ('x1' in groups) and ('x2' in groups)


def test_generalized_symmetry_fail():
    # x1^x2 + x2^x3
    def func(x):
        x1 = torch.clamp(x[:, 0], min=1.1)
        x2 = torch.clamp(x[:, 1], min=1.1)
        x3 = torch.clamp(x[:, 2], min=1.1)
        return torch.pow(x1, x2) + torch.pow(x2, x3)

    gp_model = DummyGPModel(func)

    np.random.seed(42)
    data = np.random.uniform(1.5, 4.0, size=(100, 3))
    df = pd.DataFrame(data, columns=['x1', 'x2', 'x3'])

    simplifier = GeneralizedSymmetrySimplifier(
        threshold=1e-2, 
        num_test_points=5, 
        num_samples_m=50, 
        max_group_size=2
    )
    
    success, groups = simplifier.try_simplify(gp_model, df)

    print(f"Результат Generalized Symmetry (Fail): {success}, {groups}")
    
    assert success is False

def test_compositionality_success():
    # f(x1, x2) = exp(x1^2 + x2^2)
    def func(x):
        return torch.exp(x[:, 0]**2 + x[:, 1]**2)

    gp_model = DummyGPModel(func)

    np.random.seed(42)
    data = np.random.uniform(1, 3, size=(100, 2))
    df = pd.DataFrame(data, columns=['x1', 'x2'])

    simplifier = CompositionalitySimplifier(threshold=1e-2, num_test_points=15)
    success, expr = simplifier.try_simplify(gp_model, df)

    print(f"\nРезультат Compositionality (Success): {success}, {expr}")

    assert success is True
    x1, x2 = sp.symbols('x1 x2')
    expected_expr = x1**2 + x2**2
    assert sp.simplify(expr - expected_expr) == 0


def test_compositionality_fail():
    # f(x1, x2) = sin(x1) + exp(x2)
    # Эту функцию нельзя представить в виде F(h(x1, x2)) для кандидатных h
    def func(x):
        return torch.sin(x[:, 0]) + torch.exp(x[:, 1])

    gp_model = DummyGPModel(func)

    np.random.seed(42)
    data = np.random.uniform(1, 3, size=(100, 2))
    df = pd.DataFrame(data, columns=['x1', 'x2'])

    simplifier = CompositionalitySimplifier(threshold=1e-2, num_test_points=15)
    success, expr = simplifier.try_simplify(gp_model, df)

    print(f"Результат Compositionality (Fail): {success}, {expr}")

    assert success is False
    assert expr is None

test_additive_separability_success()
test_additive_separability_fail()
test_multiplication_separability_success()
test_multiplication_separability_fail()
test_general_additive_separability_success()
test_general_additive_separability_fail()
test_translational_symmetry()
test_large_scale_symmetry()
test_multiply_symmetry()
test_generalized_symmetry_success()
test_generalized_symmetry_fail()
test_compositionality_success()
test_compositionality_fail()
