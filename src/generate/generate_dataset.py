import pandas as pd
import numpy as np
import os
from get_pi_complex import PhysicalRegistry

def load_4_tooth_ar_br_dr_Fr_1(filepath: str = None) -> tuple[pd.DataFrame, PhysicalRegistry, str]:
    if filepath is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, "..", "datasets", "4_tooth_ar_br_dr_Fr_1.txt")
    
    # aR (мм), bR (мм), dR (мм), f1 (ГГц)
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=["aR", "bR", "dR", "f1"])

    # Скорость света c0 в мм * ГГц  (так как 1 ГГц = 10^9 Гц, 1 мм = 10^-3 м => c0 = 299.792458 мм * ГГц)
    C0_VAL = 299.792458  
    ER_VAL = 4.5         # Относительная диэлектрическая проницаемость подложки FR4
    VP_VAL = C0_VAL / np.sqrt(ER_VAL)  # Фазовая скорость волны в диэлектрике (~141.323 мм * ГГц)

    df["c0"] = C0_VAL
    df["er"] = ER_VAL
    df["vp"] = VP_VAL

    reg = PhysicalRegistry()
    
    reg.register("aR", [0, 1, 0])
    reg.register("bR", [0, 1, 0])
    reg.register("dR", [0, 1, 0])
    
    reg.register("c0", [0, 1, -1])
    reg.register("vp", [0, 1, -1])
    
    reg.register("er", [0, 0, 0])

    
    reg.register("f1", [0, 0, -1])
    target_name = "f1"

    print(f"Размер выборки: {len(df)} строк")
    print(f"Колонки: {list(df.columns)}")
    print("\nРазмерности переменных [M, L, T]:")
    for var_name, var_obj in reg.variables.items():
        print(f"  {var_name:8s} -> {var_obj.dimension}")

    return df, reg, target_name

def generate_complex_gravity_relativity_dataset(n_samples: int = 400, noise_std: float = 0.0) -> tuple[pd.DataFrame, PhysicalRegistry, str]:
    """
    ТЕСТОВЫЙ ДАТАСЕТ 1: Гравитационно-релятивистская сила
    
    Формула: 
      force = (G * m1 * m2 / ((r1 - r2)**2 + h**2)) * (1 + (v / c)**2)
      
    Что тестирует:
    1. Сдвиговая симметрия (Translational): (r1 - r2)
    2. Симметрия сложения / Пифагора (Addition): (r1 - r2)**2 + h**2
    3. Масштабная симметрия (LargeScale): v / c
    4. Мультипликативная разделяемость (Multiplicative Separability): 
       F_grav(m1, m2, r1, r2, h) * F_rel(v, c)
    5. Физические размерности: [M=1, L=1, T=-2] (Сила)
    """
    np.random.seed(42)
    
    m1 = np.random.uniform(1.0, 10.0, n_samples)
    m2 = np.random.uniform(1.0, 10.0, n_samples)
    r1 = np.random.uniform(10.0, 50.0, n_samples)
    r2 = np.random.uniform(1.0, 9.0, n_samples)
    h  = np.random.uniform(2.0, 15.0, n_samples)
    v  = np.random.uniform(10.0, 100.0, n_samples)
    c  = np.random.uniform(300.0, 500.0, n_samples)
    G  = np.full(n_samples, 6.674) # Масштабированная гравитационная постоянная

    # Истинная физическая формула
    f_grav = G * m1 * m2 / ((r1 - r2)**2 + h**2)
    f_rel  = 1.0 + (v / c)**2
    y = f_grav * f_rel
    
    if noise_std > 0:
        y += np.random.normal(0, noise_std * np.std(y), n_samples)

    df = pd.DataFrame({
        "m1": m1, "m2": m2, "r1": r1, "r2": r2, "h": h,
        "v": v, "c": c, "G": G, "force": y
    })

    reg = PhysicalRegistry()
    # Размеры: [M, L, T]
    reg.register("m1", [1, 0, 0])
    reg.register("m2", [1, 0, 0])
    reg.register("r1", [0, 1, 0])
    reg.register("r2", [0, 1, 0])
    reg.register("h",  [0, 1, 0])
    reg.register("v",  [0, 1, -1])
    reg.register("c",  [0, 1, -1])
    reg.register("G",  [-1, 3, -2])
    reg.register("force", [1, 1, -2])

    target_name = "force"
    return df, reg, target_name

def generate_complex_thermo_wave_dataset(n_samples: int = 400, noise_std: float = 0.0) -> tuple[pd.DataFrame, PhysicalRegistry, str]:
    """
    ТЕСТОВЫЙ ДАТАСЕТ 2: Термодинамическая эффективная скорость волны
    
    Формула:
      velocity = sqrt( (kB * T / m) * log(1 + (p1 * p2) / p0**2) + v0**2 )
      
    Что тестирует:
    1. Обобщенная аддитивная разделяемость (General Additive): 
       velocity**2 = z_A(T, m, p1, p2, p0, kB) + z_B(v0)  (Преобразование квадрата y**2)
    2. Произведение симметрия (Multiply): p1 * p2
    3. Масштабная симметрия (LargeScale): (p1 * p2) / p0**2
    4. Композиционность (Compositionality): log(1 + ...)
    5. Физические размерности c 4 базисами: [M, L, T, Theta(Температура)]
    """
    np.random.seed(42)

    T  = np.random.uniform(100.0, 500.0, n_samples)
    m  = np.random.uniform(1.0, 5.0, n_samples)
    p1 = np.random.uniform(2.0, 10.0, n_samples)
    p2 = np.random.uniform(2.0, 10.0, n_samples)
    p0 = np.random.uniform(5.0, 20.0, n_samples)
    v0 = np.random.uniform(10.0, 50.0, n_samples)
    kB = np.full(n_samples, 1.38) # Постоянная Больцмана

    # Истинная физическая формула
    term_A = (kB * T / m) * np.log(1.0 + (p1 * p2) / (p0**2))
    term_B = v0**2
    y = np.sqrt(term_A + term_B)

    if noise_std > 0:
        y += np.random.normal(0, noise_std * np.std(y), n_samples)

    df = pd.DataFrame({
        "T": T, "m": m, "p1": p1, "p2": p2, "p0": p0,
        "v0": v0, "kB": kB, "velocity": y
    })

    reg = PhysicalRegistry()
    # Размеры: [M, L, T, Theta]
    reg.register("T",  [0, 0, 0, 1])
    reg.register("m",  [1, 0, 0, 0])
    reg.register("p1", [1, -1, -2, 0])
    reg.register("p2", [1, -1, -2, 0])
    reg.register("p0", [1, -1, -2, 0])
    reg.register("v0", [0, 1, -1, 0])
    reg.register("kB", [1, 2, -2, -1])
    reg.register("velocity", [0, 1, -1, 0])

    target_name = "velocity"
    return df, reg, target_name

def generate_test_1_trans_addsep(n_samples: int = 500, noise_std: float = 0.0) -> tuple[pd.DataFrame, PhysicalRegistry, str]:
    """
    Тест 1: Сдвиговая симметрия + Аддитивная сепарабельность
    Формула: y = (x1 - x2)^2 + x3 * x4
    Ожидаемые шаги: 
      1) TranslationalSymmetry: (x1 - x2) -> u
      2) AdditiveSeparability: y = f1(u) + f2(x3, x4)
      3) BRF: u^2 и x3 * x4
    """
    np.random.seed(42)
    x1 = np.random.uniform(5.0, 15.0, n_samples)
    x2 = np.random.uniform(1.0, 5.0, n_samples)
    x3 = np.random.uniform(1.0, 10.0, n_samples)
    x4 = np.random.uniform(1.0, 10.0, n_samples)

    y = (x1 - x2)**2 + x3 * x4
    if noise_std > 0:
        y += np.random.normal(0, noise_std * np.std(y), n_samples)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "target": y})
    reg = PhysicalRegistry()
    for col in ["x1", "x2", "x3", "x4", "target"]:
        reg.register(col, [0, 0, 0])

    return df, reg, "target"


def generate_test_2_scale_multsep(n_samples: int = 500, noise_std: float = 0.0) -> tuple[pd.DataFrame, PhysicalRegistry, str]:
    """
    Тест 2: Масштабная симметрия (отношение) + Мультипликативная сепарабельность
    Формула: y = (x1 / x2) * (x3 + x4)
    Ожидаемые шаги:
      1) LargeScaleSymmetry: (x1 / x2) -> v
      2) MultiplicativeSeparability: y = f1(v) * f2(x3, x4)
      3) BRF: v и (x3 + x4)
    """
    np.random.seed(42)
    x1 = np.random.uniform(10.0, 50.0, n_samples)
    x2 = np.random.uniform(2.0, 10.0, n_samples)
    x3 = np.random.uniform(1.0, 10.0, n_samples)
    x4 = np.random.uniform(1.0, 10.0, n_samples)

    y = (x1 / x2) * (x3 + x4)
    if noise_std > 0:
        y += np.random.normal(0, noise_std * np.std(y), n_samples)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "target": y})
    reg = PhysicalRegistry()
    for col in ["x1", "x2", "x3", "x4", "target"]:
        reg.register(col, [0, 0, 0])

    return df, reg, "target"


def generate_test_3_dim_addition(n_samples: int = 500, noise_std: float = 0.0) -> tuple[pd.DataFrame, PhysicalRegistry, str]:
    """
    Тест 3: Физическая размерность + Симметрия сложения
    Формула: force = (m * a) / (r1 + r2)
    Размерности: m [M=1], a [L=1, T=-2], r1, r2 [L=1], force [M=1, L=0, T=-2]
    Ожидаемые шаги:
      1) DimensionalAnalysisStep: выделяет якорь (m * a / r1) и Pi_1 = r2 / r1
      2) AdditionSymmetry: (r1 + r2)
      3) BRF: вычисляет 1 / (1 + Pi_1)
    """
    np.random.seed(42)
    m  = np.random.uniform(1.0, 10.0, n_samples)
    a  = np.random.uniform(1.0, 5.0, n_samples)
    r1 = np.random.uniform(2.0, 10.0, n_samples)
    r2 = np.random.uniform(1.0, 5.0, n_samples)

    force = (m * a) / (r1 + r2)
    if noise_std > 0:
        force += np.random.normal(0, noise_std * np.std(force), n_samples)

    df = pd.DataFrame({"m": m, "a": a, "r1": r1, "r2": r2, "force": force})

    reg = PhysicalRegistry()
    reg.register("m",  [1, 0, 0])   # Масса M
    reg.register("a",  [0, 1, -2])  # Ускорение L/T^2
    reg.register("r1", [0, 1, 0])   # Расстояние L
    reg.register("r2", [0, 1, 0])   # Расстояние L
    reg.register("force", [1, 0, -2]) # Сила на единицу длины M/T^2

    return df, reg, "force"


def generate_test_4_mul_trans(n_samples: int = 500, noise_std: float = 0.0) -> tuple[pd.DataFrame, PhysicalRegistry, str]:
    """
    Тест 4: Симметрия произведения + Сдвиг
    Формула: y = sin(x1 * x2) + (x3 - x4)^2
    Ожидаемые шаги:
      1) MultiplySymmetry: (x1 * x2) -> w
      2) TranslationalSymmetry: (x3 - x4) -> z
      3) BRF: sin(w) + z^2
    """
    np.random.seed(42)
    x1 = np.random.uniform(0.5, 3.0, n_samples)
    x2 = np.random.uniform(0.5, 3.0, n_samples)
    x3 = np.random.uniform(5.0, 15.0, n_samples)
    x4 = np.random.uniform(1.0, 5.0, n_samples)

    y = np.sin(x1 * x2) + (x3 - x4)**2
    if noise_std > 0:
        y += np.random.normal(0, noise_std * np.std(y), n_samples)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "target": y})
    reg = PhysicalRegistry()
    for col in ["x1", "x2", "x3", "x4", "target"]:
        reg.register(col, [0, 0, 0])

    return df, reg, "target"


def generate_test_5_genadd_scale(n_samples: int = 500, noise_std: float = 0.0) -> tuple[pd.DataFrame, PhysicalRegistry, str]:
    """
    Тест 5: Обобщенная аддитивность (Log) + Отношение
    Формула: y = exp((x1 - x2) + (x3 / x4))
    Ожидаемые шаги:
      1) GeneralAdditiveSeparability: определят g^-1(y) = ln(y)
      2) TranslationalSymmetry / LargeScaleSymmetry: сворачивают пары (x1 - x2) и (x3 / x4)
      3) BRF: решает комбинацию
    """
    np.random.seed(42)
    x1 = np.random.uniform(2.0, 5.0, n_samples)
    x2 = np.random.uniform(0.5, 2.0, n_samples)
    x3 = np.random.uniform(2.0, 10.0, n_samples)
    x4 = np.random.uniform(1.0, 4.0, n_samples)

    y = np.exp((x1 - x2) + (x3 / x4))
    if noise_std > 0:
        y += np.random.normal(0, noise_std * np.std(y), n_samples)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "target": y})
    reg = PhysicalRegistry()
    for col in ["x1", "x2", "x3", "x4", "target"]:
        reg.register(col, [0, 0, 0])

    return df, reg, "target"

def generate_test_6_gen_comp(n_samples: int = 500, noise_std: float = 0.0) -> tuple[pd.DataFrame, PhysicalRegistry, str]:
    """
    Тест 6: Сложная проверка Generalized Symmetry и Compositionality
    Формула: y = cos(x1 + x2) * (x3 - x4)^2 + exp(x1 + x2)
    Ожидаемые шаги:
      1) GeneralizedSymmetry / Compositionality: находят группу (x1, x2) -> h1 = x1 + x2
      2) GeneralizedSymmetry / Compositionality: находят группу (x3, x4) -> h2 = x3 - x4
      3) Финальный BF: сжимает cos(u) * v^2 + exp(u)
    """
    np.random.seed(42)
    x1 = np.random.uniform(0.5, 2.0, n_samples)
    x2 = np.random.uniform(0.5, 2.0, n_samples)
    x3 = np.random.uniform(3.0, 7.0, n_samples)
    x4 = np.random.uniform(0.5, 2.5, n_samples)

    u = x1 + x2
    v = x3 - x4
    y = np.cos(u) * (v**2) + np.exp(u)

    if noise_std > 0:
        y += np.random.normal(0, noise_std * np.std(y), n_samples)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "target": y})
    reg = PhysicalRegistry()
    for col in ["x1", "x2", "x3", "x4", "target"]:
        reg.register(col, [0, 0, 0])

    return df, reg, "target"

if __name__ == "__main__":
    df_raw, registry, target = load_4_tooth_ar_br_dr_Fr_1()
    print("\nПервые 5 строк датасета:")
    print(df_raw.head())