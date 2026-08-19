import pandas as pd
import numpy as np
import os
from get_pi_complex import PhysicalRegistry

def load_combined_teeth_dataset(
    data_dir: str = None, 
    mode: str = "wavelength"
) -> tuple[pd.DataFrame, PhysicalRegistry, str]:
    
    if data_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "..", "datasets")

    files_map = {
        4: "4_tooth_ar_br_dr_Fr_1.txt",
        6: "6_tooth_ar_br_dr_FR_1.txt",
        8: "8_tooth_ar_br_dr_FR_1.txt"
    }

    C0_VAL = 299.792458  # Скорость света в вакууме (мм * ГГц)
    ER_VAL = 4.5         # Диэлектрическая проницаемость подложки FR4
    VP_VAL = C0_VAL / np.sqrt(ER_VAL)  # Фазовая скорость (~141.32 мм * ГГц)

    dfs = []
    for n_teeth, filename in files_map.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            alt_path = os.path.join(data_dir, "..", filename)
            if os.path.exists(alt_path):
                filepath = alt_path
            else:
                print(f"[Предупреждение] Файл {filename} не найден по пути {filepath}. Пропуск.")
                continue

        df_sub = pd.read_csv(
            filepath, 
            sep=r'\s+', 
            header=None, 
            names=["aR", "bR", "dR", "f1"]
        )
        

        df_sub["n"] = float(n_teeth)
        df_sub["lambda_1"] = VP_VAL / df_sub["f1"]
        
        dfs.append(df_sub)

    if not dfs:
        raise FileNotFoundError(f"Не удалось найти ни одного файла датасета в {data_dir}!")

    df_combined = pd.concat(dfs, ignore_index=True)

    # Инициализация физического реестра [M, L, T]
    reg = PhysicalRegistry()
    reg.register("aR", [0, 1, 0])       # Длина [L] (мм)
    reg.register("bR", [0, 1, 0])       # Длина [L] (мм)
    reg.register("dR", [0, 1, 0])       # Длина [L] (мм)
    reg.register("n",  [0, 0, 0])       # Безразмерный счетчик [0, 0, 0]

    if mode == "wavelength":
        reg.register("lambda_1", [0, 1, 0])  # Длина [L] (мм)
        df_final = df_combined[["aR", "bR", "dR", "n", "lambda_1"]].copy()
        target_name = "lambda_1"
    else:
        reg.register("f1", [0, 0, -1])       # Частота [T^-1] (ГГц)
        df_final = df_combined[["aR", "bR", "dR", "n", "f1"]].copy()
        target_name = "f1"

    print("\n" + "="*60)
    print("=== ОБЪЕДИНЕННЫЙ ДАТАСЕТ (4, 6, 8 ЗУБЬЕВ) СФОРМИРОВАН ===")
    print(f"Всего строк: {len(df_final)}")
    print(f"Распределение по n:\n{df_final['n'].value_counts().sort_index()}")
    print("\nФизические размерности [M, L, T]:")
    for var_name, var_obj in reg.variables.items():
        print(f"  {var_name:10s} -> {var_obj.dimension}")
    print("="*60 + "\n")

    return df_final, reg, target_name

def load_4_tooth_thesis_invariants(filepath: str = None) -> tuple[pd.DataFrame, PhysicalRegistry, str]:
    if filepath is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, "..", "datasets", "4_tooth_ar_br_dr_Fr_1.txt")
    
    df_raw = pd.read_csv(filepath, sep=r'\s+', header=None, names=["aR", "bR", "dR", "f1"])

    C0_VAL = 299.792458
    ER_VAL = 4.5
    VP_VAL = C0_VAL / np.sqrt(ER_VAL)

    df = pd.DataFrame()
    
    df["ratio_ba"] = np.sqrt(df_raw["bR"])
    
    df["waist"] = df_raw["aR"] - df_raw["dR"]
    
    df["sum_ab"] = df_raw["dR"]
    
    df["lambda_1"] = VP_VAL / df_raw["f1"]

    reg = PhysicalRegistry()
    reg.register("ratio_ba", [0, 0.5, 0])  
    reg.register("waist",    [0, 1, 0])  # Длина [L]
    reg.register("sum_ab",   [0, 1, 0])  # Длина [L]
    reg.register("lambda_1", [0, 1, 0])  # Длина [L]

    print("=== Датасет с инвариантами диссертации загружен ===")
    print(f"Размер выборки: {len(df)} строк")
    print(f"Колонки: {list(df.columns)}")

    return df, reg, "lambda_1"


def load_combined_teeth_invariants(
    data_dir: str = None
) -> tuple[pd.DataFrame, PhysicalRegistry, str]:
    df_raw, _, _ = load_combined_teeth_dataset(data_dir, mode="wavelength")
    
    df = pd.DataFrame()
    df["ratio_ba"] = np.sqrt(df_raw["bR"])
    df["waist"] = df_raw["aR"] - df_raw["dR"]
    df["sum_ab"] = df_raw["n"] * df_raw["dR"]
    df["lambda_1"] = df_raw["lambda_1"]          

    reg = PhysicalRegistry()
    reg.register("ratio_ba", [0, 0.5, 0])  
    reg.register("waist",    [0, 1, 0])  # Длина [L]
    reg.register("sum_ab",   [0, 1, 0])  # Длина [L]
    reg.register("lambda_1", [0, 1, 0])  # Длина [L]

    return df, reg, "lambda_1"