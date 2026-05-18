import pandas as pd


import pandas as pd
import numpy as np


import pandas as pd


def create_target_variable(
    df: pd.DataFrame, absolute_limit_ug_m3: float
) -> pd.DataFrame:
    """
    Конвертація в приземну концентрацію та перевірка за жорстким державним лімітом (ГДК).
    За замовчуванням використовується середньодобова ГДК України: 40 мкг/м3.
    """
    df_target = df.copy()

    MOLAR_MASS_NO2 = 46.0055
    SURFACE_FACTOR = 2.0  # Фізичний коефіцієнт концентрації в приземному шарі

    # 1. Фізичний розрахунок (мкг/м3)
    df_target["NO2_ug_m3"] = (
        df_target["NO2_column_number_density"]
        * MOLAR_MASS_NO2
        * 1e6
        / df_target["boundary_layer_height"]
    ) * SURFACE_FACTOR

    # 2. Жорстка перевірка на перевищення об'єктивного ліміту
    df_target["target"] = (df_target["NO2_ug_m3"] > absolute_limit_ug_m3).astype(int)

    exceedances = df_target["target"].sum()
    print(
        f"   -> Знайдено {exceedances} днів із перевищенням ліміту {absolute_limit_ug_m3} мкг/м3"
    )

    return df_target


def add_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Додавання календаря та історичних зсувів."""
    df_feat = df.copy()

    df_feat["month"] = df_feat.index.month
    df_feat["day_of_week"] = df_feat.index.dayofweek
    df_feat["is_weekend"] = (df_feat["day_of_week"] >= 5).astype(int)

    # Лаги оригінальної щільності (якщо хочете, можна брати лаги NO2_ug_m3)
    df_feat["NO2_lag_1d"] = df_feat["NO2_column_number_density"].shift(1)
    df_feat["NO2_lag_2d"] = df_feat["NO2_column_number_density"].shift(2)
    df_feat["NO2_lag_7d"] = df_feat["NO2_column_number_density"].shift(7)

    df_feat["NO2_roll_mean_3d"] = (
        df_feat["NO2_column_number_density"].rolling(window=3).mean().shift(1)
    )

    df_feat = df_feat.dropna()
    return df_feat
