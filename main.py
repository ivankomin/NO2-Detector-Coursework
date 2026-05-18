import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import initialize_gee, get_comprehensive_data
from src.features import create_target_variable, add_comprehensive_features
from src.models import train_and_evaluate


def main():
    print("=== ЗАПУСК МУЛЬТИМІСТОВОЇ СИСТЕМИ (PBLH КОНВЕРТАЦІЯ) ===")
    initialize_gee()

    cities_config = {
        "Zaporizhzhia": (35.139, 47.838),
        "Dnipro": (35.042, 48.465),
        "Kryvyi_Rih": (33.391, 47.910),
    }

    BUFFER_KM = 15
    START_DATE = "2020-01-01"
    END_DATE = "2025-12-31"

    all_datasets = []

    for city_name, coords in cities_config.items():
        print(f"\nОбробка міста: {city_name}")
        LON, LAT = coords
        cache_path = f"data/cache_{city_name.lower()}.csv"

        if os.path.exists(cache_path):
            df_city_raw = pd.read_csv(cache_path, index_col="date", parse_dates=True)
        else:
            df_city_raw = get_comprehensive_data(
                LON, LAT, BUFFER_KM, START_DATE, END_DATE
            )
            os.makedirs("data", exist_ok=True)
            df_city_raw.to_csv(cache_path)

        # Розрахунок приземної концентрації (мкг/м3) та створення таргета
        # Замість percentile_threshold=85.0
        df_city_target = create_target_variable(df_city_raw, absolute_limit_ug_m3=25.0)
        df_city_features = add_comprehensive_features(df_city_target)
        df_city_features["city"] = city_name

        all_datasets.append(df_city_features)

    # 1. Об'єднання
    df_final = pd.concat(all_datasets, axis=0).sort_index()
    # Відразу після об'єднання df_final = pd.concat(...)
    print(
        f"\nМаксимальне зафіксоване значення NO2: {df_final['NO2_ug_m3'].max():.2f} мкг/м3"
    )
    print(
        f"Середнє зафіксоване значення NO2: {df_final['NO2_ug_m3'].mean():.2f} мкг/м3"
    )

    total_days = len(df_final)
    exceeded_days = df_final["target"].sum()
    print(f"\nФінальний розмір матриці: {total_days} спостережень.")

    if exceeded_days == 0:
        print("ПОМИЛКА: Немає жодного перевищення! Знизьте ліміт для навчання моделі.")
        return

    # 2. Очищення від витоку даних (Data Leakage)
    # Ми видаляємо сирий NO2, розрахований NO2, таргет, хмарність і назву міста.
    # Зверніть увагу: ми ЗАЛИШАЄМО 'boundary_layer_height', бо це погодний фактор.
    leakage_cols = [
        "NO2_column_number_density",
        "NO2_ug_m3",
        "cloud_fraction",
        "target",
        "city",
    ]
    X = df_final.drop(columns=leakage_cols)
    y = df_final["target"]

    # 3. Розподіл даних та розрахунок ваги для XGBoost
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imbalance_ratio = (total_days - exceeded_days) / exceeded_days

    # 4. Навчання
    metrics = train_and_evaluate(X_train, X_test, y_train, y_test, use_smote=False, pos_weight=imbalance_ratio)

    # БЛОК 6 у файлі main.py
    print("\n" + "=" * 60)
    print("ПОРІВНЯЛЬНИЙ АНАЛІЗ МОДЕЛЕЙ (ФІЗИЧНА PBLH МОДЕЛЬ):")
    print("=" * 60)
    for name, data in metrics.items():
        print(f"\nМодель: {name}")
        print("-" * 30)
        print(data["full_report"])
        print("*" * 60)


if __name__ == "__main__":
    main()
