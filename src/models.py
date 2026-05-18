from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    f1_score,
    precision_recall_curve,
    auc,
    roc_auc_score
)
# Додано імпорт SMOTE
from imblearn.over_sampling import SMOTE

def get_models_config(random_state: int = 42, pos_weight: float = 5.0, use_smote: bool = False):
    """
    Якщо використовується SMOTE, класи стають збалансованими штучно, 
    тому ми відключаємо внутрішнє алгоритмічне зважування (class_weight/scale_pos_weight),
    щоб уникнути перенавчання на міноритарний клас.
    """
    if use_smote:
        return {
            "Logistic_Regression": LogisticRegression(random_state=random_state, max_iter=1000),
            "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=random_state, eval_metric='logloss'),
            "SVM_RBF": SVC(kernel='rbf', probability=True, random_state=random_state)
        }
    else:
        return {
            "Logistic_Regression": LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced'),
            "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced'),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=random_state, eval_metric='logloss', scale_pos_weight=pos_weight),
            "SVM_RBF": SVC(kernel='rbf', probability=True, random_state=random_state, class_weight='balanced')
        }

def train_and_evaluate(X_train, X_test, y_train, y_test, random_state: int = 42, pos_weight: float = 5.0, use_smote: bool = False):
    models = get_models_config(random_state, pos_weight, use_smote)
    results = {}
    
    # Масштабування даних
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Блок SMOTE: застосовується ТІЛЬКИ до тренувальної вибірки
    if use_smote:
        print("\n--- Застосування SMOTE до тренувальної вибірки ---")
        smote = SMOTE(random_state=random_state)
        X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
        
        # Вивід для контролю, щоб бачити, як змінився баланс
        anomalies_before = (y_train == 1).sum()
        anomalies_after = (y_train_final == 1).sum()
        print(f"Клас 1 (Аномалії) ДО SMOTE: {anomalies_before}")
        print(f"Клас 1 (Аномалії) ПІСЛЯ SMOTE: {anomalies_after}")
    else:
        # Якщо SMOTE вимкнено, використовуємо оригінальні дані
        X_train_final, y_train_final = X_train_scaled, y_train
    
    print("\n--- Тренування моделей та розрахунок метрик ---")
    for name, model in models.items():
        # 1. Навчання моделі
        model.fit(X_train_final, y_train_final)
        
        # 2. Отримання прогнозів класів (0 або 1) та ймовірностей (0.0 - 1.0)
        # ПРОГНОЗ РОБИТЬСЯ НА ОРИГІНАЛЬНИХ ТЕСТОВИХ ДАНИХ (X_test_scaled)
        preds = model.predict(X_test_scaled)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            probs = preds
            
        # 3. Розрахунок PR-AUC та ROC-AUC
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(y_test, probs)
        
        # 4. Базовий звіт scikit-learn
        base_report = classification_report(y_test, preds, zero_division=0)
        
        # 5. СТВОРЕННЯ РОЗШИРЕНОГО ЗВІТУ (Склеюємо все в один текст)
        custom_full_report = (
            f"ROC-AUC  : {roc_auc:.4f}\n"
            f"PR-AUC   : {pr_auc:.4f}\n"
            f"{'-'*53}\n"
            f"{base_report}"
        )
        
        # Зберігаємо наш новий монолітний текст
        results[name] = {
            "full_report": custom_full_report,
            "pr_auc": pr_auc,  # Збережено на випадок, якщо знадобиться для графіків
            "roc_auc": roc_auc
        }
        
    return results