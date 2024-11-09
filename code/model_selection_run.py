import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from data_preprocessing import DataPreprocessor
from model_evaluation import evaluate_model
from stacking_ensemble import EnsembleStacker
from plot_performance import plot_stacked_model_performance

# 전처리 및 데이터 준비
preprocessor = DataPreprocessor('./data/경진대회용 주조 공정최적화 데이터셋.csv')
preprocessor.encode_working()
preprocessor.preprocess_datetime()
preprocessor.remove_unnecessary_columns()
preprocessor.remove_missing_values()
preprocessor.remove_outliers()
preprocessor.perform_feature_selection()
data = preprocessor.get_processed_data()

X = data.drop('passorfail', axis=1)
y = data['passorfail']

# 데이터 분할 및 스케일링
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 목록 정의
factory_deterministic_models = {
    "Logistic Regression": LogisticRegression(),
    "GaussianNB": GaussianNB()
}

factory_complex_models = {
    "Extra Trees": ExtraTreesClassifier(),
    "KNN": KNeighborsClassifier()
}

command_deterministic_models = {
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

command_complex_models = {
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "MLP": MLPClassifier()
}

# 개별 모델 평가 및 선택
results = []
for model_name, model in factory_deterministic_models.items():
    metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    results.append({"Model": model_name, **metrics})

for model_name, model in factory_complex_models.items():
    metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    results.append({"Model": model_name, **metrics})

for model_name, model in command_deterministic_models.items():
    metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    results.append({"Model": model_name, **metrics})

for model_name, model in command_complex_models.items():
    metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    results.append({"Model": model_name, **metrics})

# 개별 모델 평가 결과 출력
results_df = pd.DataFrame(results)
print("개별 모델 성능:")
print(results_df)

# 스태킹 앙상블 실행
stacking_estimators = [
    ("Logistic Regression", LogisticRegression()),
    ("Extra Trees", ExtraTreesClassifier()),
    ("Random Forest", RandomForestClassifier()),
    ("LightGBM", LGBMClassifier())
]

stacker = EnsembleStacker(stacking_estimators, X_train_scaled, y_train, X_test_scaled, y_test)
stacker.fit()
stacking_metrics = stacker.evaluate()
print("\n스태킹 앙상블 성능:")
print(stacking_metrics)

# 시각화
plot_stacked_model_performance(pd.DataFrame([stacking_metrics]))
