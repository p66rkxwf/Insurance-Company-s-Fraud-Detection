import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from scipy.stats import zscore
import statsmodels.api as sm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc as sk_auc,
    recall_score,
    precision_score,
    fbeta_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer

from imblearn.over_sampling import SMOTENC
from category_encoders import TargetEncoder

df = pd.read_csv('raw_data/train_2025.csv')
X = df.drop(columns=['claim_number', 'fraud'])  # 移除 ID 與目標欄位
y = df['fraud']

# ===  切分資料集：60% 訓練、20% 驗證、20% 測試 ===
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


# 以下特徵工程跟logit一樣，可省略不看
mask = ~X_train['zip_code'].astype(str).str.startswith('0')
X_train = X_train[mask].reset_index(drop=True)
y_train = y_train[mask].reset_index(drop=True)

cat_cols = [
    'gender', 'marital_status', 'high_education_ind', 'address_change_ind',
    'living_status', 'zip_code', 'claim_date', 'claim_day_of_week', 'accident_site',
    'witness_present_ind', 'channel', 'policy_report_filed_ind', 'vehicle_category', 'vehicle_color'
]
num_cols = [col for col in X_train.columns if col not in cat_cols]


# ===  自訂欄位轉換器 ===
def encode_gender(X):
    return (X == 'M').astype(int)

def encode_living_status(X):
    return (X == 'Own').astype(int)

class Zip3Extractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns) if hasattr(X, 'columns') else None
        return self
    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X.apply(lambda col: col.astype(str).str[:3])
    def get_feature_names_out(self, input_features=None):
        return [f"{feat}_zip3" for feat in (input_features or ['zip3'])]

class ExtractMonthYear(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns) if hasattr(X, 'columns') else None
        return self
    def transform(self, X):
        s = X.iloc[:, 0] if hasattr(X, 'iloc') else pd.Series(X[:, 0])
        date = pd.to_datetime(s, errors='coerce')
        return date.dt.to_period('M').astype(str).to_frame()
    def get_feature_names_out(self, input_features=None):
        return [f"{feat}_month_year" for feat in (input_features or ['claim_date'])]

class TargetEncoderWrapper(TargetEncoder):
    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else self.feature_names_in_

# ===  為每組欄位建立轉換流程（Pipeline）===
marital_witness_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

gender_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('label_encoder', FunctionTransformer(encode_gender, validate=False, feature_names_out='one-to-one'))
])

living_status_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('label_encoder', FunctionTransformer(encode_living_status, validate=False, feature_names_out='one-to-one'))
])

zip3_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('zip3_extract', Zip3Extractor()),
    ('target', TargetEncoderWrapper())
])

claim_date_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('extract_month_year', ExtractMonthYear()),
    ('target', TargetEncoderWrapper())
])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

target_cat_cols = ['claim_day_of_week', 'accident_site', 'channel', 'vehicle_category', 'vehicle_color']
target_cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('target', TargetEncoderWrapper())
])

# ===  整合所有欄位轉換 ===
transformers = [
    ('marital_witness', marital_witness_pipeline, ['marital_status', 'witness_present_ind']),
    ('gender', gender_pipeline, ['gender']),
    ('living_status', living_status_pipeline, ['living_status']),
    ('zip_code', zip3_pipeline, ['zip_code']),
    ('claim_date', claim_date_pipeline, ['claim_date']),
    ('num', num_pipeline, num_cols),
    ('cat_imputer', target_cat_pipeline, target_cat_cols)
]
preprocessor = ColumnTransformer(transformers, remainder='passthrough')


# 超參數設置
param_grid = {
    "n_estimators": [100, 300],          
    "max_depth": [None, 10],                 
    "min_samples_split": [2, 10],          
    "min_samples_leaf": [1, 4],               
    "max_features": ["sqrt"],
    'class_weight': [None, 'balanced']        
}

grid = ParameterGrid(param_grid)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



results = []
# 1. 計時開始
start_time = time.time()
for idx, param in enumerate(grid):
    aucs = []
    for tr_idx, val_idx in skf.split(X_train, y_train):
        X_tr_f, X_val_f = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr_f, y_val_f = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        z_scores = X_tr_f[num_cols].apply(zscore)
        mask_no_outliers = (z_scores.abs() < 3).all(axis=1)
        X_tr_f = X_tr_f.loc[mask_no_outliers].reset_index(drop=True)
        y_tr_f = y_tr_f.loc[mask_no_outliers].reset_index(drop=True)

        # 每一次迴圈都要重新 fit-transform 特徵工程
        X_tr_transformed = preprocessor.fit_transform(X_tr_f, y_tr_f)
        X_val_transformed = preprocessor.transform(X_val_f)

        feature_names = preprocessor.get_feature_names_out()
        categorical_feature_indices = [
            i for i, col in enumerate(feature_names)
            if col.startswith('gender__') 
            or col.startswith('living_status__') 
            or col.startswith('zip_code__')
            or col.startswith('claim_date__')
            or col.startswith('cat_imputer__') 
            or col.startswith('remainder__')
        ]
        smote_nc = SMOTENC(categorical_features=categorical_feature_indices, random_state=42)
        X_resampled, y_resampled = smote_nc.fit_resample(X_tr_transformed, y_tr_f)
        
        # 訓練模型
        model = RandomForestClassifier(**param, random_state=42)
        model.fit(X_resampled, y_resampled)

        # 驗證並計算 AUC
        y_pred_proba = model.predict_proba(X_val_transformed)[:, 1]
        auc = roc_auc_score(y_val_f, y_pred_proba)
        aucs.append(auc)
    
    mean_auc = np.mean(aucs)
    results.append((param, mean_auc))  
    print(f"Params {param} | Mean AUC: {np.mean(aucs):.4f}")
    
end_time = time.time()
print(f"執行時間: {end_time - start_time:.2f} 秒")
#  找出最佳組合
best_param, best_score = max(results, key=lambda x: x[1])
print("\n最佳超參數:", best_param)
print("平均 ROC-AUC:", best_score)

final_rf = RandomForestClassifier(random_state=42, n_jobs=-1, **best_param)
final_rf.fit(X_resampled, y_resampled)


X_valid_transformed = preprocessor.transform(X_valid)

y_proba = final_rf.predict_proba(X_valid_transformed)[:, 1]  # 取出為類別1的預測機率
fpr, tpr, thresholds = roc_curve(y_valid, y_proba)
roc_auc = sk_auc(fpr, tpr)

# 以 Recall 下限 尋找最佳閾值
target_recall = 0.8
best_threshold = 0
best_prec = 0
for threshold in thresholds:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    recall = recall_score(y_valid, y_pred_thresh)
    precision = precision_score(y_valid, y_pred_thresh)
    if recall >= target_recall and precision > best_prec:
        best_prec = precision
        best_threshold = threshold

print(f'最佳 Recall 下界時的閾值：{best_threshold:.4f}')



X_test_transformed = preprocessor.transform(X_test)


y_test_proba = final_rf.predict_proba(X_test_transformed)[:, 1]  # 取出為類別1的預測機率
y_test_pred = (y_test_proba >= best_threshold).astype(int)

# 計算測試集上的績效指標
print("測試集績效報告：")
print(classification_report(y_test, y_test_pred))
print()
print(f'auc: {roc_auc:.04f}')
print()
print(f'f2-score: {fbeta_score(y_test, y_test_pred, beta=2):.04f}')

# 從訓練好的隨機森林模型取得特徵重要性
importances = final_rf.feature_importances_
feature_names = preprocessor.get_feature_names_out()

# 建立 DataFrame 並排序（由高到低）
feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# 畫圖：橫向長條圖（從低到高畫起）
plt.figure(figsize=(10, len(feat_imp) * 0.3))  # 高度依照特徵數量自動調整
plt.barh(feat_imp['feature'][::-1], feat_imp['importance'][::-1], color='teal')  # 反轉順序，重要的在最上面
plt.xlabel("Feature Importance")
plt.title("All Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()

