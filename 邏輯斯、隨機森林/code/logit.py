import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, auc
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from imblearn.over_sampling import SMOTENC
from category_encoders import TargetEncoder
import statsmodels.api as sm
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import zscore
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    recall_score,
    precision_score,
    fbeta_score
)

# ===  載入資料與分離特徵與目標欄位 ===
df = pd.read_csv('kaggle data拷貝/train_2025.csv')


X = df.drop(columns=['claim_number', 'fraud'])  # 移除 ID 與目標欄位
y = df['fraud']

# ===  切分資料集：60% 訓練、20% 驗證、20% 測試 ===
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# === 輸出訓練集、驗證集、測試集 ===
X_train.to_csv('kaggle data拷貝/split/X_train.csv')
y_train.to_csv('kaggle data拷貝/split/y_train.csv')
X_valid.to_csv('kaggle data拷貝/split/X_valid.csv')
y_valid.to_csv('kaggle data拷貝/split/y_valid.csv')
X_test.to_csv('kaggle data拷貝/split/X_test.csv')
y_test.to_csv('kaggle data拷貝/split/y_test.csv')


# 特徵工程
# ===  在測試集移除 zip_code 開頭為 0 的資料 ===
mask = ~X_train['zip_code'].astype(str).str.startswith('0')
X_train = X_train[mask].reset_index(drop=True)
y_train = y_train[mask].reset_index(drop=True)

# ===  指定類別與數值欄位 ===
cat_cols = [
    'gender', 'marital_status', 'high_education_ind', 'address_change_ind',
    'living_status', 'zip_code', 'claim_date', 'claim_day_of_week', 'accident_site',
    'witness_present_ind', 'channel', 'policy_report_filed_ind', 'vehicle_category', 'vehicle_color'
]
num_cols = [col for col in X_train.columns if col not in cat_cols]

# ===  移除數值欄位離群值（z-score > 3）===
z_scores = X_train[num_cols].apply(zscore)
mask_no_outliers = (z_scores.abs() < 3).all(axis=1)
X_train = X_train.loc[mask_no_outliers].reset_index(drop=True)
y_train = y_train.loc[mask_no_outliers].reset_index(drop=True)

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
    

class FrequencyEncoderWrapper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.Series(X.ravel(), name="feature")
        freq = X.value_counts(normalize=True)
        self.freq_map_ = freq.to_dict()
        return self

    def transform(self, X):
        X = pd.Series(X.ravel(), name="feature")
        return X.map(self.freq_map_).fillna(0).values.reshape(-1, 1)

# === 為每組欄位建立轉換流程（Pipeline）===
# === 要處理的欄位有 'gender'、'living_status'、'zip_code'、'claim_date'、'claim_day_of_week'、
# ===              'accident_site'、'channel''vehicle_category'、'vehicle_color'、數值型欄位、缺失值欄位

# 僅補值（所有變數只有這兩欄有缺值，不過比例很低）
marital_witness_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# 對'gender'做lable encoding
gender_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('label_encoder', FunctionTransformer(encode_gender, validate=False, feature_names_out='one-to-one'))
])

# 對'living_status'做lable encoding
living_status_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('label_encoder', FunctionTransformer(encode_living_status, validate=False, feature_names_out='one-to-one'))
])

# 對'zip_code'做取前三值、onehot encoding
zip3_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('zip3_extract', Zip3Extractor()),
    ('target', TargetEncoderWrapper())
    ])

# 對'claim_date'取年月、target encoding
claim_date_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('extract_month_year', ExtractMonthYear()),
    ('target', TargetEncoderWrapper())
])

# 對數值行欄位做標準化
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

# 對剩下要處理的欄位做target encoding
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

# 建立欄位轉換器，之後的特徵工程都依賴preprocessor
preprocessor = ColumnTransformer(transformers, remainder='passthrough') # passthrough用於feature name

# ===  對訓練與驗證集做轉換並建立欄位名稱 ===
X_train_final_array = preprocessor.fit_transform(X_train, y_train)
X_valid_final_array = preprocessor.transform(X_valid)

# 儲存所有欄位（包括未處理的）
try:
    feature_names = preprocessor.get_feature_names_out()
except Exception as e:
    print("⚠️ 無法取得特徵名稱，原因：", e)
    feature_names = [f"feature_{i}" for i in range(X_train_final_array.shape[1])]

X_train = pd.DataFrame(X_train_final_array, columns=feature_names, index=X_train.index)
X_valid_final = pd.DataFrame(X_valid_final_array, columns=feature_names, index=X_valid.index)


# ===  使用 SMOTENC 對不平衡資料做過採樣處理 ===
# 判斷哪些欄位是類別欄位
categorical_feature_indices = [
    i for i, col in enumerate(X_train.columns)
    if col.startswith('gender__') 
    or col.startswith('living_status__') 
    or col.startswith('zip_code__')
    or col.startswith('claim_date__')
    or col.startswith('cat_imputer__') 
    or col.startswith('remainder__')
]

smote_nc = SMOTENC(categorical_features=categorical_feature_indices, random_state=42)
X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)

# ===  使用 statsmodels 建立邏輯斯迴歸模型並印出 summary（）===
X_with_const = sm.add_constant(X_resampled)
logit_model = sm.Logit(y_resampled, X_with_const).fit()
print(logit_model.summary())

# ===  使用 sklearn 的 LogisticRegression 訓練模型，做為實際預測用 ===
clf = LogisticRegression(max_iter=100, solver='liblinear')

clf.fit(X_resampled, y_resampled)

# ===  預測測試集結果並顯示績效 ===
X_test_final = preprocessor.transform(X_test)
y_pred_test = clf.predict(X_test_final)

print("閾值0.5的測試集績效報告：")
print(classification_report(y_test, y_pred_test))


cm = confusion_matrix(y_test, y_pred_test)
print("閾值0.5的測試集混淆矩陣：")
print("       Pred 0    Pred 1")
print(f"True 0   {cm[0,0]:>5}      {cm[0,1]:>5}")
print(f"True 1   {cm[1,0]:>5}      {cm[1,1]:>5}")

# === 驗證集 ROC 曲線與最佳閾值選擇 ===

# 1. 計算預測機率（類別為 1）
y_valid_proba = clf.predict_proba(X_valid_final)[:, 1]

# 2. 計算 ROC 曲線與 AUC 值
fpr, tpr, thresholds = roc_curve(y_valid, y_valid_proba)
roc_auc = auc(fpr, tpr)

# 3. 畫出驗證集的 ROC 曲線
#plt.figure()
#plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
#plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate (Recall)')
#plt.title('ROC Curve on Validation Set')
#plt.legend(loc='lower right')
#plt.grid()
#plt.show()

# 4. 以 recall 下限 = 0.8，尋找最佳 precision 對應的閾值
target_recall = 0.8
best_threshold = 0
best_prec = 0

for threshold in thresholds:
    y_pred_thresh = (y_valid_proba >= threshold).astype(int)
    recall = recall_score(y_valid, y_pred_thresh)
    precision = precision_score(y_valid, y_pred_thresh)
    if recall >= target_recall and precision > best_prec:
        best_prec = precision
        best_threshold = threshold

print(f'最佳閾值（使 recall ≥ 0.8 時 precision 最大）：{best_threshold:.4f}')



# === 測試集 ROC、AUC、閾值應用與評估 ===

# 1. 測試集預測機率（類別為 1）
y_test_proba = clf.predict_proba(X_test_final)[:, 1]

# 2. 計算 ROC 曲線與 AUC 值
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

# 3. 畫出測試集 ROC 曲線
#plt.figure()
#plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
#plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate (Recall)')
#plt.title('ROC Curve on Test Set')
#plt.legend(loc='lower right')
#plt.grid()
#plt.show()

# 4. 使用驗證集上選出的最佳閾值做預測
y_test_pred = (y_test_proba >= best_threshold).astype(int)

# 5. 顯示最佳閾值下的測試集績效評估報告
print("最佳閾值的測試集績效報告：")
print(classification_report(y_test, y_test_pred))
print(f'AUC: {roc_auc:.04f}')
print(f'F2-score: {fbeta_score(y_test, y_test_pred, beta=2):.04f}')

cm = confusion_matrix(y_test, y_test_pred )
print("最佳閾值的測試集混淆矩陣：")
print("       Pred 0    Pred 1")
print(f"True 0   {cm[0,0]:>5}      {cm[0,1]:>5}")
print(f"True 1   {cm[1,0]:>5}      {cm[1,1]:>5}")


