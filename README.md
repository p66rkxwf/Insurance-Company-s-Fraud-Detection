# Insurance-Company-s-Fraud-Detection

本專案使用機器學習方法（Logistic Regression、Random Forest）針對保險詐欺資料集進行模型建立與性能評估，包含前處理、特徵工程、模型訓練與最佳化。

---

## 📁 資料夾說明


---

## 🧪 模型與腳本說明

| 檔名 | 說明 |
|------|------|
| `logit_model.py` | 使用邏輯斯迴歸進行模型訓練與評估 |
| `rf_gridsearch.py` | 使用 GridSearchCV 對 Random Forest 進行超參數搜尋 |
| `rf_halving_basic.py` | 使用 Halving 技術加速 RF 搜尋（基本特徵） |
| `rf_halving_feature.py` | 使用 Halving 技術加上衍生特徵（例如：claim/income） |

---

## 🛠️ 依賴套件

請先安裝以下 Python 套件：

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn category_encoders statsmodels

# 進行資料切分與 logistic regression 模型
python scripts/logit_model.py

# 使用 grid search 訓練 RF 模型
python scripts/rf_gridsearch.py
