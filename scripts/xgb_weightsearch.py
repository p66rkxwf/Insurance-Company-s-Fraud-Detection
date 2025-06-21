# xgb_weightsearch.py
# -------------------------------------------------------------
# • 權重網格 {5,7,9} → 根據成本挑選最優
# • 結果檔案／圖片統一前綴為 “XGB_”
# -------------------------------------------------------------
import os, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, precision_recall_curve, auc)
from scipy.stats import zscore
from imblearn.over_sampling import SMOTENC
from category_encoders.target_encoder import TargetEncoder
import xgboost as xgb, shap

# -------- 全域設定 --------
warnings.filterwarnings("ignore", category=FutureWarning)
FP_COST, FN_COST = 150, 10000          # 誤判為正 & 漏判為負的成本設定
WEIGHTS = [5, 7, 9]                    # 權重網格
out_dir = os.path.dirname(__file__)    # 當前腳本所在資料夾
PFX = "XGB_"                           # 統一檔名前綴

# -------- 中文欄位對應（可選）--------
map_path = os.path.join(out_dir, "column_map.csv")
name_map = (pd.read_csv(map_path).set_index("raw_name")["nice_name"].to_dict()
            if os.path.exists(map_path) else {})

# -------- 1. 讀取與清洗資料 --------
df = pd.read_csv("train_2025.csv").rename(str.strip, axis=1)

# 移除不必要欄位
df.drop(columns=[c for c in ("claim_number", "claim_id") if c in df.columns],
        errors="ignore", inplace=True)
df.drop(columns=[c for c in df.columns
                 if c.startswith(("vehicle_color", "claim_day_of_week"))],
        errors="ignore", inplace=True)

# 新增衍生欄位
df["claim_date"] = pd.to_datetime(df["claim_date"])
df["payout_income_ratio"] = df["claim_est_payout"] / (df["annual_income"] + 1)
df["past_claim_rate"] = df["past_num_of_claims"] / (df["age_of_driver"] + 1)
if "age_of_vehicle" not in df.columns and "vehicle_make_year" in df.columns:
    df["age_of_vehicle"] = df["claim_date"].dt.year - \
        pd.to_datetime(df["vehicle_make_year"], errors="coerce").dt.year.fillna(df["claim_date"].dt.year)
df["claim_cnt_12m"] = df["past_num_of_claims"].clip(0, 12)

# -------- 2. 指定特徵欄位 --------
num_cols = ["age_of_driver","safty_rating","annual_income","past_num_of_claims",
            "liab_prct","claim_est_payout","vehicle_price","vehicle_weight",
            "age_of_vehicle","payout_income_ratio","past_claim_rate","claim_cnt_12m"]
cat_cols = ["gender","marital_status","high_education_ind","address_change_ind",
            "living_status","zip_code","accident_site","witness_present_ind",
            "channel","policy_report_filed_ind","vehicle_category"]

# 篩選存在欄位
num_cols = [c for c in num_cols if c in df.columns]
cat_cols = [c for c in cat_cols if c in df.columns]
df[cat_cols] = df[cat_cols].astype("category")
X, y = df[num_cols + cat_cols], df["fraud"]

# -------- 3. 資料切分 --------
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
dup_mask = ~X_tr.duplicated()
X_tr, y_tr = X_tr[dup_mask], y_tr[dup_mask]

# -------- 4. 移除離群值 --------
if num_cols:
    good = ~(X_tr[num_cols].fillna(X_tr[num_cols].median()).apply(zscore).abs() >= 3).any(axis=1)
    if not good.any():
        good = pd.Series(True, index=X_tr.index)
    X_tr, y_tr = X_tr[good], y_tr[good]

# -------- 5. 特徵預處理流程 --------
enc_gender = lambda X: (X == "M").astype(int)
enc_living = lambda X: (X == "Own").astype(int)

pipes = [
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc",  MinMaxScaler())]), num_cols),
    ("cat_oh", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh",  OneHotEncoder(handle_unknown="ignore"))]),
     [c for c in cat_cols if c not in {"zip_code","gender","living_status"}]),
    ("gender", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("lab", FunctionTransformer(enc_gender, validate=False))]),
     ["gender"] if "gender" in cat_cols else []),
    ("living", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("lab", FunctionTransformer(enc_living, validate=False))]),
     ["living_status"] if "living_status" in cat_cols else []),
]
if "zip_code" in cat_cols:
    pipes.insert(1, ("zip_te", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("te",  TargetEncoder(smoothing=0.3))
    ]), ["zip_code"]))

pre = ColumnTransformer(pipes)
X_tr_arr  = pre.fit_transform(X_tr, y_tr)
X_val_arr = pre.transform(X_val)
X_te_arr  = pre.transform(X_te)

# -------- 6. 建立特徵名稱 --------
feat_names = num_cols.copy()
if "zip_code" in cat_cols:
    feat_names.append("zip_te")
oh_cols = [c for c in cat_cols if c not in {"zip_code","gender","living_status"}]
if oh_cols:
    feat_names.extend(pre.named_transformers_["cat_oh"].named_steps["oh"].get_feature_names_out(oh_cols))
if "gender" in cat_cols:
    feat_names.append("gender_is_M")
if "living_status" in cat_cols:
    feat_names.append("living_is_Own")
feat_names = [name_map.get(n, n) for n in feat_names]

X_tr_df, X_val_df, X_te_df = [pd.DataFrame(a, columns=feat_names)
                              for a in (X_tr_arr, X_val_arr, X_te_arr)]

# -------- 7. 進行 SMOTENC 過採樣 --------
cat_idx = list(range(len(num_cols) + (1 if "zip_code" in cat_cols else 0), X_tr_arr.shape[1]))
X_res, y_res = SMOTENC(cat_idx, random_state=42).fit_resample(X_tr_df, y_tr)

# -------- 8. 權重網格搜尋 --------
rows = []
for w in WEIGHTS:
    clf = xgb.XGBClassifier(
        n_estimators=900, learning_rate=0.03, max_depth=7, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=w,
        tree_method="hist", objective="binary:logistic", eval_metric="logloss", n_jobs=4
    )
    clf.fit(X_res, y_res)
    v_prob = clf.predict_proba(X_val_df)[:, 1]
    p, r, t = precision_recall_curve(y_val, v_prob)
    f1 = 2 * p * r / (p + r + 1e-8)
    best_thr = t[np.argmax(f1)]
    cm = confusion_matrix(y_val, (v_prob >= best_thr).astype(int))
    FP, FN = cm[0, 1], cm[1, 0]
    cost = FP * FP_COST + FN * FN_COST
    rows.append((w, cost, best_thr, FP, FN))

rows.sort(key=lambda x: x[1])
best_w, best_cost, best_thr, _, _ = rows[0]

with open(os.path.join(out_dir, PFX + "weight_costs.txt"), "w") as f:
    f.write("weight,cost,thr,FP,FN\n")
    for r in rows:
        f.write(",".join(map(str, r)) + "\n")

print("\n權重比較（FP=150, FN=10000）")
for w, c, t, fp, fn in rows:
    print(f"w={w:<2d} | 成本={c:>10,.0f} | 閾值={t:.3f} | FP={fp} FN={fn}")
print(f"✅  最佳選擇 w={best_w}, 閾值={best_thr:.3f}")

# -------- 9. 重新訓練 + 二階段 LR --------
clf = xgb.XGBClassifier(
    n_estimators=900, learning_rate=0.03, max_depth=7, min_child_weight=3,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=best_w,
    tree_method="hist", objective="binary:logistic", eval_metric="logloss", n_jobs=4
)
clf.fit(X_res, y_res)

low_thr, high_thr = 0.10, best_thr
val_prob = clf.predict_proba(X_val_df)[:, 1]
mid_mask = (val_prob >= low_thr) & (val_prob < high_thr)
lr = LogisticRegression(max_iter=2000, n_jobs=-1)
if mid_mask.sum() > 20:
    lr.fit(X_val_df[mid_mask], y_val[mid_mask])

# -------- 10. 測試集預測 --------
te_prob = clf.predict_proba(X_te_df)[:, 1]
pred = np.zeros_like(te_prob)
pred[te_prob >= high_thr] = 1
mid_test = (te_prob >= low_thr) & (te_prob < high_thr)
if mid_mask.sum() > 20:
    pred[mid_test] = lr.predict(X_te_df[mid_test])

# -------- 11. 評估報告 --------
report = classification_report(y_te, pred, digits=4)
cm = confusion_matrix(y_te, pred)
print("\n---- 測試集評估 ----\n", report)

# -------- 12. 圖表繪製與 SHAP 解釋 --------
def save_fig(name):
    plt.savefig(os.path.join(out_dir, PFX + name), dpi=120, bbox_inches="tight")
    plt.close()

fpr, tpr, _ = roc_curve(y_val, val_prob)
plt.figure(); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], '--')
plt.title(f"ROC AUC={auc(fpr, tpr):.3f}"); save_fig("roc_curve.png")

p, r, _ = precision_recall_curve(y_val, val_prob)
plt.figure(); plt.plot(r, p); plt.title(f"PR AUC={auc(r, p):.3f}"); save_fig("pr_curve.png")

plt.figure(); plt.imshow(cm); plt.title("Confusion Matrix")
plt.xticks([0, 1], ["Pred0", "Pred1"]); plt.yticks([0, 1], ["True0", "True1"])
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white')
save_fig("confusion_matrix.png")

explainer = shap.TreeExplainer(clf)
shap_vals = explainer.shap_values(X_te_df.iloc[:1000])
plt.figure(); shap.summary_plot(shap_vals, X_te_df.iloc[:1000], feature_names=feat_names, show=False, plot_type="bar")
save_fig("shap_summary.png")
plt.figure(); shap.summary_plot(shap_vals, X_te_df.iloc[:1000], feature_names=feat_names, show=False)
save_fig("shap_beeswarm.png")

with open(os.path.join(out_dir, PFX + "evaluation_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"最佳權重={best_w}, 閾值={best_thr:.4f}, 成本={best_cost:,.0f}\n")
    f.write(report + "\n")
    f.write(pd.DataFrame(cm, index=["True0", "True1"], columns=["Pred0", "Pred1"]).to_string())

print("\n✅  XGB_weight_costs.txt 與所有 XGB_* 圖表／報告 已完成生成。")
