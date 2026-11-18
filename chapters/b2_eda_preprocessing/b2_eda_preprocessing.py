"""B2 数据探索与预处理：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/b2_eda_preprocessing/b2_eda_preprocessing.py
"""

# ---- 代码 1 ----
import numpy as np
import pandas as pd

pd.set_option("display.unicode.east_asian_width", True)   # 中文按两个字符宽对齐

rng = np.random.default_rng(42)
n = 1000
age = rng.normal(36, 10, n).clip(18, 70).round()
tenure = rng.integers(1, 73, n)                                  # 在网月数 1–72
city = rng.choice(["一线", "二线", "三线及以下"], n, p=[0.3, 0.4, 0.3])
member = rng.choice(["普通", "银卡", "金卡"], n, p=[0.6, 0.3, 0.1])
spend = rng.lognormal(mean=5, sigma=0.6, size=n).round()        # 右偏：少数人消费很高
income = (rng.lognormal(9.0, 0.4, n) / 100).round() * 100

# 流失倾向：在网越久越稳，消费越高越容易走，会员等级越高越稳，三线及以下略高
logit = (0.6 - 0.05 * tenure + 0.003 * (spend - 150)
         - 0.8 * (member == "金卡") - 0.4 * (member == "银卡") + 0.3 * (city == "三线及以下"))
no_income = rng.random(n) < np.where(logit > 0, 0.30, 0.08)     # 想走的人更少填收入
churn = (rng.random(n) < 1 / (1 + np.exp(-(logit + 0.8 * no_income)))).astype(int)

df = pd.DataFrame({"年龄": age, "月消费": spend, "在网月数": tenure, "月收入": income,
                   "城市等级": city, "会员类型": member, "流失": churn})
df.loc[no_income, "月收入"] = np.nan
df.loc[rng.choice(n, 50, replace=False), "年龄"] = np.nan        # 随机缺失
df.loc[rng.choice(n, 4, replace=False), "年龄"] = 999            # 系统占位符
df.loc[rng.choice(n, 30, replace=False), "城市等级"] = np.nan
print(df.head())

# ---- 代码 2 ----
print(pd.DataFrame({"类型": df.dtypes.astype(str),
                    "缺失数": df.isna().sum(),
                    "缺失率": df.isna().mean().round(3)}))

# ---- 代码 3 ----
num_cols = ["年龄", "月消费", "在网月数", "月收入"]
print(df[num_cols].describe().round(1))
print(df[num_cols].skew().round(2))

# ---- 代码 4 ----
print(df["流失"].value_counts(normalize=True).round(3))
print(df.groupby("会员类型")["流失"].mean().round(3))
print(df[num_cols].corrwith(df["流失"]).round(3))

# ---- 代码 5 ----
def iqr_outliers(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    return (s < q1 - k * (q3 - q1)) | (s > q3 + k * (q3 - q1))

def zscore_outliers(s, t=3):
    return ((s - s.mean()) / s.std()).abs() > t

for col in ["年龄", "月消费"]:
    s = df[col]
    for name, mask in [("IQR", iqr_outliers(s)), ("3σ", zscore_outliers(s))]:
        vals = np.sort(s[mask].unique())
        print(f"{col} {name}: {mask.sum():>2} 个，取值 {vals[:4].tolist()} …")

# ---- 代码 6 ----
big = iqr_outliers(df["月消费"])
print(f"高消费用户流失率 {df.loc[big, '流失'].mean():.3f}，其他用户 {df.loc[~big, '流失'].mean():.3f}")

df["年龄"] = df["年龄"].replace(999, np.nan)     # 占位符是错误数据，改成缺失
print(f"年龄缺失 {df['年龄'].isna().sum()} 个，最大值 {df['年龄'].max():.0f}")

# ---- 代码 7 ----
from sklearn.model_selection import train_test_split

X, y = df.drop(columns="流失"), df["流失"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

complete = X_train.notna().all(axis=1)
print(f"训练集 {len(X_train)} 行，删掉含缺失的行后剩 {complete.sum()} 行")
print(f"流失率：全部 {y_train.mean():.3f}，删后 {y_train[complete].mean():.3f}")
print(y_train.groupby(X_train["月收入"].isna().rename("收入缺失")).agg(["size", "mean"]).round(3))

# ---- 代码 8 ----
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="median", add_indicator=True).set_output(transform="pandas")
filled = imp.fit_transform(X_train[num_cols])
print({c: float(v) for c, v in zip(num_cols, imp.statistics_)})
print(filled.columns.tolist())
print(filled[["月收入", "missingindicator_月收入"]].head(3))

# ---- 代码 9 ----
price = np.array([4, 8, 15, 21, 21, 24, 25, 28, 34])   # 已排序
bins = price.reshape(3, 3)                              # 等频分箱：每箱 3 个
by_mean = np.repeat(bins.mean(axis=1, keepdims=True), 3, axis=1)
lo, hi = bins[:, [0]], bins[:, [-1]]
by_boundary = np.where(bins - lo <= hi - bins, lo, hi)  # 离哪个边界近就换成哪个
print("箱均值光滑：", by_mean.astype(int).tolist())
print("箱边界光滑：", by_boundary.tolist())

# ---- 代码 10 ----
from sklearn.preprocessing import KBinsDiscretizer

for strategy in ["uniform", "quantile"]:
    kb = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy=strategy)
    codes = kb.fit_transform(X_train[["月消费"]]).ravel().astype(int)
    print(f"{strategy:>8}: 边界 {kb.bin_edges_[0].round(1).tolist()}，每箱人数 {np.bincount(codes).tolist()}")

# ---- 代码 11 ----
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

new = pd.DataFrame({"城市等级": ["一线", "海外"], "会员类型": ["金卡", "普通"]})

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(X_train[["城市等级"]].dropna())
print(ohe.get_feature_names_out().tolist())
print(ohe.transform(new[["城市等级"]]))

ord_enc = OrdinalEncoder(categories=[["普通", "银卡", "金卡"]])   # 显式给出顺序
ord_enc.fit(X_train[["会员类型"]])
print(ord_enc.transform(new[["会员类型"]]).ravel())

# ---- 代码 12 ----
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder().fit(X_train["城市等级"].dropna())
print({c: int(i) for c, i in zip(le.classes_, le.transform(le.classes_))})

# ---- 代码 13 ----
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

for scaler in [StandardScaler(), MinMaxScaler(), RobustScaler()]:
    z = scaler.fit_transform(X_train[["月消费"]]).ravel()
    p5, p50, p95 = np.percentile(z, [5, 50, 95])
    print(f"{type(scaler).__name__:<15} P5 {p5:6.2f}  中位数 {p50:6.2f}  P95 {p95:6.2f}  最大 {z.max():6.2f}")

# ---- 代码 14 ----
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

Xw, yw = load_wine(return_X_y=True)
print(f"特征取值范围：最小的列 {np.ptp(Xw, axis=0).min():.2f}，最大的列 {np.ptp(Xw, axis=0).max():.0f}")
for name, model in [("kNN", KNeighborsClassifier()), ("SVM (RBF)", SVC()),
                    ("逻辑回归", LogisticRegression(max_iter=5000)),
                    ("决策树", DecisionTreeClassifier(random_state=0))]:
    raw = cross_val_score(model, Xw, yw, cv=5).mean()
    scaled = cross_val_score(make_pipeline(StandardScaler(), model), Xw, yw, cv=5).mean()
    print(f"{name}：不缩放 {raw:.3f}，标准化后 {scaled:.3f}")

# ---- 代码 15 ----
from sklearn.compose import ColumnTransformer

def make_preprocessor(indicator=True):
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=indicator),
                              StandardScaler()), num_cols),
        ("city", make_pipeline(SimpleImputer(strategy="constant", fill_value="未知"),
                               OneHotEncoder(handle_unknown="ignore")), ["城市等级"]),
        ("member", OrdinalEncoder(categories=[["普通", "银卡", "金卡"]]), ["会员类型"]),
    ])

pre = make_preprocessor()
Xt_train = pre.fit_transform(X_train)     # 只在训练集上 fit
Xt_test = pre.transform(X_test)           # 测试集只 transform
print(Xt_train.shape, Xt_test.shape)
print(*pre.get_feature_names_out(), sep="\n")

# ---- 代码 16 ----
from sklearn.metrics import roc_auc_score

for indicator in [False, True]:
    pre = make_preprocessor(indicator)
    clf = LogisticRegression().fit(pre.fit_transform(X_train), y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(pre.transform(X_test))[:, 1])
    print(f"add_indicator={indicator}: 测试集 ROC-AUC {auc:.3f}")
