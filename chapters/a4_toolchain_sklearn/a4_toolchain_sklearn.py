"""A4 工具链与 sklearn 的设计：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/a4_toolchain_sklearn/a4_toolchain_sklearn.py
"""

# ---- 代码 1 ----
import sys
import sklearn

print(f"Python {sys.version_info.major}.{sys.version_info.minor}")
print("scikit-learn", ".".join(sklearn.__version__.split(".")[:2]))

# ---- 代码 2 ----
import re
from importlib.metadata import requires

deps = [re.split(r"[<>=!~; \[]", r)[0] for r in requires("scikit-learn") if "extra ==" not in r]
print(deps)
print("pandas" in deps, "matplotlib" in deps)

# ---- 代码 3 ----
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True, as_frame=True)
print(type(X).__name__, X.shape, type(y).__name__)
print(X[["alcohol", "malic_acid", "proline"]].describe().loc[["mean", "std", "min", "max"]].round(2))
arr = X.to_numpy()
print(type(arr).__name__, arr.dtype, arr.shape)

# ---- 代码 4 ----
from sklearn.linear_model import LinearRegression

alcohol = X["alcohol"].to_numpy()
try:
    LinearRegression().fit(alcohol, X["proline"])
except ValueError as e:
    lines = str(e).splitlines()
    print(lines[0])
    print(lines[-1])

X1 = alcohol.reshape(-1, 1)
print(alcohol.shape, "->", X1.shape)
reg = LinearRegression().fit(X1, X["proline"])
print(f"酒精度每高 1 度，脯氨酸平均高 {reg.coef_[0]:.1f}")

# ---- 代码 5 ----
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

for est in [StandardScaler(), PCA(), LogisticRegression(), DecisionTreeClassifier(), KMeans()]:
    methods = [m for m in ["fit", "transform", "predict", "score"] if hasattr(est, m)]
    print(f"{type(est).__name__:<24}{', '.join(methods)}")

# ---- 代码 6 ----
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0)

scaler = StandardScaler()
print("fit 之前有 mean_ 吗：", hasattr(scaler, "mean_"))
scaler.fit(X_train)
print("fit 之后有 mean_ 吗：", hasattr(scaler, "mean_"), "特征数", scaler.n_features_in_)

cols = ["alcohol", "proline"]
idx = [X.columns.get_loc(c) for c in cols]
for c, i in zip(cols, idx):
    print(f"{c}: 学到的均值 {scaler.mean_[i]:.2f}，标准差 {scaler.scale_[i]:.2f}")

# ---- 代码 7 ----
import numpy as np

X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
print(f"训练集转换后：均值全为 0 {np.allclose(X_train_s.mean(axis=0), 0)}，"
      f"标准差全为 1 {np.allclose(X_train_s.std(axis=0), 1)}")
for c, i in zip(cols, idx):
    print(f"测试集 {c}: 均值 {X_test_s[:, i].mean():.2f}，标准差 {X_test_s[:, i].std():.2f}")

# ---- 代码 8 ----
wrong = StandardScaler().fit_transform(X_test)
row = X_test.iloc[0]
for c, i in zip(cols, idx):
    print(f"{c} = {row[c]}: 正确转换 {X_test_s[0, i]:.2f}，测试集单独 fit {wrong[0, i]:.2f}")

# ---- 代码 9 ----
from sklearn.exceptions import NotFittedError

clf = LogisticRegression(max_iter=1000)
try:
    clf.predict(X_test_s)
except NotFittedError as e:
    print("NotFittedError:", e)

clf.fit(X_train_s, y_train)
print("前 5 个预测", clf.predict(X_test_s[:5]), "真实", y_test.to_numpy()[:5])
print(f"测试集准确率 {clf.score(X_test_s, y_test):.3f}")
print("coef_ 的形状", clf.coef_.shape, "classes_", clf.classes_)

# ---- 代码 10 ----
from sklearn.base import clone

m = LogisticRegression(C=0.5, max_iter=1000)
print(m)
params = m.get_params()
print(f"C = {params['C']}，solver = {params['solver']}，fit_intercept = {params['fit_intercept']}")

m.set_params(C=-1)              # 构造和 set_params 都不校验
try:
    m.fit(X_train_s, y_train)
except ValueError as e:
    print(type(e).__name__, e)

m.set_params(C=0.01).fit(X_train_s, y_train)
m2 = clone(m)
print(m2, "| m 有 coef_：", hasattr(m, "coef_"), "| m2 有 coef_：", hasattr(m2, "coef_"))

# ---- 代码 11 ----
scaler_pd = StandardScaler().set_output(transform="pandas").fit(X_train)
X_test_df = scaler_pd.transform(X_test)
print(type(X_test_df).__name__)
print(X_test_df[cols].head(3).round(2))
print(scaler_pd.get_feature_names_out()[:3])

# ---- 代码 12 ----
try:
    from sklearn.datasets import load_boston
except ImportError as e:
    print(str(e).strip().splitlines()[0])

# ---- 代码 13 ----
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

feat = pd.DataFrame({"城市": ["北京", "上海", "广州", "上海"], "尺码": ["S", "M", "L", "M"]})
try:
    LabelEncoder().fit_transform(feat)
except ValueError as e:
    print("ValueError:", e)

le = LabelEncoder()
print(le.fit_transform(["好评", "差评", "好评", "中评"]), le.classes_)
print(OrdinalEncoder().fit(feat).categories_)
