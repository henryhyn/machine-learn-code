"""B5 数据泄漏与 Pipeline：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/b5_leakage_pipeline/b5_leakage_pipeline.py
"""

# ---- 代码 1 ----
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.standard_normal((100, 5000))     # 100 个样本、5000 个纯噪声特征
y = rng.integers(0, 2, size=100)         # 随机标签，和 X 毫无关系

# 错误：先在全量数据上选出 20 个“最相关”的特征，再交叉验证
X_selected = SelectKBest(k=20).fit_transform(X, y)
leaky = cross_val_score(LogisticRegression(), X_selected, y, cv=5).mean()

# 正确：特征选择放进 Pipeline，每一折只在训练部分上选
pipe = make_pipeline(SelectKBest(k=20), LogisticRegression())
clean = cross_val_score(pipe, X, y, cv=5).mean()
print(f"泄漏版本 {leaky:.3f}，Pipeline 版本 {clean:.3f}")

# ---- 代码 2 ----
for seed in range(1, 6):
    r = np.random.default_rng(seed)
    Xs, ys = r.standard_normal((100, 5000)), r.integers(0, 2, size=100)
    a = cross_val_score(LogisticRegression(), SelectKBest(k=20).fit_transform(Xs, ys), ys, cv=5).mean()
    b = cross_val_score(pipe, Xs, ys, cv=5).mean()
    print(f"种子 {seed}: 泄漏 {a:.2f}，Pipeline {b:.2f}")

# ---- 代码 3 ----
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X_bc, y_bc = load_breast_cancer(return_X_y=True)
for model in [LogisticRegression(), KNeighborsClassifier(), SVC()]:
    leaky = cross_val_score(model, StandardScaler().fit_transform(X_bc), y_bc, cv=5).mean()
    clean = cross_val_score(make_pipeline(StandardScaler(), model), X_bc, y_bc, cv=5).mean()
    print(f"{type(model).__name__:<22} 泄漏 {leaky:.4f}，Pipeline {clean:.4f}")

# ---- 代码 4 ----
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

rng = np.random.default_rng(0)
city = pd.DataFrame({"城市": rng.integers(0, 500, size=1000)})   # 500 个取值的类别特征
y_city = rng.integers(0, 2, size=1000)                           # 随机标签

# 错误：用全量数据算“每个城市的平均标签”，再交叉验证
naive = pd.Series(y_city).groupby(city["城市"]).transform("mean").to_frame()
leaky = cross_val_score(LogisticRegression(), naive, y_city, cv=5).mean()

# 正确：TargetEncoder 放进 Pipeline
encoder = TargetEncoder(cv=StratifiedKFold(5, shuffle=True, random_state=0))
clean = cross_val_score(make_pipeline(encoder, LogisticRegression()), city, y_city, cv=5).mean()
print(f"泄漏版本 {leaky:.3f}，Pipeline 版本 {clean:.3f}")

# ---- 代码 5 ----
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

X_churn, churn = make_classification(n_samples=1000, n_features=10, n_informative=3,
                                     flip_y=0.2, random_state=0)
rng = np.random.default_rng(1)
calls = churn * rng.poisson(2, size=1000)        # 挽留电话次数：只有提出注销的人才会接到
X_leak = np.column_stack([X_churn, calls])

rf = RandomForestClassifier(random_state=0)
print(f"正常特征 {cross_val_score(rf, X_churn, churn, cv=5).mean():.3f}")
print(f"加上挽留电话次数 {cross_val_score(rf, X_leak, churn, cv=5).mean():.3f}")

# ---- 代码 6 ----
for j in range(X_leak.shape[1]):
    acc = cross_val_score(RandomForestClassifier(n_estimators=50, random_state=0),
                          X_leak[:, [j]], churn, cv=5).mean()
    if acc > 0.6:
        print(f"特征 {j}: 单独使用的准确率 {acc:.3f}")

# ---- 代码 7 ----
from sklearn.model_selection import GroupKFold, KFold

rng = np.random.default_rng(0)
n_people, n_visits = 100, 10
person = rng.normal(0, 1, size=(n_people, 5))                    # 每个人的“底子”
label = (person[:, 0] + rng.normal(0, 1, size=n_people) > 0).astype(int)
X_visit = np.repeat(person, n_visits, axis=0) + rng.normal(0, 0.1, size=(1000, 5))
y_visit = np.repeat(label, n_visits)
groups = np.repeat(np.arange(n_people), n_visits)               # 每条记录属于哪个人

rf = RandomForestClassifier(random_state=0)
kf = cross_val_score(rf, X_visit, y_visit, cv=KFold(5, shuffle=True, random_state=0)).mean()
gkf = cross_val_score(rf, X_visit, y_visit, cv=GroupKFold(5), groups=groups).mean()
print(f"KFold {kf:.3f}，GroupKFold {gkf:.3f}")

# ---- 代码 8 ----
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

rng = np.random.default_rng(0)
day = np.arange(730)                                  # 两年的日销量
sales = 100 + 0.2 * day + 10 * np.sin(2 * np.pi * day / 7) + np.cumsum(rng.normal(0, 2, 730))
X_day = np.column_stack([day, day % 7])               # 特征：第几天、星期几

reg = RandomForestRegressor(random_state=0)
for name, cv in [("KFold(shuffle=True)", KFold(5, shuffle=True, random_state=0)),
                 ("TimeSeriesSplit", TimeSeriesSplit(5))]:
    mae = -cross_val_score(reg, X_day, sales, cv=cv, scoring="neg_mean_absolute_error").mean()
    print(f"{name:<20} MAE {mae:.2f}")

# ---- 代码 9 ----
for i, (tr, te) in enumerate(TimeSeriesSplit(5).split(X_day)):
    print(f"第 {i + 1} 折：训练 第 {tr[0]}–{tr[-1]} 天，验证 第 {te[0]}–{te[-1]} 天")

# ---- 代码 10 ----
print(pipe)
print(pipe.named_steps)

# ---- 代码 11 ----
rng = np.random.default_rng(42)
n = 800
members = pd.DataFrame({
    "年龄": rng.integers(18, 65, size=n).astype(float),
    "月消费": rng.gamma(2, 150, size=n).round(1),
    "城市等级": rng.choice(["一线", "二线", "三线及以下"], size=n),
    "渠道": rng.choice(["App", "小程序", "门店"], size=n),
})
logit = -0.004 * members["月消费"] + 0.03 * (members["年龄"] - 40) + (members["渠道"] == "门店") * 1.2
members["流失"] = (logit + rng.logistic(size=n) > 0).astype(int)
members.loc[rng.choice(n, 80, replace=False), "月消费"] = np.nan    # 10% 缺失
X_m, y_m = members.drop(columns="流失"), members["流失"]
print(X_m.head(3))
print(f"流失率 {y_m.mean():.3f}")

# ---- 代码 12 ----
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

preprocess = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["年龄", "月消费"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["城市等级", "渠道"]),
])
model = Pipeline([("prep", preprocess), ("clf", LogisticRegression())])

X_tr, X_te, y_tr, y_te = train_test_split(X_m, y_m, test_size=0.25, stratify=y_m, random_state=0)
grid = GridSearchCV(model, {
    "clf__C": [0.01, 0.1, 1, 10],
    "prep__num__simpleimputer__strategy": ["mean", "median"],
}, cv=5)
grid.fit(X_tr, y_tr)
print(grid.best_params_)
print(f"交叉验证 {grid.best_score_:.3f}，测试集 {grid.score(X_te, y_te):.3f}")

# ---- 代码 13 ----
new_users = pd.DataFrame({
    "年龄": [23, 58], "月消费": [np.nan, 520.0],
    "城市等级": ["二线", "海外"], "渠道": ["门店", "App"],
})
print(grid.predict_proba(new_users)[:, 1].round(3))
