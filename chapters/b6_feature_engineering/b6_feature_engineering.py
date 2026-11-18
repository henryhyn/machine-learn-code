"""B6 特征工程与特征选择：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/b6_feature_engineering/b6_feature_engineering.py
"""

# ---- 代码 1 ----
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

rng = np.random.default_rng(0)
n = 2000
income = rng.lognormal(9, 0.5, n)                  # 月收入
payment = income * rng.uniform(0.05, 0.9, n)       # 月供
debt = pd.DataFrame({"收入": income, "月供": payment})
y_debt = (payment / income > 0.5) ^ (rng.random(n) < 0.05)    # 月供超过收入一半算高风险，5% 标签噪声

price, qty = rng.lognormal(3, 1, n), rng.lognormal(2, 1, n)
order = pd.DataFrame({"单价": price, "数量": qty})
y_order = (price * qty > 150) ^ (rng.random(n) < 0.05)         # 订单金额超过 150 算大单

models = {"逻辑回归": make_pipeline(StandardScaler(), LogisticRegression()),
          "决策树": DecisionTreeClassifier(max_depth=5, random_state=0),
          "随机森林": RandomForestClassifier(random_state=0)}

def scores(X, y):
    return {name: cross_val_score(m, X, y, cv=5).mean() for name, m in models.items()}

print(pd.DataFrame({
    "贷款：原始": scores(debt, y_debt),
    "贷款：+月供/收入": scores(debt.assign(负债比=debt["月供"] / debt["收入"]), y_debt),
    "订单：原始": scores(order, y_order),
    "订单：+单价×数量": scores(order.assign(金额=order["单价"] * order["数量"]), y_order),
    "订单：取对数": scores(np.log(order), y_order),
}).T.round(3))

# ---- 代码 2 ----
from sklearn.preprocessing import PolynomialFeatures

abc = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
inter = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit(abc)
print(inter.get_feature_names_out().tolist())
for d in [10, 50, 200]:
    m = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit(np.zeros((1, d)))
    print(f"{d} 个特征 → {m.n_output_features_} 个")

# ---- 代码 3 ----
from sklearn.preprocessing import OneHotEncoder

rng = np.random.default_rng(1)
w = 1 / np.arange(1, 201) ** 1.2                     # 200 个城市，长尾分布
cities = rng.choice([f"城市{i:03d}" for i in range(200)], size=3000, p=w / w.sum())
city_train, city_test = pd.DataFrame({"城市": cities[:2000]}), pd.DataFrame({"城市": cities[2000:]})

freq = city_train["城市"].value_counts()             # 只在训练集上数
print(f"训练集 {freq.size} 个城市，其中 {(freq < 20).sum()} 个出现不到 20 次，合计占 {freq[freq < 20].sum() / 2000:.0%} 的样本")
for mf in [None, 20]:
    ohe = OneHotEncoder(min_frequency=mf, handle_unknown="infrequent_if_exist", sparse_output=False)
    ohe.fit(city_train)
    print(f"min_frequency={mf}: {len(ohe.get_feature_names_out())} 列")

unseen = sorted(set(city_test["城市"]) - set(freq.index))
examples = pd.DataFrame({"城市": ["城市000", freq.index[-1], unseen[0]]})    # 常见、罕见、没见过
onehot = ohe.transform(examples)
print(pd.DataFrame({
    "城市": examples["城市"],
    "独热:城市000": onehot[:, 0].astype(int),
    "独热:罕见合并": onehot[:, -1].astype(int),
    "计数编码": examples["城市"].map(freq).fillna(0).astype(int),
}).to_string(index=False))

# ---- 代码 4 ----
ts = pd.Series(pd.to_datetime(["2026-09-30 23:40", "2026-10-01 09:15", "2026-10-03 14:05", "2026-10-08 20:30"]))
holidays = pd.date_range("2026-10-01", "2026-10-07")          # 节假日表要自己维护
print(pd.DataFrame({
    "时间": ts.dt.strftime("%m-%d %H:%M"),
    "星期": ts.dt.day_of_week,                                # 0 是周一
    "小时": ts.dt.hour,
    "周末": (ts.dt.day_of_week >= 5).astype(int),
    "节假日": ts.dt.normalize().isin(holidays).astype(int),
    "小时sin": np.sin(2 * np.pi * ts.dt.hour / 24).round(3),
    "小时cos": np.cos(2 * np.pi * ts.dt.hour / 24).round(3),
}).to_string(index=False))

# ---- 代码 5 ----
rng = np.random.default_rng(2)
orders = pd.DataFrame({
    "用户": rng.integers(1, 501, 5000),
    "下单时间": (pd.Timestamp("2026-01-01")
               + pd.to_timedelta(rng.uniform(0, 300, 5000), unit="D")).floor("min"),
    "金额": rng.lognormal(4, 0.8, 5000).round(1),
})
cutoff = pd.Timestamp("2026-09-01")                  # 预测时间点
hist = orders[orders["下单时间"] < cutoff]            # 只用这之前的记录
recent = hist[hist["下单时间"] >= cutoff - pd.Timedelta(days=30)]

user = hist.groupby("用户").agg(订单数=("金额", "size"), 总金额=("金额", "sum"),
                              平均金额=("金额", "mean"), 最近下单=("下单时间", "max"))
user["距上次下单天数"] = (cutoff - user.pop("最近下单")).dt.days
user["近30天订单数"] = recent.groupby("用户").size().reindex(user.index, fill_value=0)
print(f"{len(orders)} 条订单 → {len(hist)} 条在预测时间点之前 → {user.shape[0]} 个用户 × {user.shape[1]} 个特征")
print(user.round(1).head(4))

# ---- 代码 6 ----
from sklearn.preprocessing import FunctionTransformer

def add_ratio(X):
    return X.assign(负债比=X["月供"] / X["收入"])

ratio_tree = make_pipeline(FunctionTransformer(add_ratio), DecisionTreeClassifier(max_depth=5, random_state=0))
print(f"交叉验证准确率 {cross_val_score(ratio_tree, debt, y_debt, cv=5).mean():.3f}")
ratio_tree.fit(debt, y_debt)
new = pd.DataFrame({"收入": [8000.0, 8000.0], "月供": [2000.0, 6000.0]})
print("线上只传原始字段，预测：", ratio_tree.predict(new))

# ---- 代码 7 ----
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer

rng = np.random.default_rng(3)
hour = rng.uniform(0, 24, 3000)
volume = (10 + 6 * np.cos(2 * np.pi * (hour - 20) / 24)        # 晚上 8 点是高峰
          + 2 * np.cos(4 * np.pi * (hour - 12) / 24) + rng.normal(0, 1.5, 3000))
H = pd.DataFrame({"小时": hour})

def sin_cos(X):
    x = np.asarray(X, dtype=float)
    return np.hstack([np.sin(2 * np.pi * x / 24), np.cos(2 * np.pi * x / 24)])

cv = KFold(5, shuffle=True, random_state=0)
for name, m in {
    "原始小时": LinearRegression(),
    "sin/cos": make_pipeline(FunctionTransformer(sin_cos), LinearRegression()),
    "周期样条": make_pipeline(SplineTransformer(knots=np.linspace(0, 24, 7).reshape(-1, 1),
                                              extrapolation="periodic"), LinearRegression()),
}.items():
    print(f"{name:<8} R² {cross_val_score(m, H, volume, cv=cv).mean():.3f}")

# ---- 代码 8 ----
from functools import partial
from sklearn.datasets import make_classification
from sklearn.feature_selection import (RFE, RFECV, SelectFromModel, SelectKBest,
                                       SequentialFeatureSelector, f_classif, mutual_info_classif)
from sklearn.model_selection import StratifiedKFold, cross_validate

X, y = make_classification(n_samples=300, n_features=50, n_informative=5, n_redundant=0,
                           shuffle=False, random_state=0)   # 前 5 列有用，其余 45 列是噪声
cv = StratifiedKFold(5, shuffle=True, random_state=0)
selectors = {
    "不选 (50 个)": "passthrough",
    "过滤：F 检验": SelectKBest(f_classif, k=5),
    "过滤：互信息": SelectKBest(partial(mutual_info_classif, random_state=0), k=5),
    "包裹：RFE": RFE(LogisticRegression(), n_features_to_select=5),
    "包裹：RFECV": RFECV(LogisticRegression(), cv=5),
    "包裹：前向逐步": SequentialFeatureSelector(LogisticRegression(), n_features_to_select=5, cv=3),
    "嵌入：L1": SelectFromModel(LogisticRegression(l1_ratio=1, C=0.1, solver="liblinear", random_state=0)),
    "嵌入：随机森林": SelectFromModel(RandomForestClassifier(random_state=0), max_features=5, threshold=-np.inf),
}
for name, sel in selectors.items():
    pipe = make_pipeline(StandardScaler(), sel, LogisticRegression())
    r = cross_validate(pipe, X, y, cv=cv, return_estimator=True)
    if sel == "passthrough":
        picked = [np.arange(50)] * 5
    else:
        picked = [np.flatnonzero(est[1].get_support()) for est in r["estimator"]]
    n_sel = np.mean([len(p) for p in picked])
    hit = np.mean([np.sum(p < 5) for p in picked])
    print(f"{name:<9} 平均选 {n_sel:4.1f} 个，命中 {hit:.1f} 个，准确率 {r['test_score'].mean():.3f}")
oracle = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()), X[:, :5], y, cv=cv)
print(f"只用真实的 5 个  准确率 {oracle.mean():.3f}")
F_all = f_classif(X, y)[0]                          # 只为看清数据，不参与建模
print(f"真实 5 列的 F 值 {F_all[:5].round(1)}，噪声列最大 {F_all[5:].max():.1f}")

# ---- 代码 9 ----
rng = np.random.default_rng(4)
Z = rng.normal(size=(1000, 10))
y_xor = ((Z[:, 0] > 0) ^ (Z[:, 1] > 0)).astype(int)   # 只看前两列的“异或”，单看任何一列都没用
F, _ = f_classif(Z, y_xor)
mi = mutual_info_classif(Z, y_xor, random_state=0)
rf_imp = RandomForestClassifier(random_state=0).fit(Z, y_xor).feature_importances_
print(pd.DataFrame({"F 值": F, "互信息": mi, "随机森林重要性": rf_imp},
                   index=[f"z{i}" for i in range(10)]).round(3).head(4))
