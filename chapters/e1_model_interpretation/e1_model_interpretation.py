"""E1 模型解释：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/e1_model_interpretation/e1_model_interpretation.py
"""

# ---- 代码 1 ----
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True, as_frame=True, scaled=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

raw = LinearRegression().fit(X_train, y_train)
lin = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
print(pd.DataFrame({"标准差": X_train.std(ddof=0), "原始系数": raw.coef_,
                    "标准化后系数": lin[-1].coef_}).round(2))
print(f"测试集 R² {lin.score(X_test, y_test):.3f}")

# ---- 代码 2 ----
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, cross_validate

print(f"s1 与 s2 的相关系数 {X['s1'].corr(X['s2']):.2f}")
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
for name, reg in [("线性回归", LinearRegression()), ("岭回归 alpha=10", Ridge(alpha=10))]:
    res = cross_validate(make_pipeline(StandardScaler(), reg), X, y, cv=cv, return_estimator=True)
    coefs = pd.DataFrame([est[-1].coef_ for est in res["estimator"]], columns=X.columns)
    print(f"{name}: 25 次拟合的系数范围")
    for col in ["bmi", "s1", "s2", "s3"]:
        print(f"  {col:>3}  {coefs[col].min():6.1f} ~ {coefs[col].max():6.1f}")

# ---- 代码 3 ----
x0 = X_test.iloc[[0]]
z0 = lin[0].transform(x0)[0]                    # 标准化后的特征值
contrib = pd.Series(lin[-1].coef_ * z0, index=X.columns)
print(contrib.sort_values(key=abs, ascending=False).head(4).round(1))
print(f"截距 {lin[-1].intercept_:.1f} + 贡献之和 {contrib.sum():.1f} = {lin[-1].intercept_ + contrib.sum():.1f}")
print(f"模型预测 {lin.predict(x0)[0]:.1f}")

# ---- 代码 4 ----
from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

Xf, yf = make_friedman1(n_samples=2000, n_features=7, noise=1.0, random_state=0)
Xf = pd.DataFrame(Xf, columns=[f"x{i}" for i in range(1, 8)])
Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf, yf, test_size=0.3, random_state=0)

rf = RandomForestRegressor(random_state=0, n_jobs=-1).fit(Xf_train, yf_train)
print(f"测试集 R² {rf.score(Xf_test, yf_test):.3f}")
pi = permutation_importance(rf, Xf_test, yf_test, n_repeats=10, random_state=0, n_jobs=-1)
print(pd.DataFrame({"均值": pi.importances_mean, "标准差": pi.importances_std},
                   index=Xf.columns).round(3))

# ---- 代码 5 ----
rng = np.random.default_rng(0)
Xc = Xf.assign(x4_copy=Xf["x4"] + rng.normal(0, 0.02, len(Xf)))   # x4 的“另一种测量”
Xc_train, Xc_test = Xc.loc[Xf_train.index], Xc.loc[Xf_test.index]
print(f"x4 与 x4_copy 的相关系数 {Xc['x4'].corr(Xc['x4_copy']):.3f}")

rf_c = RandomForestRegressor(random_state=0, n_jobs=-1).fit(Xc_train, yf_train)
print(f"测试集 R² {rf_c.score(Xc_test, yf_test):.3f}")
pi_c = permutation_importance(rf_c, Xc_test, yf_test, n_repeats=10, random_state=0, n_jobs=-1)
print(pd.Series(pi_c.importances_mean, index=Xc.columns).round(3))

# ---- 代码 6 ----
def grouped_permutation_importance(model, X, y, groups, n_repeats=10, seed=0):
    """同一组的列用同一个行顺序一起打乱，组内的相关关系保持不变。"""
    rng = np.random.default_rng(seed)
    base = model.score(X, y)
    rows = {}
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            drops.append(base - model.score(X_perm, y))
        rows[name] = {"均值": np.mean(drops), "标准差": np.std(drops)}
    return pd.DataFrame(rows).T.round(3)

print(grouped_permutation_importance(rf_c, Xc_test, yf_test, {
    "x4 + x4_copy": ["x4", "x4_copy"], "x1": ["x1"], "x6": ["x6"]}))

# ---- 代码 7 ----
pi_lin = permutation_importance(lin, X_test, y_test, n_repeats=10, random_state=0)
print(pd.Series(pi_lin.importances_mean, index=X_test.columns)
      .sort_values(ascending=False).head(5).round(3))

# ---- 代码 8 ----
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence

hgb = HistGradientBoostingRegressor(random_state=0).fit(Xf_train, yf_train)
print(f"测试集 R² {hgb.score(Xf_test, yf_test):.3f}")
grid = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
truth = {"x3": 20 * (grid - 0.5) ** 2, "x4": 10 * grid, "x6": 0 * grid}
for col, true in truth.items():
    pdp = partial_dependence(hgb, Xf_train, [col], custom_values={col: grid})["average"][0]
    print(f"{col}  部分依赖 {np.round(pdp - pdp.mean(), 2)}")
    print(f"    真实形状 {np.round(true - true.mean(), 2)}")

# ---- 代码 9 ----
rng = np.random.default_rng(0)
n = 1000
Xi = pd.DataFrame({"x": rng.uniform(-1, 1, n),
                   "s": rng.choice([-1, 1], n),       # 一个开关：决定 x 的作用方向
                   "z": rng.normal(size=n)})
yi = 3 * Xi["x"] * Xi["s"] + Xi["z"] + rng.normal(0, 0.3, n)
hgb_i = HistGradientBoostingRegressor(random_state=0).fit(Xi, yi)

res = partial_dependence(hgb_i, Xi, ["x"], kind="both", custom_values={"x": [-0.8, 0, 0.8]})
print("部分依赖 (平均)：", res["average"][0].round(2))
ice = res["individual"][0]                          # 每个样本一条曲线：n × 3
change = ice[:, -1] - ice[:, 0]                     # x 从 -0.8 到 0.8，预测变了多少
for s in [-1, 1]:
    print(f"s = {s:>2} 的 ICE：平均变化 {change[Xi['s'] == s].mean():.2f}")

# ---- 代码 10 ----
rng = np.random.default_rng(0)
x1 = rng.uniform(0, 1, 1000)
x2 = x1 + rng.normal(0, 0.05, 1000)               # 和 x1 高度相关，比如身高和臂展
gap = np.abs(x1 - x2).max()
print(f"训练数据里 |x1 - x2| 最大 {gap:.2f}")
for v in [0.1, 0.5, 0.9]:
    unseen = np.mean(np.abs(v - x2) > gap)
    print(f"把 x1 全部设为 {v}：{unseen:.0%} 的样本落到训练数据从没出现过的组合")

# ---- 代码 11 ----
rng = np.random.default_rng(0)
n = 3000
income = rng.normal(0, 1, n)                                     # 收入：模型看不到
level = (income + rng.normal(0, 0.5, n) > 0.5).astype(int)       # 会员等级：收入高的人更可能是高级会员
tenure = rng.uniform(0, 5, n)                                    # 注册年数
spend = 100 + 30 * income + 5 * tenure + rng.normal(0, 10, n)    # 消费额：只取决于收入和注册年数

Xk = pd.DataFrame({"level": level, "tenure": tenure})
Xk_train, Xk_test, sk_train, sk_test = train_test_split(Xk, spend, test_size=0.3, random_state=0)
hgb_k = HistGradientBoostingRegressor(random_state=0).fit(Xk_train, sk_train)
pi_k = permutation_importance(hgb_k, Xk_test, sk_test, n_repeats=10, random_state=0)
print(f"测试集 R² {hgb_k.score(Xk_test, sk_test):.3f}")
for col, imp in zip(Xk.columns, pi_k.importances_mean):
    print(f"置换重要性 {col}: {imp:.3f}")

low = Xk_test[Xk_test["level"] == 0]
uplift = hgb_k.predict(low.assign(level=1)) - hgb_k.predict(low)
print(f"把普通会员都改成高级会员，模型预测的消费平均增加 {uplift.mean():.1f}")

# ---- 代码 12 ----
Xs, ys = make_friedman1(n_samples=300, n_features=7, noise=3.0, random_state=1)
Xs = pd.DataFrame(Xs, columns=[f"x{i}" for i in range(1, 8)])
Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, test_size=0.5, random_state=0)
grid = np.linspace(0.05, 0.95, 10)
for cst in [None, {"x4": 1, "x5": 1}]:
    m = HistGradientBoostingRegressor(monotonic_cst=cst, random_state=0).fit(Xs_train, ys_train)
    pdp = partial_dependence(m, Xs_train, ["x5"], custom_values={"x5": grid})["average"][0]
    print(f"monotonic_cst={cst}: 测试集 R² {m.score(Xs_test, ys_test):.3f}，"
          f"x5 的部分依赖下降了 {(np.diff(pdp) < 0).sum()} 次")
    print("  ", pdp.round(2))

# ---- 代码 13 ----
for name, cst in [("不限制", None), ("完全不许交互", "no_interactions"),
                  ("只许 x1、x2 交互", [{0, 1}, {2}, {3}, {4}, {5}, {6}])]:
    m = HistGradientBoostingRegressor(interaction_cst=cst, random_state=0).fit(Xf_train, yf_train)
    print(f"{name}: 测试集 R² {m.score(Xf_test, yf_test):.3f}")
