"""C1 线性回归：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/c1_linear_regression/c1_linear_regression.py
"""

# ---- 代码 1 ----
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True, as_frame=True, scaled=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X.shape)
print(X.head(3))

# ---- 代码 2 ----
from sklearn.linear_model import LinearRegression

A = np.column_stack([np.ones(len(X_train)), X_train])     # 第一列全是 1，对应截距
t = y_train.to_numpy()
w = np.linalg.solve(A.T @ A, A.T @ t)                     # 正规方程 (AᵀA) w = Aᵀt

lr = LinearRegression().fit(X_train, y_train)
print(f"正规方程: 截距 {w[0]:.3f}，bmi 系数 {w[3]:.3f}")
print(f"sklearn:  截距 {lr.intercept_:.3f}，bmi 系数 {lr.coef_[2]:.3f}")
print("10 个系数都相同:", np.allclose(w[1:], lr.coef_))
residual = t - A @ w
print("残差与每一列都正交:", np.allclose(A.T @ residual, 0, atol=1e-6))

# ---- 代码 3 ----
mu, sd = X_train.to_numpy().mean(axis=0), X_train.to_numpy().std(axis=0)
Z = np.column_stack([np.ones(len(X_train)), (X_train.to_numpy() - mu) / sd])

w_gd = np.zeros(Z.shape[1])
for step in range(1, 10001):
    grad = 2 / len(t) * Z.T @ (Z @ w_gd - t)              # 均方误差对 w 的梯度
    w_gd -= 0.1 * grad                                     # 学习率 0.1
    if step in (10, 100, 1000, 10000):
        mse = ((Z @ w_gd - t) ** 2).mean()
        diff = np.abs(w_gd[1:] / sd - lr.coef_).max()      # 换回原始单位再比
        print(f"第 {step:>5} 步: 均方误差 {mse:.2f}，与 sklearn 系数的最大差异 {diff:.4f}")
print(f"最小二乘解的均方误差 {((A @ w - t) ** 2).mean():.2f}")

# ---- 代码 4 ----
coef = pd.DataFrame({"系数": lr.coef_, "标准差": X_train.std(), "每个标准差的影响": lr.coef_ * X_train.std()},
                    index=X.columns)
print(coef.round(2))
print(f"测试集 R² {lr.score(X_test, y_test):.3f}")

# ---- 代码 5 ----
alone = LinearRegression().fit(X_train[["s1"]], y_train)
print(f"只用 s1: 系数 {alone.coef_[0]:.3f}")
print(f"10 个特征一起: s1 系数 {lr.coef_[4]:.3f}")
print(f"s1 与 s2 的相关系数 {X_train['s1'].corr(X_train['s2']):.2f}")

# ---- 代码 6 ----
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, 200).reshape(-1, 1)
y_curve = 0.5 * x[:, 0] ** 2 - x[:, 0] + 2 + rng.normal(0, 0.5, 200)   # 真实规律是抛物线

line = LinearRegression().fit(x, y_curve)
poly = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression()).fit(x, y_curve)
print(f"直线:     R² {line.score(x, y_curve):.3f}")
print(f"二次多项式: R² {poly.score(x, y_curve):.3f}")
print("特征:", poly[0].get_feature_names_out(), "系数:", poly[-1].coef_.round(3), "截距:", round(poly[-1].intercept_, 3))

# ---- 代码 7 ----
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler

models = {
    "线性回归": LinearRegression(),
    "RidgeCV": RidgeCV(alphas=np.logspace(-2, 3, 30)),
    "LassoCV": LassoCV(max_iter=100_000),
}
for name, reg in models.items():
    pipe = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), reg)
    pipe.fit(X_train, y_train)
    print(f"{name:<8} 特征 {pipe[-1].n_features_in_}，非零系数 {(pipe[-1].coef_ != 0).sum():>2}，"
          f"训练集 R² {pipe.score(X_train, y_train):.3f}，测试集 R² {pipe.score(X_test, y_test):.3f}")

# ---- 代码 8 ----
from sklearn.linear_model import Lasso, Ridge

rows = {}
for a in [0.1, 1, 5, 10, 20]:
    lasso = make_pipeline(StandardScaler(), Lasso(alpha=a)).fit(X_train, y_train)
    rows[f"Lasso {a}"] = lasso[-1].coef_
for a in [1, 10, 100, 1000]:
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X_train, y_train)
    rows[f"Ridge {a}"] = ridge[-1].coef_
print(pd.DataFrame(rows, index=X.columns).T.round(1) + 0.0)   # + 0.0 把 -0.0 显示成 0.0

# ---- 代码 9 ----
for name, model in [("不标准化", Lasso(alpha=5, max_iter=100_000)),
                    ("标准化",   make_pipeline(StandardScaler(), Lasso(alpha=5)))]:
    model.fit(X_train, y_train)
    c = model[-1].coef_ if hasattr(model, "steps") else model.coef_
    print(f"{name}: 被置零 {list(X.columns[c == 0])}，测试集 R² {model.score(X_test, y_test):.3f}")

# ---- 代码 10 ----
from sklearn.linear_model import ElasticNet

def make_collinear(seed, n=50):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = x1 + rng.normal(0, 0.05, n)               # x2 几乎就是 x1
    y = 2 * x1 + 2 * x2 + rng.normal(0, 1, n)       # 真实系数 (2, 2)
    return np.column_stack([x1, x2]), y

for seed in range(5):
    Xc, yc = make_collinear(seed)
    out = []
    for reg in [LinearRegression(), Ridge(alpha=1), Lasso(alpha=0.1), ElasticNet(alpha=0.1, l1_ratio=0.5)]:
        c = reg.fit(Xc, yc).coef_
        out.append(f"({c[0]:5.2f}, {c[1]:5.2f})")
    print(f"种子 {seed}: " + "  ".join(out))

# ---- 代码 11 ----
from sklearn.linear_model import HuberRegressor

rng = np.random.default_rng(3)
x = rng.uniform(0, 10, 100).reshape(-1, 1)
y_h = 3 * x[:, 0] + 5 + rng.normal(0, 1, 100)     # 真实规律 y = 3x + 5
y_h[np.argsort(x[:, 0])[-5:]] = 0                 # x 最大的 5 个点被错记成 0

for name, reg in [("LinearRegression", LinearRegression()), ("HuberRegressor", HuberRegressor())]:
    reg.fit(x, y_h)
    print(f"{name:<16} 斜率 {reg.coef_[0]:.2f}，截距 {reg.intercept_:.2f}")

# ---- 代码 12 ----
rng = np.random.default_rng(4)
x = rng.uniform(0, 3, 100).reshape(-1, 1)
y_log = 10 * np.log1p(x[:, 0]) + rng.normal(0, 0.5, 100)   # 增长越来越慢的曲线

line = LinearRegression().fit(x, y_log)
cubic = make_pipeline(PolynomialFeatures(degree=3), LinearRegression()).fit(x, y_log)
for v in [1, 3, 5, 10]:
    print(f"x = {v:>2}: 真实 {10 * np.log1p(v):6.2f}，直线 {line.predict([[v]])[0]:6.2f}，"
          f"三次多项式 {cubic.predict([[v]])[0]:7.2f}")
