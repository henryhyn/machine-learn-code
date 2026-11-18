"""C2 逻辑回归：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/c2_logistic_regression/c2_logistic_regression.py
"""

# ---- 代码 1 ----
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(2, 1, 50), rng.normal(6, 1, 50)])   # 负类在 2 附近，正类在 6 附近
y = np.r_[np.zeros(50), np.ones(50)].astype(int)

def fit_both(x, y):
    lin = LinearRegression().fit(x.reshape(-1, 1), y)
    log = LogisticRegression().fit(x.reshape(-1, 1), y)
    lin_cut = (0.5 - lin.intercept_) / lin.coef_[0]      # 线性回归预测值 = 0.5 的位置
    log_cut = -log.intercept_[0] / log.coef_[0, 0]       # 逻辑回归概率 = 0.5 的位置
    return lin, log, lin_cut, log_cut

lin, log, lin_cut, log_cut = fit_both(x, y)
print("线性回归在 x = -2, 4, 12 的预测：", lin.predict([[-2], [4], [12]]).round(2))
print(f"分界点：线性回归 {lin_cut:.2f}，逻辑回归 {log_cut:.2f}")

# 再加 20 个远在 x≈30 的正类样本：它们本来就该判为正类，不难分
x2 = np.r_[x, rng.normal(30, 2, 20)]
y2 = np.r_[y, np.ones(20, dtype=int)]
lin2, log2, lin_cut2, log_cut2 = fit_both(x2, y2)
acc_lin = ((lin2.predict(x.reshape(-1, 1)) > 0.5) == y).mean()
print(f"加远处样本后分界点：线性回归 {lin_cut2:.2f}，逻辑回归 {log_cut2:.2f}")
print(f"原来 100 个样本的准确率：线性回归 {acc_lin:.2f}，逻辑回归 {log2.score(x.reshape(-1, 1), y):.2f}")

# ---- 代码 2 ----
w, b = log.coef_[0, 0], log.intercept_[0]
for v in [3.0, 4.0, 5.0]:
    p = log.predict_proba([[v]])[0, 1]
    print(f"x = {v}: 概率 {p:.3f}，几率 {p / (1 - p):7.3f}，对数几率 {np.log(p / (1 - p)):6.3f}，w·x+b = {w * v + b:6.3f}")
print(f"x 每加 1，几率乘以 exp(w) = {np.exp(w):.2f}")

# ---- 代码 3 ----
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gd_logistic(X, y, C=np.inf, lr=1.0, n_iter=5000):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(n_iter):
        p = sigmoid(X @ w + b)
        w -= lr * (X.T @ (p - y) / n + w / (C * n))    # 交叉熵的梯度 + L2 正则的梯度
        b -= lr * (p - y).mean()
    return w, b

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
Xs = StandardScaler().fit_transform(X_train)

w_gd, b_gd = gd_logistic(Xs, y_train.to_numpy(), C=1.0)
sk = LogisticRegression(C=1.0, tol=1e-8, max_iter=10000).fit(Xs, y_train)
print("手写前 4 个系数：", w_gd[:4].round(4), f"截距 {b_gd:.4f}")
print("sklearn 前 4 个：", sk.coef_[0, :4].round(4), f"截距 {sk.intercept_[0]:.4f}")
print("30 个系数全部一致 (误差 < 1e-4)：", np.allclose(w_gd, sk.coef_[0], atol=1e-4))

# ---- 代码 4 ----
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_train, y_train)
print(f"训练集准确率 {model.score(X_train, y_train):.3f}，测试集准确率 {model.score(X_test, y_test):.3f}")

z = model.decision_function(X_test[:3])
print("线性打分 z：", z.round(3))
print("sigmoid(z)：", sigmoid(z).round(4).tolist())
print("predict_proba 第 2 列：", model.predict_proba(X_test[:3])[:, 1].round(4).tolist())

# ---- 代码 5 ----
import pandas as pd

coef = pd.Series(model[-1].coef_[0], index=X.columns)
top = coef.reindex(coef.abs().sort_values(ascending=False).index)[:5]
print(pd.DataFrame({"系数": top.round(3), "exp(系数)": np.exp(top).round(3)}))

# ---- 代码 6 ----
from sklearn.model_selection import cross_val_score

for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=10000))
    acc = cross_val_score(m, X_train, y_train, cv=5).mean()
    norm = np.linalg.norm(m.fit(X_train, y_train)[-1].coef_)
    print(f"C={C:<6} 交叉验证准确率 {acc:.3f}，系数的 L2 范数 {norm:6.2f}")

# ---- 代码 7 ----
import warnings

for max_iter in [100, 1000, 10000]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = LogisticRegression(max_iter=max_iter).fit(X_train, y_train)
    names = sorted({w.category.__name__ for w in caught}) or ["无"]
    print(f"不标准化，max_iter={max_iter:<5} 实际迭代 {m.n_iter_[0]:>4} 次，警告：{names[0]}")
print(f"标准化后，默认 max_iter=100，实际迭代 {model[-1].n_iter_[0]} 次")

# ---- 代码 8 ----
for C in [0.01, 0.1, 1]:
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(C=C, l1_ratio=1, solver="liblinear", random_state=0)).fit(X_train, y_train)
    print(f"L1，C={C:<4} 非零系数 {(m[-1].coef_ != 0).sum():>2} / 30，测试集准确率 {m.score(X_test, y_test):.3f}")

# ---- 代码 9 ----
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier

Xi, yi = load_iris(return_X_y=True)
Xi_train, Xi_test, yi_train, yi_test = train_test_split(
    Xi, yi, test_size=0.3, stratify=yi, random_state=0)

soft = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xi_train, yi_train)
ovr = make_pipeline(StandardScaler(), OneVsRestClassifier(LogisticRegression())).fit(Xi_train, yi_train)
print("softmax 的 coef_ 形状：", soft[-1].coef_.shape)

Z = soft.decision_function(Xi_test[:2])
P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)      # 手算 softmax
for i in range(2):
    print(f"样本 {i}：手算 softmax {P[i].round(3)}，predict_proba {soft.predict_proba(Xi_test[i:i+1])[0].round(3)}，"
          f"OvR {ovr.predict_proba(Xi_test[i:i+1])[0].round(3)}")
for name, m in [("softmax", soft), ("OvR", ovr)]:
    print(f"{name} 5 折交叉验证准确率 {cross_val_score(m, Xi, yi, cv=5).mean():.3f}")

# ---- 代码 10 ----
from sklearn.datasets import make_classification

Xsep, ysep = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                                 n_clusters_per_class=1, class_sep=2.0, random_state=0)
for tol in [1e-4, 1e-6, 1e-8, 1e-10]:
    free = LogisticRegression(C=np.inf, tol=tol, max_iter=100000).fit(Xsep, ysep)
    reg = LogisticRegression(C=1.0, tol=tol, max_iter=100000).fit(Xsep, ysep)
    print(f"tol={tol:<6} 不加正则：训练准确率 {free.score(Xsep, ysep):.2f}，系数范数 {np.linalg.norm(free.coef_):6.2f}"
          f"；C=1：系数范数 {np.linalg.norm(reg.coef_):.2f}")

# ---- 代码 11 ----
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

Xm, ym = make_moons(n_samples=500, noise=0.2, random_state=0)
for name, m in [("原始特征", make_pipeline(StandardScaler(), LogisticRegression())),
                ("加 3 次多项式特征", make_pipeline(PolynomialFeatures(degree=3), StandardScaler(),
                                              LogisticRegression(max_iter=1000)))]:
    print(f"{name}：5 折交叉验证准确率 {cross_val_score(m, Xm, ym, cv=5).mean():.3f}")
