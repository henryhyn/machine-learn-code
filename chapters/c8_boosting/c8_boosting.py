"""C8 集成学习 (二)：Boosting —— 正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/c8_boosting/c8_boosting.py
"""

# ---- 代码 1 ----
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

stump = DecisionTreeClassifier(max_depth=1, random_state=42)
models = {
    "单个树桩": stump,
    "Bagging 100 个树桩": BaggingClassifier(stump, n_estimators=100, random_state=42),
    "AdaBoost 100 个树桩": AdaBoostClassifier(stump, n_estimators=100, random_state=42),
}
for name, m in models.items():
    m.fit(X_train, y_train)
    print(f"{name}: 训练 {m.score(X_train, y_train):.3f}，测试 {m.score(X_test, y_test):.3f}")

# ---- 代码 2 ----
Xa = X_train.to_numpy()
ya = np.where(y_train.to_numpy() == 1, 1, -1)        # 标签改成 ±1
w = np.full(len(ya), 1 / len(ya))
for t in range(3):
    h = DecisionTreeClassifier(max_depth=1, random_state=42).fit(Xa, ya, sample_weight=w)
    pred = h.predict(Xa)
    wrong = pred != ya
    err = w[wrong].sum()
    alpha = 0.5 * np.log((1 - err) / err)
    w = w * np.exp(-alpha * ya * pred)
    w = w / w.sum()
    print(f"第 {t + 1} 轮：{X.columns[h.tree_.feature[0]]:<20} 加权错误率 {err:.3f}，"
          f"alpha {alpha:.3f}，分错样本的新权重合计 {w[wrong].sum():.3f}")

ada = models["AdaBoost 100 个树桩"]
print("sklearn 错误率:", ada.estimator_errors_[:3].round(3))
print("sklearn 权重 / 2:", (ada.estimator_weights_[:3] / 2).round(3))

# ---- 代码 3 ----
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

Xd, yd = load_diabetes(return_X_y=True)
Xd_train, Xd_test, yd_train, yd_test = train_test_split(Xd, yd, test_size=0.25, random_state=42)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

lr = 0.1
rng = np.random.RandomState(0)               # 所有树共用一个随机数生成器，和 sklearn 的做法一致
F_train = np.full(len(yd_train), yd_train.mean())   # 起点：训练集均值
F_test = np.full(len(yd_test), yd_train.mean())
for m in range(1, 101):
    residual = yd_train - F_train            # 平方损失的负梯度 = 残差
    tree = DecisionTreeRegressor(max_depth=3, random_state=rng).fit(Xd_train, residual)
    F_train += lr * tree.predict(Xd_train)
    F_test += lr * tree.predict(Xd_test)
    if m in (1, 10, 50, 100):
        print(f"第 {m:>3} 棵树后：训练 RMSE {rmse(yd_train, F_train):.2f}，测试 RMSE {rmse(yd_test, F_test):.2f}")

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
gbr.fit(Xd_train, yd_train)
print(f"与 GradientBoostingRegressor 的最大差异：{np.abs(gbr.predict(Xd_test) - F_test).max():.1e}")

# ---- 代码 4 ----
def my_gbdt(X, y, X_new, loss, n_trees=100, lr=0.1):
    rng = np.random.RandomState(0)
    F0 = y.mean() if loss == "squared_error" else np.median(y)
    F, F_new = np.full(len(y), F0), np.full(len(X_new), F0)
    for _ in range(n_trees):
        r = y - F                                            # 残差
        g = r if loss == "squared_error" else np.where(r >= 0, 1.0, -1.0)   # 负梯度
        tree = DecisionTreeRegressor(max_depth=3, random_state=rng).fit(X, g)
        if loss == "absolute_error":        # 线搜索：叶子值改成叶子里残差的中位数
            leaf = tree.apply(X)
            for k in np.unique(leaf):
                tree.tree_.value[k, 0, 0] = np.percentile(r[leaf == k], 50, method="inverted_cdf")
        F += lr * tree.predict(X)
        F_new += lr * tree.predict(X_new)
    return F_new

r0 = yd_train[:8] - np.median(yd_train)
print("残差         ", r0)
print("绝对值负梯度 ", np.where(r0 >= 0, 1.0, -1.0))
for loss in ["squared_error", "absolute_error"]:
    mine = my_gbdt(Xd_train, yd_train, Xd_test, loss)
    sk = GradientBoostingRegressor(loss=loss, random_state=0).fit(Xd_train, yd_train)
    print(f"{loss}: 与 sklearn 的最大差异 {np.abs(sk.predict(Xd_test) - mine).max():.1e}")

# ---- 代码 5 ----
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

clfs = {
    "随机森林": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
}
for name, m in clfs.items():
    m.fit(X_train, y_train)
    auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
    print(f"{name}: 测试准确率 {m.score(X_test, y_test):.3f}，AUC {auc:.3f}")

regs = {
    "随机森林": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
}
for name, m in regs.items():
    r2 = cross_val_score(m, Xd, yd, cv=5, scoring="r2").mean()
    print(f"糖尿病 {name}: 5 折 R² {r2:.3f}")

# ---- 代码 6 ----
rng = np.random.default_rng(42)
n = 3000
city = rng.integers(0, 40, size=n)
city_effect = rng.normal(0, 10, size=40)       # 每个城市的影响，和编号大小无关
income = rng.normal(50, 15, size=n)
target = city_effect[city] + 0.5 * income + rng.normal(0, 3, size=n)
income[rng.random(n) < 0.2] = np.nan            # 20% 的收入缺失，不填充

as_category = pd.DataFrame({"城市": pd.Categorical(city), "收入": income})
as_number = pd.DataFrame({"城市": city, "收入": income})
for max_iter in [20, 100]:
    for name, data in [("城市按类别", as_category), ("城市按编号", as_number)]:
        m = HistGradientBoostingRegressor(max_iter=max_iter, random_state=42)
        r2 = cross_val_score(m, data, target, cv=5).mean()
        print(f"max_iter={max_iter:<3} {name}：5 折 R² {r2:.3f}")

# ---- 代码 7 ----
from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_squared_error

Xf, yf = make_friedman1(n_samples=1000, noise=1.0, random_state=42)
Xf_train, Xf_val, yf_train, yf_val = train_test_split(Xf, yf, test_size=0.25, random_state=42)
for lr, sub in [(1.0, 1.0), (0.3, 1.0), (0.1, 1.0), (0.05, 1.0), (0.05, 0.5)]:
    m = GradientBoostingRegressor(learning_rate=lr, n_estimators=1000, subsample=sub,
                                  random_state=42).fit(Xf_train, yf_train)
    errs = [mean_squared_error(yf_val, p) for p in m.staged_predict(Xf_val)]
    best = int(np.argmin(errs))
    print(f"learning_rate={lr:<4} subsample={sub}：最好在第 {best + 1:>3} 棵，"
          f"验证 MSE {errs[best]:.3f}，1000 棵时 {errs[-1]:.3f}")

# ---- 代码 8 ----
es = GradientBoostingRegressor(n_estimators=2000, n_iter_no_change=10,
                               validation_fraction=0.1, random_state=42).fit(Xf_train, yf_train)
print(f"GradientBoosting 早停：用了 {es.n_estimators_} 棵树，"
      f"验证 MSE {mean_squared_error(yf_val, es.predict(Xf_val)):.3f}")

for early in ["auto", True]:
    h = HistGradientBoostingRegressor(early_stopping=early, random_state=42).fit(Xf_train, yf_train)
    print(f"HistGradientBoosting early_stopping={early!r}：启用 {h.do_early_stopping_}，用了 {h.n_iter_} 棵树")

# ---- 代码 9 ----
rng = np.random.default_rng(42)
flip = rng.random(len(y_train)) < 0.2
y_noisy = np.where(flip, 1 - y_train, y_train)
print(f"翻转了 {flip.sum()} 个标签")

noisy_models = {
    "随机森林": RandomForestClassifier(random_state=42),
    "GradientBoosting 100 棵": GradientBoostingClassifier(random_state=42),
    "GradientBoosting 500 棵": GradientBoostingClassifier(n_estimators=500, random_state=42),
    "AdaBoost 500 个树桩": AdaBoostClassifier(n_estimators=500, random_state=42),
}
for name, m in noisy_models.items():
    m.fit(X_train, y_noisy)
    print(f"{name}: 带噪训练集 {m.score(X_train, y_noisy):.3f}，干净测试集 {m.score(X_test, y_test):.3f}")

# ---- 代码 10 ----
from sklearn.metrics import mean_absolute_error

yd_outlier = yd_train.copy()
idx = np.random.default_rng(42).choice(len(yd_outlier), size=15, replace=False)
yd_outlier[idx] += 1000
for loss in ["squared_error", "absolute_error", "huber"]:
    clean = GradientBoostingRegressor(loss=loss, random_state=0).fit(Xd_train, yd_train)
    dirty = GradientBoostingRegressor(loss=loss, random_state=0).fit(Xd_train, yd_outlier)
    print(f"{loss:<15} 测试 MAE：干净 {mean_absolute_error(yd_test, clean.predict(Xd_test)):.2f}，"
          f"有离群点 {mean_absolute_error(yd_test, dirty.predict(Xd_test)):.2f}")
