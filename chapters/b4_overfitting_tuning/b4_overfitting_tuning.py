"""B4 过拟合、偏差与方差：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/b4_overfitting_tuning/b4_overfitting_tuning.py
"""

# ---- 代码 1 ----
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

def f(x):                                   # 真实规律，实际问题里永远看不到
    return np.sin(3 * x) + 0.5 * x

sigma = 0.3                                 # 噪声标准差，噪声项 = 0.09
x_test = np.linspace(-2, 2, 50)             # 固定的 50 个测试点
rng = np.random.default_rng(0)

print(" k   偏差²    方差   噪声  三项之和  实测误差")
for k in [1, 2, 4, 8, 16, 30, 50]:
    preds = np.empty((200, len(x_test)))
    for r in range(200):                    # 200 份独立的训练集，每份 60 个点
        x = rng.uniform(-2, 2, 60)
        y = f(x) + rng.normal(0, sigma, 60)
        model = KNeighborsRegressor(n_neighbors=k).fit(x.reshape(-1, 1), y)
        preds[r] = model.predict(x_test.reshape(-1, 1))
    bias2 = ((preds.mean(axis=0) - f(x_test)) ** 2).mean()
    var = preds.var(axis=0).mean()
    y_new = f(x_test) + rng.normal(0, sigma, preds.shape)   # 测试点上重新观测的带噪标签
    err = ((preds - y_new) ** 2).mean()
    print(f"{k:>2}  {bias2:.4f}  {var:.4f}  {sigma ** 2:.2f}  {bias2 + var + sigma ** 2:.4f}  {err:.4f}")

# ---- 代码 2 ----
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, learning_curve
from sklearn.tree import DecisionTreeRegressor

X, y = make_friedman1(n_samples=2000, noise=1.0, random_state=0)
print(f"噪声决定的 R² 上限约 {1 - 1.0 / y.var():.3f}")
cv = KFold(n_splits=5, shuffle=True, random_state=0)
for name, model in [("线性回归", LinearRegression()),
                    ("不剪枝的决策树", DecisionTreeRegressor(random_state=0))]:
    sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=[50, 100, 200, 400, 800, 1600], cv=cv, scoring="r2")
    print(name)
    for n, tr, va in zip(sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)):
        print(f"  训练样本 {n:>4}  训练 R² {tr:.3f}  验证 R² {va:.3f}  差距 {tr - va:.3f}")

# ---- 代码 3 ----
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC

X_d, y_d = load_digits(return_X_y=True)
gammas = np.logspace(-6, -1, 6)             # 1e-6, 1e-5, ..., 1e-1
train_scores, val_scores = validation_curve(
    SVC(), X_d, y_d, param_name="gamma", param_range=gammas, cv=5)
for g, tr, va in zip(gammas, train_scores.mean(axis=1), val_scores.mean(axis=1)):
    print(f"gamma={g:.0e}  训练 {tr:.3f}  验证 {va:.3f}")

# ---- 代码 4 ----
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
pipe = make_pipeline(StandardScaler(), SVC())
grid = GridSearchCV(pipe, {"svc__C": [0.1, 1, 10, 100],
                           "svc__gamma": [1e-4, 1e-3, 1e-2, 1e-1]}, cv=5)
grid.fit(X_train, y_train)
print(f"试了 {len(grid.cv_results_['params'])} 组，最好 {grid.best_params_}")
print(f"best_score_ {grid.best_score_:.3f}，测试集 {grid.score(X_test, y_test):.3f}")

# ---- 代码 5 ----
def score(a, b):                            # 只有 a 重要，b 几乎不影响结果
    return np.exp(-((a - 0.37) / 0.15) ** 2) + 0.02 * b

levels = np.array([1 / 6, 1 / 2, 5 / 6])    # 3×3 网格，均匀铺满 [0, 1]²
A, B = np.meshgrid(levels, levels)
grid_best = score(A, B).max()
print(f"网格 9 组：a 只试了 {len(np.unique(A))} 个值，最好得分 {grid_best:.3f}")

rng = np.random.default_rng(0)
bests = np.array([score(*rng.uniform(0, 1, size=(2, 9))).max() for _ in range(1000)])
print(f"随机 9 组：a 试了 9 个值，最好得分平均 {bests.mean():.3f}，"
      f"{(bests > grid_best).mean():.0%} 的情况比网格好")

# ---- 代码 6 ----
from scipy.stats import loguniform, uniform

for name, dist in [("uniform(0.01, 100)", uniform(0.01, 100 - 0.01)),
                   ("loguniform(0.01, 100)", loguniform(0.01, 100))]:
    s = dist.rvs(size=10000, random_state=0)
    counts = np.histogram(s, bins=[0.01, 0.1, 1, 10, 100])[0] / len(s)
    print(f"{name:<22}" + "  ".join(f"{c:6.1%}" for c in counts))

# ---- 代码 7 ----
from sklearn.model_selection import RandomizedSearchCV

rand = RandomizedSearchCV(pipe, {"svc__C": loguniform(1e-1, 1e3),
                                 "svc__gamma": loguniform(1e-5, 1)},
                          n_iter=16, cv=5, random_state=0)
rand.fit(X_train, y_train)
print({k: round(float(v), 4) for k, v in rand.best_params_.items()})
print(f"best_score_ {rand.best_score_:.3f}，测试集 {rand.score(X_test, y_test):.3f}")

# ---- 代码 8 ----
from sklearn.experimental import enable_halving_search_cv  # noqa: F401，必须先导入
from sklearn.model_selection import HalvingRandomSearchCV

Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    X_d, y_d, test_size=0.25, stratify=y_d, random_state=0)
halving = HalvingRandomSearchCV(
    SVC(), {"C": loguniform(1e-2, 1e3), "gamma": loguniform(1e-6, 1e-1)},
    n_candidates=81, factor=3, cv=5, random_state=0)
halving.fit(Xd_train, yd_train)
print(f"每轮候选数 {halving.n_candidates_}，每轮样本数 {halving.n_resources_}")
cost = sum(c * r for c, r in zip(halving.n_candidates_, halving.n_resources_))
print(f"候选×样本 总和 {cost}，全量评估 81 组需要 {81 * len(Xd_train)}")
print({k: round(float(v), 4) for k, v in halving.best_params_.items()})
print(f"best_score_ {halving.best_score_:.3f}，测试集 {halving.score(Xd_test, yd_test):.3f}")

# ---- 代码 9 ----
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score

param_grid = {"svc__C": np.logspace(-1, 3, 5), "svc__gamma": np.logspace(-3, 0, 4)}
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
rows = []
for seed in range(5):
    X_s, y_s = make_classification(n_samples=5150, n_features=20, n_informative=4,
                                   flip_y=0.2, random_state=seed)
    X_s_tr, y_s_tr = X_s[:150], y_s[:150]           # 只有 150 个样本可用
    X_s_new, y_s_new = X_s[150:], y_s[150:]         # 5000 个“未来的数据”，只用来核对
    search = GridSearchCV(make_pipeline(StandardScaler(), SVC()), param_grid, cv=inner)
    search.fit(X_s_tr, y_s_tr)
    nested = cross_val_score(search, X_s_tr, y_s_tr, cv=outer).mean()
    rows.append((search.best_score_, nested, search.score(X_s_new, y_s_new)))
    print(f"数据 {seed}: best_score_ {rows[-1][0]:.3f}  嵌套 CV {rows[-1][1]:.3f}  新数据 {rows[-1][2]:.3f}")
m = np.mean(rows, axis=0)
print(f"平均  : best_score_ {m[0]:.3f}  嵌套 CV {m[1]:.3f}  新数据 {m[2]:.3f}")

# ---- 代码 10 ----
res = grid.cv_results_
for i in np.argsort(res["rank_test_score"], kind="stable")[:6]:
    p = res["params"][i]
    print(f"第 {res['rank_test_score'][i]} 名  C={p['svc__C']:<5} gamma={p['svc__gamma']:<6}  "
          f"{res['mean_test_score'][i]:.3f} ± {res['std_test_score'][i]:.3f}")
