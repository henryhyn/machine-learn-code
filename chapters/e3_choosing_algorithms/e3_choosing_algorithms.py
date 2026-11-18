"""E3 怎么选算法：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/e3_choosing_algorithms/e3_choosing_algorithms.py
"""

# ---- 代码 1 ----
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

rng = np.random.default_rng(0)
X_lin = rng.normal(size=(500, 10))                                   # 边界是一个超平面
y_lin = (X_lin @ rng.normal(size=10) + rng.normal(0, 1, 500) > 0).astype(int)

vocab = rng.dirichlet(np.full(1000, 0.1))                            # 1000 个“词”的出现频率
vocab_b = vocab * np.exp(rng.normal(0, 0.5, 1000))                   # 第二类文档的用词略有不同
y_txt = rng.integers(0, 2, 300)
X_txt = np.array([rng.multinomial(60, vocab if c == 0 else vocab_b / vocab_b.sum()) for c in y_txt])

datasets = {
    "乳腺癌": load_breast_cancer(return_X_y=True),
    "线性边界": (X_lin, y_lin),
    "月牙": make_moons(n_samples=500, noise=0.3, random_state=0),
    "词频": (X_txt, y_txt),
}
models = {
    "Dummy": DummyClassifier(),
    "逻辑回归": LogisticRegression(max_iter=1000),
    "kNN": KNeighborsClassifier(),
    "高斯NB": GaussianNB(),
    "决策树": DecisionTreeClassifier(random_state=0),
    "SVC": SVC(),
    "随机森林": RandomForestClassifier(random_state=0, n_jobs=-1),
    "HGB": HistGradientBoostingClassifier(random_state=0),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = pd.DataFrame({
    name: {m: cross_val_score(make_pipeline(StandardScaler(), est), X, y, cv=cv).mean()
           for m, est in models.items()}
    for name, (X, y) in datasets.items()
}).T
print(scores.round(3).to_string())
print("各数据集的第一名：", scores.idxmax(axis=1).to_dict())

# ---- 代码 2 ----
gap = scores.max(axis=1).to_numpy()[:, None] - scores
print("离每份数据第一名的最大差距：")
print(gap.drop(columns="Dummy").max().sort_values().round(3).to_string())

# ---- 代码 3 ----
rng = np.random.default_rng(1)
X_all = rng.normal(size=(15000, 10))
logit = X_all[:, 0] - X_all[:, 1] + 0.5 * X_all[:, 2] + X_all[:, 3] * X_all[:, 4]   # 线性部分 + 一个交互
y_all = (rng.random(15000) < 1 / (1 + np.exp(-2 * logit))).astype(int)
X_pool, y_pool, X_te, y_te = X_all[:10000], y_all[:10000], X_all[10000:], y_all[10000:]

trio = {"逻辑回归": make_pipeline(StandardScaler(), LogisticRegression()),
        "随机森林": RandomForestClassifier(random_state=0, n_jobs=-1),
        "HGB": HistGradientBoostingClassifier(random_state=0)}
for n in [50, 200, 1000, 5000]:
    acc = {m: [] for m in trio}
    for rep in range(3):                                             # 每个样本量抽 3 次取平均
        idx = np.random.default_rng(rep).choice(10000, size=n, replace=False)
        for m, est in trio.items():
            acc[m].append(est.fit(X_pool[idx], y_pool[idx]).score(X_te, y_te))
    print(f"训练样本 {n:>4}：" + "，".join(f"{m} {np.mean(a):.3f}" for m, a in acc.items()))

# ---- 代码 4 ----
import pickle

X_bc, y_bc = datasets["乳腺癌"]
for m in ["逻辑回归", "kNN", "决策树", "SVC", "随机森林", "HGB"]:
    fitted = make_pipeline(StandardScaler(), models[m]).fit(X_bc, y_bc)
    print(f"{m}: 模型文件 {len(pickle.dumps(fitted)) / 1024:.0f} KB")
