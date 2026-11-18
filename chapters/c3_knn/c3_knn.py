"""C3 k 近邻：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/c3_knn/c3_knn.py
"""

# ---- 代码 1 ----
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
print(f"存下的训练样本 {knn.n_samples_fit_} 个，特征 {knn.n_features_in_} 个")
print("内部存的就是训练集本身：", np.array_equal(knn._fit_X, X_train))
print("找邻居的方式：", knn._fit_method)

# ---- 代码 2 ----
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)

class MyKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):                      # 懒惰学习：只存数据
        self.X_, self.y_ = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(self.y_)
        return self

    def predict(self, X):
        d = np.sqrt(((X[:, None, :] - self.X_[None, :, :]) ** 2).sum(axis=2))  # (测试数, 训练数)
        idx = np.argsort(d, axis=1, kind="stable")[:, :self.k]             # 最近 k 个的下标
        votes = (self.y_[idx][:, :, None] == self.classes_).sum(axis=1)    # 每类得票数
        return self.classes_[votes.argmax(axis=1)]

mine = MyKNN(k=5).fit(Xs_train, y_train).predict(Xs_test)
ref = KNeighborsClassifier(n_neighbors=5).fit(Xs_train, y_train).predict(Xs_test)
print(f"手写版测试集准确率 {(mine == y_test).mean():.3f}")
print(f"与 sklearn 的预测一致 {(mine == ref).sum()}/{len(ref)}")

# ---- 代码 3 ----
X_toy = np.array([[0.1], [1.0], [1.2], [5.0], [6.0]])
y_toy = np.array([1, 0, 0, 1, 1])
for w in ["uniform", "distance"]:
    m = KNeighborsClassifier(n_neighbors=3, weights=w).fit(X_toy, y_toy)
    print(f"{w:>8}: 预测 {m.predict([[0.0]])[0]}，概率 {m.predict_proba([[0.0]]).round(3)[0]}")

# ---- 代码 4 ----
model = KNeighborsClassifier(n_neighbors=5).fit(Xs_train, y_train)
dist, ind = model.kneighbors(Xs_test[8:9])
print("邻居距离：", dist.round(2)[0])
print("邻居标签：", y_train[ind][0], " 真实标签：", y_test[8])
print("predict_proba：", model.predict_proba(Xs_test[8:9])[0])

# ---- 代码 5 ----
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for name, prep in [("不缩放", None), ("StandardScaler", StandardScaler()),
                   ("MinMaxScaler", MinMaxScaler())]:
    accs = []
    for k in [1, 5, 15]:
        clf = KNeighborsClassifier(n_neighbors=k)
        model = clf if prep is None else make_pipeline(prep, clf)
        accs.append(cross_val_score(model, X, y, cv=cv).mean())
    print(f"{name}: k=1 {accs[0]:.3f}  k=5 {accs[1]:.3f}  k=15 {accs[2]:.3f}")
only = cross_val_score(KNeighborsClassifier(), X[:, [12]], y, cv=cv).mean()
print(f"只用脯氨酸一列 (k=5): {only:.3f}")

# ---- 代码 6 ----
for metric, kw in [("euclidean", {}), ("manhattan", {}), ("minkowski", {"p": 3}),
                   ("chebyshev", {}), ("cosine", {})]:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(metric=metric, **kw))
    print(f"{metric:>9} {str(kw):>8}: {cross_val_score(model, X, y, cv=cv).mean():.3f}")

# ---- 代码 7 ----
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_validate

Xb, yb = load_breast_cancer(return_X_y=True)
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    Xb, yb, test_size=0.25, stratify=yb, random_state=42)
for k in [1, 3, 5, 9, 15, 25, 50, 100, 200]:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    r = cross_validate(model, Xb_train, yb_train, cv=cv, return_train_score=True)
    print(f"k={k:>3}  训练 {r['train_score'].mean():.3f}  验证 {r['test_score'].mean():.3f}")

# ---- 代码 8 ----
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
grid = {"kneighborsclassifier__n_neighbors": list(range(1, 32, 2)),
        "kneighborsclassifier__weights": ["uniform", "distance"]}
search = GridSearchCV(pipe, grid, cv=cv).fit(Xb_train, yb_train)
print(search.best_params_)
print(f"交叉验证 {search.best_score_:.3f}，测试集 {search.score(Xb_test, yb_test):.3f}")

# ---- 代码 9 ----
rng = np.random.default_rng(0)
for d in [3, 15, 16, 50]:
    A = rng.normal(size=(1000, d))
    ya = (A[:, 0] > 0).astype(int)
    m = KNeighborsClassifier().fit(A, ya)
    print(f"{d:>2} 维: auto 选 {m._fit_method}")
m = KNeighborsClassifier(metric="cosine").fit(A[:, :3], ya)
print(f" 3 维 + 余弦距离: auto 选 {m._fit_method}")

# ---- 代码 10 ----
import time

for d in [3, 50]:
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(20000, d)), rng.normal(size=(2000, d))
    ya = (A[:, 0] > 0).astype(int)
    for alg in ["brute", "kd_tree", "ball_tree"]:
        m = KNeighborsClassifier(algorithm=alg).fit(A, ya)
        t0 = time.perf_counter()
        m.predict(B)
        print(f"{d:>2} 维 {alg:>9}: 预测 2000 个样本耗时 {time.perf_counter() - t0:.3f} 秒")

# ---- 代码 11 ----
rng = np.random.default_rng(0)
for d in [1, 2, 10, 100, 1000, 10000]:
    P = rng.uniform(size=(1000, d))
    q = rng.uniform(size=d)
    dist = np.sqrt(((P - q) ** 2).sum(axis=1))
    print(f"d={d:>5}: 最近 / 最远 = {dist.min() / dist.max():.3f}")

# ---- 代码 12 ----
rng = np.random.default_rng(0)
noise = rng.normal(size=(len(Xb), 200))
for n_noise in [0, 20, 50, 100, 200]:
    Xn = np.hstack([Xb, noise[:, :n_noise]])
    model = make_pipeline(StandardScaler(), KNeighborsClassifier())
    print(f"加 {n_noise:>3} 列噪声: {cross_val_score(model, Xn, yb, cv=cv).mean():.3f}")

# ---- 代码 13 ----
from sklearn.neighbors import KNeighborsRegressor

rng = np.random.default_rng(2)
x = rng.uniform(0, 5, size=80)
y_reg = 2 * x + rng.normal(0, 0.5, size=80)
grid = np.linspace(0, 5, 1001).reshape(-1, 1)
for k, w in [(1, "uniform"), (10, "uniform"), (10, "distance")]:
    reg = KNeighborsRegressor(n_neighbors=k, weights=w).fit(x.reshape(-1, 1), y_reg)
    levels = len(np.unique(reg.predict(grid).round(10)))
    pred = " / ".join(f"{v:.2f}" for v in reg.predict([[2.5], [4.9], [8], [100]]))
    print(f"k={k:>2} {w:>8}: 1001 个网格点上有 {levels:>4} 种预测值；x=2.5/4.9/8/100 预测 {pred}")

# ---- 代码 14 ----
from sklearn.datasets import make_classification
from sklearn.metrics import recall_score

Xi, yi = make_classification(n_samples=2000, n_features=5, n_informative=3, n_redundant=0,
                             weights=[0.95, 0.05], class_sep=0.8, random_state=0)
Xi_train, Xi_test, yi_train, yi_test = train_test_split(
    Xi, yi, test_size=0.3, stratify=yi, random_state=0)
print(f"测试集正类 {yi_test.sum()} 个 / {len(yi_test)} 个")
for k in [1, 5, 15, 51]:
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(Xi_train, yi_train)
    p = m.predict(Xi_test)
    print(f"k={k:>2}: 准确率 {(p == yi_test).mean():.3f}，预测为正 {p.sum():>2} 个，召回率 {recall_score(yi_test, p):.3f}")

proba = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)).fit(
    Xi_train, yi_train).predict_proba(Xi_test)[:, 1]
for t in [0.5, 0.2, 0.1]:
    print(f"k=15，阈值 {t}: 召回率 {recall_score(yi_test, (proba >= t).astype(int)):.3f}")
