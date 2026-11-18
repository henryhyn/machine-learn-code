"""C6 支持向量机：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/c6_svm/c6_svm.py
"""

# ---- 代码 1 ----
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC

X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.0, random_state=3)
for name, model in [("感知机", Perceptron(random_state=0)),
                    ("逻辑回归", LogisticRegression()),
                    ("SVM", SVC(kernel="linear", C=1000))]:
    model.fit(X, y)
    w, b = model.coef_.ravel(), model.intercept_[0]
    dist = np.abs(X @ w + b) / np.linalg.norm(w)      # 每个点到边界的距离
    print(f"{name}: 训练准确率 {model.score(X, y):.2f}，离边界最近的点 {dist.min():.3f}")

# ---- 代码 2 ----
svm = SVC(kernel="linear", C=1000).fit(X, y)
print(f"支持向量 {len(svm.support_)} 个 (共 {len(X)} 个样本)，下标 {svm.support_}")
print(f"它们的 decision_function: {svm.decision_function(X[svm.support_]).round(3)}")
print(f"间隔宽度 2/||w|| = {2 / np.linalg.norm(svm.coef_):.3f}")

sv_only = SVC(kernel="linear", C=1000).fit(X[svm.support_], y[svm.support_])
print(f"全部样本:     w = {svm.coef_.ravel().round(3)}, b = {svm.intercept_[0]:.3f}")
print(f"只用支持向量: w = {sv_only.coef_.ravel().round(3)}, b = {sv_only.intercept_[0]:.3f}")

# ---- 代码 3 ----
from sklearn.model_selection import cross_val_score

Xn, yn = make_blobs(n_samples=200, centers=2, cluster_std=2.5, random_state=3)
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    m = SVC(kernel="linear", C=C).fit(Xn, yn)
    cv = cross_val_score(SVC(kernel="linear", C=C), Xn, yn, cv=5).mean()
    margin = 2 / np.linalg.norm(m.coef_)
    print(f"C={C:<6} 间隔宽度 {margin:5.2f}  支持向量 {len(m.support_):3d}  "
          f"训练 {m.score(Xn, yn):.3f}  交叉验证 {cv:.3f}")

# ---- 代码 4 ----
z = np.array([-2, -1, 0, 0.5, 1, 2, 3])
print("y·f(x)   " + "".join(f"{v:>7}" for v in z))
print("合页损失 " + "".join(f"{v:7.3f}" for v in np.maximum(0, 1 - z)))
print("对数损失 " + "".join(f"{v:7.3f}" for v in np.log2(1 + np.exp(-z))))

# ---- 代码 5 ----
from sklearn.datasets import make_circles

Xc, yc = make_circles(n_samples=300, noise=0.1, factor=0.4, random_state=0)
for name, model in [("线性核", SVC(kernel="linear")),
                    ("多项式核 (degree=2)", SVC(kernel="poly", degree=2)),
                    ("RBF 核", SVC(kernel="rbf"))]:
    print(f"{name}: 交叉验证准确率 {cross_val_score(model, Xc, yc, cv=5).mean():.3f}")

Xc_lifted = np.c_[Xc, (Xc ** 2).sum(axis=1)]          # 手工加一列 x1² + x2²
acc = cross_val_score(SVC(kernel="linear"), Xc_lifted, yc, cv=5).mean()
print(f"线性核 + 手工特征: 交叉验证准确率 {acc:.3f}")

# ---- 代码 6 ----
rbf = SVC(kernel="rbf", gamma=1.0).fit(Xc, yc)
x_new = np.array([[0.1, 0.2], [0.9, -0.5]])
sq_dist = ((x_new[:, None, :] - rbf.support_vectors_[None, :, :]) ** 2).sum(axis=2)
manual = np.exp(-1.0 * sq_dist) @ rbf.dual_coef_.ravel() + rbf.intercept_[0]
print(f"支持向量 {len(rbf.support_)} 个")
print(f"按对偶公式计算: {manual.round(4)}")
print(f"decision_function: {rbf.decision_function(x_new).round(4)}")

# ---- 代码 7 ----
from sklearn.datasets import make_moons

Xm, ym = make_moons(n_samples=300, noise=0.25, random_state=0)
for g in [0.01, 0.1, 1, 10, 100]:
    m = SVC(gamma=g).fit(Xm, ym)
    cv = cross_val_score(SVC(gamma=g), Xm, ym, cv=5).mean()
    print(f"gamma={g:<5} 影响半径 {np.sqrt(np.log(2) / g):6.3f}  支持向量 {len(m.support_):3d}  "
          f"训练 {m.score(Xm, ym):.3f}  交叉验证 {cv:.3f}")

# ---- 代码 8 ----
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
print(f"各列方差：最小 {X_train.var(axis=0).min():.1e}，最大 {X_train.var(axis=0).max():.1e}")
for kernel in ["rbf", "linear"]:
    raw = SVC(kernel=kernel).fit(X_train, y_train)
    scaled = make_pipeline(StandardScaler(), SVC(kernel=kernel)).fit(X_train, y_train)
    print(f"{kernel:>6}：不缩放 {raw.score(X_test, y_test):.3f} (支持向量 {len(raw.support_)})，"
          f"标准化后 {scaled.score(X_test, y_test):.3f} (支持向量 {len(scaled[-1].support_)})")

# ---- 代码 9 ----
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

models = {"SVC(kernel='linear')": SVC(kernel="linear"),
          "LinearSVC()": LinearSVC(),
          "LinearSVC(loss='hinge')": LinearSVC(loss="hinge", max_iter=10000),
          "SGDClassifier()": SGDClassifier(random_state=0),
          "LogisticRegression()": LogisticRegression()}
directions = {}
for name, model in models.items():
    pipe = make_pipeline(StandardScaler(), model).fit(X_train, y_train)
    w = pipe[-1].coef_.ravel()
    directions[name] = w / np.linalg.norm(w)
    cos = directions[name] @ directions["SVC(kernel='linear')"]
    print(f"{name:<24} 测试集 {pipe.score(X_test, y_test):.3f}  与 SVC 的 w 夹角余弦 {cos:.3f}")

# ---- 代码 10 ----
from sklearn.calibration import CalibratedClassifierCV

svc = make_pipeline(StandardScaler(), SVC()).fit(X_train, y_train)
print(f"有 predict_proba 吗：{hasattr(svc, 'predict_proba')}")
print(f"decision_function: {svc.decision_function(X_test[:5]).round(3)}")
print(f"predict:           {svc.predict(X_test[:5])}")

cal = CalibratedClassifierCV(make_pipeline(StandardScaler(), SVC()), ensemble=False)
cal.fit(X_train, y_train)
print(f"校准后 P(良性):    {cal.predict_proba(X_test[:5])[:, 1].round(3)}")

# ---- 代码 11 ----
from sklearn.datasets import load_digits

X_d, y_d = load_digits(return_X_y=True)
ovo = SVC(decision_function_shape="ovo").fit(X_d, y_d)
print(f"SVC 一对一的决策值: {ovo.decision_function(X_d[:1]).shape}")
print(f"SVC 默认 (ovr 形状): {SVC().fit(X_d, y_d).decision_function(X_d[:1]).shape}")
lin = make_pipeline(StandardScaler(), LinearSVC()).fit(X_d, y_d)
print(f"LinearSVC 的 coef_: {lin[-1].coef_.shape}")

# ---- 代码 12 ----
from sklearn.svm import SVR

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 6, 120)).reshape(-1, 1)
t = np.sin(x).ravel() + rng.normal(0, 0.1, 120)       # 噪声标准差 0.1
for eps in [0.01, 0.1, 0.2, 0.5]:
    r = SVR(epsilon=eps).fit(x, t)
    inside = (np.abs(t - r.predict(x)) < eps).mean()
    rmse = np.sqrt(((r.predict(x) - np.sin(x).ravel()) ** 2).mean())
    print(f"epsilon={eps:<4} 支持向量 {len(r.support_):3d}/120  落在管道内 {inside:.0%}  "
          f"与真实 sin 的 RMSE {rmse:.3f}")

# ---- 代码 13 ----
Xm2, ym2 = make_moons(n_samples=300, noise=0.4, random_state=0)
for C in [0.1, 1, 10, 100, 1000, 10000]:
    m = SVC(C=C, gamma=5).fit(Xm2, ym2)
    cv = cross_val_score(SVC(C=C, gamma=5), Xm2, ym2, cv=5).mean()
    print(f"C={C:<6} 支持向量 {len(m.support_):3d}  训练 {m.score(Xm2, ym2):.3f}  交叉验证 {cv:.3f}")
