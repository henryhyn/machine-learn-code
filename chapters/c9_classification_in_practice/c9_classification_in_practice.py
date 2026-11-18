"""C9 分类实战问题：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/c9_classification_in_practice/c9_classification_in_practice.py
"""

# ---- 代码 1 ----
import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=10000, n_features=10, n_informative=5,
                           weights=[0.95], flip_y=0.01, class_sep=1.5, random_state=0)
print(f"正类 {y.sum()} 个，占比 {y.mean():.3f}")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

def evaluate(name, model, X, y, sampler=None):
    """手写 5 折交叉验证；sampler 不为空时，只对每一折的训练部分重采样。"""
    scores = []
    for tr, va in cv.split(X, y):
        X_tr, y_tr = X[tr], y[tr]
        if sampler is not None:
            X_tr, y_tr = sampler(X_tr, y_tr)
        m = clone(model).fit(X_tr, y_tr)
        p, pred = m.predict_proba(X[va])[:, 1], m.predict(X[va])
        scores.append([average_precision_score(y[va], p), roc_auc_score(y[va], p),
                       precision_score(y[va], pred), recall_score(y[va], pred), f1_score(y[va], pred)])
    ap, auc, prec, rec, f1 = np.mean(scores, axis=0)
    print(f"AP {ap:.3f}  ROC-AUC {auc:.3f}  精确率 {prec:.3f}  召回率 {rec:.3f}  F1 {f1:.3f} | {name}")

lr = make_pipeline(StandardScaler(), LogisticRegression())
evaluate("逻辑回归", lr, X, y)

# ---- 代码 2 ----
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

print("balanced 权重 (负类, 正类)：", compute_class_weight("balanced", classes=np.array([0, 1]), y=y).round(3))
a = LogisticRegression(class_weight="balanced").fit(X, y)
b = LogisticRegression().fit(X, y, sample_weight=compute_sample_weight("balanced", y))
print(f"class_weight 与 sample_weight 两种写法的系数最大差 {np.abs(a.coef_ - b.coef_).max():.1e}")

lr_bal = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
evaluate("class_weight='balanced'", lr_bal, X, y)

# ---- 代码 3 ----
from sklearn.utils import resample

def oversample(X, y, random_state=0):
    """随机过采样：把少数类有放回地抽到和多数类一样多。"""
    n_neg = (y == 0).sum()
    X_pos = resample(X[y == 1], replace=True, n_samples=n_neg, random_state=random_state)
    return np.vstack([X[y == 0], X_pos]), np.r_[np.zeros(n_neg, int), np.ones(n_neg, int)]

def undersample(X, y, random_state=0):
    """随机欠采样：把多数类无放回地抽到和少数类一样多。"""
    n_pos = (y == 1).sum()
    X_neg = resample(X[y == 0], replace=False, n_samples=n_pos, random_state=random_state)
    return np.vstack([X_neg, X[y == 1]]), np.r_[np.zeros(n_pos, int), np.ones(n_pos, int)]

evaluate("随机过采样", lr, X, y, sampler=oversample)
evaluate("随机欠采样", lr, X, y, sampler=undersample)

# ---- 代码 4 ----
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)
X_over, y_over = oversample(X, y)                 # 错误：先在全部数据上过采样
evaluate("先过采样，再交叉验证", tree, X_over, y_over)
evaluate("每一折只对训练部分过采样", tree, X, y, sampler=oversample)

# ---- 代码 5 ----
from sklearn.model_selection import FixedThresholdClassifier, TunedThresholdClassifierCV

evaluate(f"阈值固定为正类占比 {y.mean():.3f}", FixedThresholdClassifier(lr, threshold=y.mean()), X, y)
evaluate("按 F1 调阈值", TunedThresholdClassifierCV(lr, scoring="f1"), X, y)
tuned = TunedThresholdClassifierCV(lr, scoring="f1").fit(X, y)
print(f"在全部数据上选出的阈值 {tuned.best_threshold_:.3f}")

# ---- 代码 6 ----
from sklearn.metrics import confusion_matrix, make_scorer

def neg_cost(y_true, y_pred, fn_cost=5, fp_cost=1):
    """漏报一个正类的代价是误报的 5 倍；返回负的总代价，越大越好。"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return -(fn * fn_cost + fp * fp_cost)

by_cost = TunedThresholdClassifierCV(lr, scoring=make_scorer(neg_cost), store_cv_results=True).fit(X, y)
print(f"按代价选出的阈值 {by_cost.best_threshold_:.3f}")
res = by_cost.cv_results_
for t in [0.1, 0.3, 0.5]:
    i = np.argmin(np.abs(res["thresholds"] - t))
    print(f"阈值 {res['thresholds'][i]:.3f}: 每折平均总代价 {-res['scores'][i]:.1f}")
print(f"最优阈值: 每折平均总代价 {-by_cost.best_score_:.1f}")

# ---- 代码 7 ----
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

X_d, y_d = load_digits(return_X_y=True)
for name, clf in [("原生 softmax", LogisticRegression(max_iter=1000)),
                  ("一对其余 OvR", OneVsRestClassifier(LogisticRegression(max_iter=1000))),
                  ("一对一 OvO", OneVsOneClassifier(LogisticRegression(max_iter=1000)))]:
    m = make_pipeline(StandardScaler(), clf).fit(X_d, y_d)
    n_models = len(m[-1].estimators_) if hasattr(m[-1], "estimators_") else 1
    acc = cross_val_score(m, X_d, y_d, cv=5).mean()
    print(f"{name}: 二分类器 {n_models:>2} 个，5 折准确率 {acc:.3f}")

# ---- 代码 8 ----
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

X3, y3 = make_classification(n_samples=3000, n_features=10, n_informative=5, n_classes=3,
                             weights=[0.8, 0.15, 0.05], n_clusters_per_class=1, random_state=0)
pred3 = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression()), X3, y3, cv=cv)
print(f"各类样本数 {np.bincount(y3)}，各类 F1 {f1_score(y3, pred3, average=None).round(3)}")
for avg in ["macro", "weighted", "micro"]:
    print(f"average={avg!r:<10} F1 {f1_score(y3, pred3, average=avg):.3f}")
print(f"准确率 {accuracy_score(y3, pred3):.3f}")

# ---- 代码 9 ----
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

Xm, Ym = make_multilabel_classification(n_samples=2000, n_features=20, n_classes=5, n_labels=2,
                                        allow_unlabeled=False, random_state=0)
print(f"Y 的形状 {Ym.shape}，前两行 {Ym[:2].tolist()}，平均每个样本 {Ym.sum(axis=1).mean():.2f} 个标签")
Xm_tr, Xm_te, Ym_tr, Ym_te = train_test_split(Xm, Ym, test_size=0.3, random_state=0)
for name, m in [("MultiOutput + 逻辑回归", MultiOutputClassifier(make_pipeline(StandardScaler(), LogisticRegression()))),
                ("随机森林 (原生多标签)", RandomForestClassifier(random_state=0))]:
    P = m.fit(Xm_tr, Ym_tr).predict(Xm_te)
    print(f"完全命中 {accuracy_score(Ym_te, P):.3f}  汉明损失 {hamming_loss(Ym_te, P):.3f}  "
          f"F1(samples) {f1_score(Ym_te, P, average='samples'):.3f} | {name}")

# ---- 代码 10 ----
from sklearn.calibration import calibration_curve

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
rf = RandomForestClassifier(random_state=0)
probs = {name: m.fit(X_train, y_train).predict_proba(X_test)[:, 1]
         for name, m in [("逻辑回归", lr), ("逻辑回归 balanced", lr_bal), ("随机森林", rf)]}

edges = np.linspace(0, 1, 6)
for name, p in probs.items():
    frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=5)
    counts = np.histogram(p, bins=edges)[0]
    print(f"{name}：平均预测概率 {p.mean():.3f}，测试集正类占比 {y_test.mean():.3f}")
    for n, mp, fp in zip(counts[counts > 0], mean_pred, frac_pos):
        print(f"  {n:>5} 个样本  平均预测 {mp:.2f}  实际正类比例 {fp:.2f}")

# ---- 代码 11 ----
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

models = {"逻辑回归": lr, "逻辑回归 balanced": lr_bal,
          "逻辑回归 balanced + sigmoid": CalibratedClassifierCV(lr_bal, method="sigmoid"),
          "随机森林": rf,
          "随机森林 + sigmoid": CalibratedClassifierCV(rf, method="sigmoid"),
          "随机森林 + isotonic": CalibratedClassifierCV(rf, method="isotonic")}
for name, m in models.items():
    p = m.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    print(f"Brier {brier_score_loss(y_test, p):.4f}  对数损失 {log_loss(y_test, p):.3f}  "
          f"AP {average_precision_score(y_test, p):.3f}  平均预测 {p.mean():.3f} | {name}")

# ---- 代码 12 ----
def perceptron(X, y, eta=1.0, max_epochs=100):
    """李航《统计学习方法》第 2 章的原始形式；y 取 +1 / -1。"""
    w, b = np.zeros(X.shape[1]), 0.0
    for epoch in range(1, max_epochs + 1):
        errors = 0
        for xi, yi in zip(X, y):
            if yi * (xi @ w + b) <= 0:            # 误分类 (或正好在边界上)
                w, b = w + eta * yi * xi, b + eta * yi
                errors += 1
        if errors == 0:
            break
    return w, b, epoch, errors

X4 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for name, y4 in [("与 (AND)", np.array([-1, -1, -1, 1])), ("异或 (XOR)", np.array([-1, 1, 1, -1]))]:
    w, b, epoch, errors = perceptron(X4, y4)
    print(f"{name}: 跑了 {epoch} 轮，最后一轮还错 {errors} 个，w = {w}，b = {b}")

# ---- 代码 13 ----
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier

rng = np.random.default_rng(0)
X_xor = rng.uniform(-1, 1, size=(1000, 2))
y_xor = (X_xor[:, 0] * X_xor[:, 1] > 0).astype(int)      # 一、三象限为正类
X_moon, y_moon = make_moons(n_samples=1000, noise=0.25, random_state=0)

mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=0)
for dname, (Xd, yd) in [("异或", (X_xor, y_xor)), ("双月牙", (X_moon, y_moon))]:
    for name, m in [("逻辑回归", LogisticRegression()), ("MLP 一个 16 单元隐藏层", mlp)]:
        acc = cross_val_score(make_pipeline(StandardScaler(), m), Xd, yd, cv=5).mean()
        print(f"{dname}  {name}: 5 折准确率 {acc:.3f}")
net = make_pipeline(StandardScaler(), mlp).fit(X_xor, y_xor)[-1]
print(f"各层权重形状 {[c.shape for c in net.coefs_]}，输出层激活 {net.out_activation_!r}，迭代 {net.n_iter_} 轮")

# ---- 代码 14 ----
from sklearn.datasets import load_breast_cancer

X_bc, y_bc = load_breast_cancer(return_X_y=True)
X_bc = StandardScaler().fit_transform(X_bc)
logreg = LogisticRegression(C=1.0, tol=1e-8, max_iter=10000).fit(X_bc, y_bc)
no_hidden = MLPClassifier(hidden_layer_sizes=(), alpha=1.0, solver="lbfgs",
                          max_iter=10000, tol=1e-10, random_state=0).fit(X_bc, y_bc)
print(f"没有隐藏层的 MLP：权重形状 {no_hidden.coefs_[0].shape}，输出层激活 {no_hidden.out_activation_!r}")
print(f"和逻辑回归的系数最大差 {np.abs(logreg.coef_.ravel() - no_hidden.coefs_[0].ravel()).max():.4f}，"
      f"预测概率最大差 {np.abs(logreg.predict_proba(X_bc) - no_hidden.predict_proba(X_bc)).max():.4f}")
