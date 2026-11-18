"""B3 模型评估：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/b3_evaluation/b3_evaluation.py
"""

# ---- 代码 1 ----
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
knn.fit(X_train, y_train)
print(f"训练集准确率 {knn.score(X_train, y_train):.3f}，测试集准确率 {knn.score(X_test, y_test):.3f}")

# ---- 代码 2 ----
from sklearn.linear_model import LogisticRegression

scores = []
for seed in range(10):
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xa, ya)
    scores.append(lr.score(Xb, yb))
print(np.round(scores, 3))
print(f"最低 {min(scores):.3f}，最高 {max(scores):.3f}")

# ---- 代码 3 ----
from sklearn.model_selection import cross_val_score, cross_validate

lr = make_pipeline(StandardScaler(), LogisticRegression())
s = cross_val_score(lr, X_train, y_train, cv=5)
print(np.round(s, 3), f"均值 {s.mean():.3f} ± {s.std():.3f}")

r = cross_validate(lr, X_train, y_train, cv=5,
                   scoring=["accuracy", "recall", "roc_auc"], return_train_score=True)
print(sorted(r.keys()))
print(f"训练 {r['train_accuracy'].mean():.3f}，验证 {r['test_accuracy'].mean():.3f}，验证 AUC {r['test_roc_auc'].mean():.3f}")

# ---- 代码 4 ----
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, StratifiedKFold

Xi, yi = load_iris(return_X_y=True)
print(yi[:5], yi[50:55], yi[100:105])
model = LogisticRegression(max_iter=1000)
print("KFold(3)           ", cross_val_score(model, Xi, yi, cv=KFold(3)))
print("StratifiedKFold(3) ", cross_val_score(model, Xi, yi, cv=StratifiedKFold(3)).round(2))

# ---- 代码 5 ----
from sklearn.model_selection import GroupKFold

rng = np.random.default_rng(0)
n_patients, per_patient = 100, 10
centers = rng.normal(0, 1, size=(n_patients, 5))
labels = rng.integers(0, 2, size=n_patients)      # 标签和特征无关
Xg = np.repeat(centers, per_patient, axis=0) + rng.normal(0, 0.1, size=(n_patients * per_patient, 5))
yg = np.repeat(labels, per_patient)
groups = np.repeat(np.arange(n_patients), per_patient)

nn1 = KNeighborsClassifier(n_neighbors=1)
kf = cross_val_score(nn1, Xg, yg, cv=KFold(5, shuffle=True, random_state=0))
gkf = cross_val_score(nn1, Xg, yg, cv=GroupKFold(5), groups=groups)
print(f"KFold {kf.mean():.3f}，GroupKFold {gkf.mean():.3f}")

# ---- 代码 6 ----
from sklearn.model_selection import TimeSeriesSplit

for train_idx, test_idx in TimeSeriesSplit(n_splits=3).split(np.arange(10)):
    print("训练", train_idx, "验证", test_idx)

# ---- 代码 7 ----
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

Xc, yc = make_classification(n_samples=10000, n_features=10, n_informative=5,
                             weights=[0.95], flip_y=0.01, class_sep=1.5, random_state=0)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.3, stratify=yc, random_state=0)
print(f"正类占比 {yc.mean():.3f}")

dummy = DummyClassifier().fit(Xc_train, yc_train)
clf = LogisticRegression().fit(Xc_train, yc_train)
for name, m in [("多数类基线", dummy), ("逻辑回归", clf)]:
    pred = m.predict(Xc_test)
    print(f"{name}: 准确率 {accuracy_score(yc_test, pred):.3f}")
    print(confusion_matrix(yc_test, pred))

# ---- 代码 8 ----
from sklearn.metrics import classification_report

print(classification_report(yc_test, clf.predict(Xc_test), digits=3))

# ---- 代码 9 ----
from sklearn.metrics import precision_score, recall_score, f1_score

proba = clf.predict_proba(Xc_test)[:, 1]
for t in [0.5, 0.3, 0.2, 0.1]:
    pred = (proba >= t).astype(int)
    print(f"阈值 {t}: 精确率 {precision_score(yc_test, pred):.3f}，"
          f"召回率 {recall_score(yc_test, pred):.3f}，F1 {f1_score(yc_test, pred):.3f}")

# ---- 代码 10 ----
from sklearn.metrics import roc_auc_score, average_precision_score

for name, m in [("多数类基线", dummy), ("逻辑回归", clf)]:
    p = m.predict_proba(Xc_test)[:, 1]
    print(f"{name}: ROC-AUC {roc_auc_score(yc_test, p):.3f}，PR-AUC (AP) {average_precision_score(yc_test, p):.3f}")

# ---- 代码 11 ----
Xb, yb = make_classification(n_samples=40000, n_features=10, n_informative=5,
                             class_sep=1.0, random_state=1)
Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.5, random_state=1)
pb = LogisticRegression().fit(Xb_train, yb_train).predict_proba(Xb_test)[:, 1]

neg = np.flatnonzero(yb_test == 0)
pos = np.flatnonzero(yb_test == 1)
rng = np.random.default_rng(0)
for rate in [0.5, 0.1, 0.01]:
    n_pos = int(len(neg) * rate / (1 - rate))
    idx = np.concatenate([neg, rng.choice(pos, size=n_pos, replace=False)])
    print(f"正类占比 {rate:<4}: ROC-AUC {roc_auc_score(yb_test[idx], pb[idx]):.3f}，"
          f"AP {average_precision_score(yb_test[idx], pb[idx]):.3f}")

# ---- 代码 12 ----
from sklearn.model_selection import cross_val_predict

y_mal_train, y_mal_test = 1 - y_train, 1 - y_test     # 1 = 恶性
cancer = make_pipeline(StandardScaler(), LogisticRegression())
oof = cross_val_predict(cancer, X_train, y_mal_train, cv=5, method="predict_proba")[:, 1]

def cost(y_true, pred, fn_cost=20, fp_cost=1):
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return fn * fn_cost + fp * fp_cost

thresholds = np.arange(0.05, 0.96, 0.05)
best_t = thresholds[np.argmin([cost(y_mal_train, (oof >= t).astype(int)) for t in thresholds])]
print(f"按训练集交叉验证选出的阈值 {best_t:.2f}")

p_test = cancer.fit(X_train, y_mal_train).predict_proba(X_test)[:, 1]
for t in [0.5, best_t]:
    pred = (p_test >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_mal_test, pred).ravel()
    print(f"阈值 {t:.2f}: 漏诊 {fn}，误诊 {fp}，总代价 {cost(y_mal_test, pred)}")

# ---- 代码 13 ----
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

Xd, yd = load_diabetes(return_X_y=True)
Xd_train, Xd_test, yd_train, yd_test = train_test_split(Xd, yd, test_size=0.25, random_state=42)
print(f"目标值范围 {yd.min():.0f}–{yd.max():.0f}，均值 {yd.mean():.1f}")
for name, m in [("均值基线", DummyRegressor()),
                ("线性回归", LinearRegression()),
                ("1 近邻", KNeighborsRegressor(n_neighbors=1))]:
    pred = m.fit(Xd_train, yd_train).predict(Xd_test)
    print(f"{name}: MAE {mean_absolute_error(yd_test, pred):.1f}，"
          f"MSE {mean_squared_error(yd_test, pred):.0f}，"
          f"RMSE {root_mean_squared_error(yd_test, pred):.1f}，R² {r2_score(yd_test, pred):.3f}")

# ---- 代码 14 ----
print(cross_val_score(LinearRegression(), Xd_train, yd_train, cv=5,
                      scoring="neg_root_mean_squared_error").round(1))

# ---- 代码 15 ----
from sklearn.metrics import mean_absolute_percentage_error

y_true = np.array([100.0, 80.0, 120.0, 0.5])
y_a = np.array([110.0, 70.0, 130.0, 2.0])
y_b = np.array([150.0, 30.0, 170.0, 0.5])
for name, p in [("A", y_a), ("B", y_b)]:
    print(f"预测 {name}: MAE {mean_absolute_error(y_true, p):.1f}，MAPE {mean_absolute_percentage_error(y_true, p):.2f}")
print(f"真实值为 0 时：MAPE {mean_absolute_percentage_error([0.0, 100.0], [1.0, 100.0]):.3g}")

# ---- 代码 16 ----
rng = np.random.default_rng(2)
Xn = rng.normal(size=(200, 20))
yn = rng.integers(0, 2, size=200)          # 纯噪声，任何模型的真实准确率都是 0.5
Xn_train, Xn_test, yn_train, yn_test = train_test_split(Xn, yn, test_size=0.25, random_state=0)

results = []
for k in range(1, 51):
    for w in ["uniform", "distance"]:
        m = KNeighborsClassifier(n_neighbors=k, weights=w).fit(Xn_train, yn_train)
        results.append((m.score(Xn_test, yn_test), k, w))
best_score, best_k, best_w = max(results)
print(f"试了 {len(results)} 组超参数，测试集上最好的一组 {best_score:.3f}")

Xn_new = rng.normal(size=(2000, 20))
yn_new = rng.integers(0, 2, size=2000)
best = KNeighborsClassifier(n_neighbors=best_k, weights=best_w).fit(Xn_train, yn_train)
print(f"同一组超参数在新数据上 {best.score(Xn_new, yn_new):.3f}")
