"""D4 异常检测：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/d4_anomaly_detection/d4_anomaly_detection.py
"""

# ---- 代码 1 ----
import numpy as np
from sklearn.covariance import EllipticEnvelope

rng = np.random.default_rng(0)
height = rng.normal(170, 8, size=500)                                  # 身高 (cm)
weight = 0.9 * (height - 170) + 65 + rng.normal(0, 4, size=500)        # 体重 (kg)，和身高强相关
X_hw = np.vstack([np.column_stack([height, weight]),
                  [[186, 52], [155, 80]]])        # 第 500、501 行：高而很轻、矮而很重

z = (X_hw - X_hw.mean(axis=0)) / X_hw.std(axis=0)                      # 普通 z-score
med = np.median(X_hw, axis=0)
mad = np.median(np.abs(X_hw - med), axis=0)
robust_z = 0.6745 * (X_hw - med) / mad                                 # 稳健 z-score (中位数 + MAD)
ee = EllipticEnvelope(random_state=0).fit(X_hw)
md2 = -ee.score_samples(X_hw)                                          # 平方马氏距离
for i in [500, 501]:
    print(f"{X_hw[i]}: z {np.round(z[i], 2)}，稳健 z {np.round(robust_z[i], 2)}，"
          f"平方马氏距离 {md2[i]:.1f} (第 {(md2 >= md2[i]).sum()} 大)")
print("有某一列 |z| > 3 的点:", np.flatnonzero((np.abs(z) > 3).any(axis=1)))
print("平方马氏距离第 3 大:", round(np.sort(md2)[-3], 1))

# ---- 代码 2 ----
from sklearn.datasets import make_blobs
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

dense = make_blobs(400, centers=[[0, 0]], cluster_std=0.3, random_state=0)[0]    # 密集的簇
sparse = make_blobs(100, centers=[[6, 6]], cluster_std=1.5, random_state=1)[0]   # 稀疏的簇
angle = np.random.default_rng(1).uniform(0, 2 * np.pi, size=10)
local = 1.6 * np.column_stack([np.cos(angle), np.sin(angle)])          # 10 个贴着密集簇的局部异常
X_loc = np.vstack([dense, sparse, local])
y_loc = np.r_[np.zeros(500), np.ones(10)].astype(int)

def knn_score(X, k=10):
    """到第 k 个近邻的距离 (第 0 个近邻是自己，所以多取一个)"""
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return dist[:, -1]

lof = LocalOutlierFactor(n_neighbors=20).fit(X_loc)
for name, score in [("kNN 距离", knn_score(X_loc)), ("LOF", -lof.negative_outlier_factor_)]:
    top10 = np.argsort(-score)[:10]
    print(f"{name}: AP {average_precision_score(y_loc, score):.3f}，最可疑的 10 个里 "
          f"局部异常 {(top10 >= 500).sum()} 个、稀疏簇的正常点 {((top10 >= 400) & (top10 < 500)).sum()} 个")
    print(f"    分数中位数：密集簇 {np.median(score[:400]):.2f}，稀疏簇 {np.median(score[400:500]):.2f}，"
          f"局部异常 {np.median(score[500:]):.2f}")

# ---- 代码 3 ----
from sklearn.ensemble import IsolationForest

iso = IsolationForest(random_state=0).fit(X_hw)
print(f"max_samples_ = {iso.max_samples_}，offset_ = {iso.offset_}")

def c(n):
    """n 个点的二叉搜索树里，一次失败查找的平均路径长度"""
    return 2 * (np.log(n - 1) + np.euler_gamma) - 2 * (n - 1) / n

center = np.argmin(((X_hw - X_hw.mean(axis=0)) ** 2).sum(axis=1))     # 离均值最近的点
tallest = np.argmax(X_hw[:500, 0])                                     # 最高的正常人
s_all = -iso.score_samples(X_hw)                                       # 论文里的异常分数 s
for name, i in [("中心点", center), ("最高的人", tallest), ("高而很轻", 500), ("矮而很重", 501)]:
    s = s_all[i]
    print(f"{name} {X_hw[i].round(1)}: s = {s:.3f}，平均路径长度 E(h) = {-c(256) * np.log2(s):.2f}，"
          f"排名 {(s_all >= s).sum()}，predict {iso.predict(X_hw[[i]])[0]:>2}")
print(f"c(256) = {c(256):.2f}")
print("decision_function == score_samples - offset_:",
      np.allclose(iso.decision_function(X_hw), iso.score_samples(X_hw) - iso.offset_))
print("predict 为 -1 的个数:", (iso.predict(X_hw) == -1).sum())

# ---- 代码 4 ----
X_clean = X_loc[:500]                              # 只有正常点的训练数据
X_new = np.array([[0, 0.2], [1.6, 0], [6, 6]])     # 密集簇中心、密集簇旁边、稀疏簇中心

outlier_lof = LocalOutlierFactor(n_neighbors=20)   # novelty=False：离群点检测
print("novelty=False 有 predict 吗？", hasattr(outlier_lof, "predict"))
print("fit_predict 标出的异常:", (outlier_lof.fit_predict(X_loc) == -1).sum())

novelty_lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X_clean)
print("novelty=True 有 fit_predict 吗？", hasattr(novelty_lof, "fit_predict"))
print("新样本的 predict:", novelty_lof.predict(X_new))
print("新样本的 score_samples:", novelty_lof.score_samples(X_new).round(2))

# ---- 代码 5 ----
from sklearn.datasets import make_moons
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

def with_anomalies(X_normal, n_anom, seed):
    """在正常数据的外接矩形 (四周各放宽 1) 里均匀撒 n_anom 个异常点，再一起标准化"""
    rng = np.random.default_rng(seed)
    lo, hi = X_normal.min(axis=0) - 1, X_normal.max(axis=0) + 1
    X = np.vstack([X_normal, rng.uniform(lo, hi, size=(n_anom, X_normal.shape[1]))])
    y = np.r_[np.zeros(len(X_normal)), np.ones(n_anom)].astype(int)
    return StandardScaler().fit_transform(X), y

def detector_scores(X):
    """五种检测器的异常分数，统一成“越大越异常”"""
    return {
        "马氏距离": -EllipticEnvelope(random_state=0).fit(X).score_samples(X),
        "kNN 距离": knn_score(X),
        "LOF": -LocalOutlierFactor(n_neighbors=20).fit(X).negative_outlier_factor_,
        "iForest": -IsolationForest(random_state=0).fit(X).score_samples(X),
        "OCSVM": -OneClassSVM(nu=0.05).fit(X).score_samples(X),
    }

toy_sets = {
    "斜椭圆": make_blobs(500, centers=[[0, 0]], random_state=0)[0] @ np.array([[2, 1.5], [0, 0.6]]),
    "疏密两簇": np.vstack([dense, make_blobs(100, centers=[[5, 5]], cluster_std=2.0, random_state=1)[0]]),
    "月牙": make_moons(n_samples=500, noise=0.06, random_state=0)[0],
}
results = {}
for name, Xn in toy_sets.items():
    res = []
    for seed in range(5):                          # 换 5 次异常点的位置，取平均
        X, y = with_anomalies(Xn, 25, seed)
        res.append([(roc_auc_score(y, s), average_precision_score(y, s)) for s in detector_scores(X).values()])
    results[name] = np.mean(res, axis=0)

print("          ROC-AUC：斜椭圆 疏密两簇 月牙    AP：斜椭圆 疏密两簇 月牙")
for j, method in enumerate(["马氏距离", "kNN 距离", "LOF", "iForest", "OCSVM"]):
    auc = "".join(f"{results[d][j, 0]:>8.2f}" for d in toy_sets)
    ap = "".join(f"{results[d][j, 1]:>8.2f}" for d in toy_sets)
    print(f"{method:<8}{auc}    {ap}")

# ---- 代码 6 ----
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

Xb, yb = load_breast_cancer(return_X_y=True)        # yb：0 恶性，1 良性

def make_bc(seed, n_malignant=20):
    """357 个良性当正常，随机抽 n_malignant 个恶性当异常"""
    rng = np.random.default_rng(seed)
    idx = np.r_[np.flatnonzero(yb == 1), rng.choice(np.flatnonzero(yb == 0), n_malignant, replace=False)]
    return Xb[idx], (yb[idx] == 0).astype(int)

def pca_error(X, n_components=0.9):
    """PCA 重构误差：压到保留 90% 方差的维数，再还原回去"""
    p = PCA(n_components=n_components).fit(X)
    return ((X - p.inverse_transform(p.transform(X))) ** 2).sum(axis=1)

ap = []
for seed in range(5):
    Xbc, ybc = make_bc(seed)
    Xbc_s = StandardScaler().fit_transform(Xbc)
    scores = detector_scores(Xbc_s) | {"PCA 重构误差": pca_error(Xbc_s)}
    ap.append([average_precision_score(ybc, s) for s in scores.values()])
print(f"异常比例 {ybc.mean():.3f} (随机打分的 AP 约等于它)")
for name, m, sd in zip(scores, np.mean(ap, axis=0), np.std(ap, axis=0)):
    print(f"{name:<8} AP {m:.3f} ± {sd:.3f}")

# ---- 代码 7 ----
Xbc, ybc = make_bc(0)                                # 后面固定用这一份
Xbc_s = StandardScaler().fit_transform(Xbc)
for cont in ["auto", 0.01, 0.05, 0.1, 0.2]:
    iso = IsolationForest(contamination=cont, random_state=0).fit(Xbc_s)
    flagged = iso.predict(Xbc_s) == -1
    print(f"contamination={cont!s:>4}: offset_ {iso.offset_:.3f}，标为异常 {flagged.sum():>3} 个，"
          f"其中真异常 {(flagged & (ybc == 1)).sum():>2} 个，"
          f"AP {average_precision_score(ybc, -iso.score_samples(Xbc_s)):.3f}")

# ---- 代码 8 ----
for name, make in [("kNN 距离", knn_score),
                   ("LOF", lambda X: -LocalOutlierFactor(n_neighbors=20).fit(X).negative_outlier_factor_),
                   ("OCSVM", lambda X: -OneClassSVM(nu=0.05).fit(X).score_samples(X)),
                   ("马氏距离", lambda X: -EllipticEnvelope(random_state=0).fit(X).score_samples(X)),
                   ("iForest", lambda X: -IsolationForest(random_state=0).fit(X).score_samples(X))]:
    raw, scaled = make(Xbc), make(Xbc_s)
    print(f"{name:<8} 不缩放 AP {average_precision_score(ybc, raw):.3f}，标准化后 {average_precision_score(ybc, scaled):.3f}，"
          f"排序完全相同: {(np.argsort(raw) == np.argsort(scaled)).all()}")

# ---- 代码 9 ----
rng = np.random.default_rng(0)
X_mask = np.vstack([rng.normal(0, 1, size=(2000, 2)),       # 2000 个正常点
                    rng.normal(4, 0.2, size=(50, 2))])      # 50 个抱团的异常点
y_mask = np.r_[np.zeros(2000), np.ones(50)].astype(int)

for k in [20, 50, 100]:
    s = -LocalOutlierFactor(n_neighbors=k).fit(X_mask).negative_outlier_factor_
    print(f"LOF n_neighbors={k:<3}: AP {average_precision_score(y_mask, s):.3f}")
for m in [64, 256, 1.0]:
    s = -IsolationForest(max_samples=m, random_state=0).fit(X_mask).score_samples(X_mask)
    print(f"iForest max_samples={m!s:<4}: AP {average_precision_score(y_mask, s):.3f}")

# ---- 代码 10 ----
rng = np.random.default_rng(0)
angle = rng.uniform(0, 2 * np.pi, size=30)
X_rel = np.vstack([rng.normal(size=(1000, 2)),                            # 正常点：二维标准正态
                   4 * np.column_stack([np.cos(angle), np.sin(angle)])])  # 异常点：半径为 4 的圆上
y_rel = np.r_[np.zeros(1000), np.ones(30)].astype(int)     # 30 个异常只在前 2 维上异常

for n_noise in [0, 8, 48, 198]:
    X = np.hstack([X_rel, rng.normal(size=(len(X_rel), n_noise))])   # 补上无关的噪声维度
    aps = [average_precision_score(y_rel, s) for s in
           [knn_score(X), -LocalOutlierFactor(n_neighbors=20).fit(X).negative_outlier_factor_,
            -IsolationForest(random_state=0).fit(X).score_samples(X)]]
    print(f"{2 + n_noise:>3} 维: kNN 距离 {aps[0]:.3f}，LOF {aps[1]:.3f}，iForest {aps[2]:.3f}")

# ---- 代码 11 ----
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

X_tr, X_te, y_tr, y_te = train_test_split(Xbc, ybc, test_size=0.5, stratify=ybc, random_state=0)
print(f"训练集 {len(y_tr)} 个 (异常 {y_tr.sum()} 个)，测试集 {len(y_te)} 个 (异常 {y_te.sum()} 个)")
iso = IsolationForest(random_state=0).fit(X_tr)                       # 无监督：不看标签
print(f"iForest (不用标签): AP {average_precision_score(y_te, -iso.score_samples(X_te)):.3f}")

neg, pos = np.flatnonzero(y_tr == 0), np.flatnonzero(y_tr == 1)
for n_labeled in [1, 2, 5, 10]:
    keep = np.r_[neg, pos[:n_labeled]]                                # 只知道 n_labeled 个异常的标签
    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
    lr.fit(X_tr[keep], y_tr[keep])
    print(f"逻辑回归，{n_labeled:>2} 个异常标签: AP {average_precision_score(y_te, lr.predict_proba(X_te)[:, 1]):.3f}")
hgb = HistGradientBoostingClassifier(class_weight="balanced", random_state=0).fit(X_tr, y_tr)
print(f"HistGradientBoosting，10 个异常标签: AP {average_precision_score(y_te, hgb.predict_proba(X_te)[:, 1]):.3f}")

# ---- 代码 12 ----
from sklearn.cluster import HDBSCAN
from sklearn.metrics import precision_score, recall_score
from sklearn.mixture import GaussianMixture

for name, Xn in toy_sets.items():
    X, y = with_anomalies(Xn, 25, 0)
    noise = HDBSCAN(min_cluster_size=15, copy=True).fit_predict(X) == -1
    gmm = -GaussianMixture(n_components=3, random_state=0).fit(X).score_samples(X)   # 负对数似然
    print(f"{name:<6} HDBSCAN 噪声点 {noise.sum():>3} 个，精确率 {precision_score(y, noise):.2f}，"
          f"召回率 {recall_score(y, noise):.2f}；GMM AP {average_precision_score(y, gmm):.2f}，"
          f"iForest AP {average_precision_score(y, -IsolationForest(random_state=0).fit(X).score_samples(X)):.2f}")
