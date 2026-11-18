"""D1 聚类：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/d1_clustering/d1_clustering.py
"""

# ---- 代码 1 ----
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

wine = load_wine(as_frame=True)
X, y = wine.data, wine.target          # y 只在事后检验时用，聚类时不给
print(X.std().sort_values(ascending=False).round(1).head(3).to_string())

km_raw = KMeans(n_clusters=3, random_state=0).fit(X)
X_std = StandardScaler().fit_transform(X)
km_std = KMeans(n_clusters=3, random_state=0).fit(X_std)
centers = pd.DataFrame(km_raw.cluster_centers_, columns=X.columns)
print(centers[["proline", "magnesium", "alcohol", "flavanoids"]].round(1).to_string())
print(f"ARI：不缩放 {adjusted_rand_score(y, km_raw.labels_):.3f}，"
      f"标准化后 {adjusted_rand_score(y, km_std.labels_):.3f}")

# ---- 代码 2 ----
profile = X.groupby(km_std.labels_)[["alcohol", "flavanoids", "color_intensity", "proline"]].mean()
profile.insert(0, "样本数", np.bincount(km_std.labels_))
print(profile.round(2).to_string())

# ---- 代码 3 ----
from sklearn.datasets import make_blobs

Xb, yb = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=7)

def lloyd(X, k, seed, max_iter=100):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), size=k, replace=False)]   # 随机挑 k 个样本当初始中心
    history = []
    for _ in range(max_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)   # 每个点到每个中心的平方距离
        labels = d2.argmin(axis=1)                                      # 分配：归入最近的中心
        history.append(f"{d2.min(axis=1).sum():.1f}")                  # 当前的簇内平方和
        new = np.array([X[labels == j].mean(axis=0) for j in range(k)]) # 更新：中心移到簇的均值
        if np.allclose(new, centers):
            break
        centers = new
    return labels, history

for seed in [0, 7]:
    _, history = lloyd(Xb, 4, seed)
    print(f"种子 {seed}：" + " → ".join(history))

# ---- 代码 4 ----
for init in ["random", "k-means++"]:
    runs = [KMeans(n_clusters=4, init=init, n_init=1, random_state=s).fit(Xb).inertia_
            for s in range(20)]
    bad = sum(r > min(runs) + 1 for r in runs)
    print(f"{init:>9}：最好 {min(runs):.1f}，最差 {max(runs):.1f}，20 次里 {bad} 次没到最好")

# ---- 代码 5 ----
from sklearn.metrics import silhouette_score

def scan_k(X, ks):
    for k in ks:
        km = KMeans(n_clusters=k, random_state=0).fit(X)
        print(f"  k={k}  inertia {km.inertia_:8.1f}  轮廓系数 {silhouette_score(X, km.labels_):.3f}")

print("make_blobs (真实 4 簇)：")
scan_k(Xb, range(2, 8))
print("葡萄酒 (标准化，真实 3 个品种)：")
scan_k(X_std, range(2, 7))

# ---- 代码 6 ----
from sklearn.datasets import make_moons

Xm, ym = make_moons(n_samples=300, noise=0.06, random_state=0)
print(f"月牙：ARI {adjusted_rand_score(ym, KMeans(n_clusters=2, random_state=0).fit_predict(Xm)):.3f}")

Xv, yv = make_blobs(n_samples=[500, 50, 50], centers=[[0, 0], [4, 0], [0, 4]],
                    cluster_std=[1.5, 0.5, 0.5], random_state=0)
lab = KMeans(n_clusters=3, random_state=0).fit_predict(Xv)
print(f"大小悬殊：真实簇大小 {np.bincount(yv)}，K-Means 簇大小 {np.bincount(lab)}，"
      f"ARI {adjusted_rand_score(yv, lab):.3f}")

rng = np.random.default_rng(0)
Xo = np.vstack([Xb, rng.uniform(-40, 40, size=(10, 2))])   # 再加 10 个散落各处的离群点
lab = KMeans(n_clusters=4, random_state=0).fit_predict(Xo)
print(f"离群点：簇大小 {np.bincount(lab)}，原 300 个点的 ARI {adjusted_rand_score(yb, lab[:300]):.3f}")

# ---- 代码 7 ----
from sklearn.cluster import AgglomerativeClustering

for linkage in ["ward", "complete", "average", "single"]:
    moons = AgglomerativeClustering(n_clusters=2, linkage=linkage).fit_predict(Xm)
    outl = AgglomerativeClustering(n_clusters=4, linkage=linkage).fit_predict(Xo)
    print(f"{linkage:>8}：月牙 ARI {adjusted_rand_score(ym, moons):.3f}  "
          f"带离群点的 4 簇 簇大小 {np.bincount(outl)}")

# ---- 代码 8 ----
from sklearn.cluster import DBSCAN, HDBSCAN

for eps in [0.05, 0.1, 0.2, 0.3]:
    lab = DBSCAN(eps=eps, min_samples=5).fit_predict(Xm)
    print(f"DBSCAN eps={eps:<4}：{lab.max() + 1:>2} 个簇，噪声点 {(lab == -1).sum():>3}，"
          f"ARI {adjusted_rand_score(ym, lab):.3f}")
lab = HDBSCAN(min_cluster_size=15, copy=True).fit_predict(Xm)
print(f"HDBSCAN         ：{lab.max() + 1:>2} 个簇，噪声点 {(lab == -1).sum():>3}，"
      f"ARI {adjusted_rand_score(ym, lab):.3f}")
lab = HDBSCAN(min_cluster_size=15, copy=True).fit_predict(Xo)
print(f"HDBSCAN 带离群点：{lab.max() + 1} 个簇，噪声点 {(lab == -1).sum()}，"
      f"原 300 个点的 ARI {adjusted_rand_score(yb, lab[:300]):.3f}")

# ---- 代码 9 ----
from sklearn.mixture import GaussianMixture

Xa, ya = make_blobs(n_samples=600, centers=3, random_state=170)
Xa = Xa @ np.array([[0.6, -0.6], [-0.4, 0.8]])    # 线性变换，把圆形的簇拉成斜椭圆
print(f"K-Means                ARI {adjusted_rand_score(ya, KMeans(n_clusters=3, random_state=0).fit_predict(Xa)):.3f}")
for cov in ["full", "tied", "diag", "spherical"]:
    gmm = GaussianMixture(n_components=3, covariance_type=cov, random_state=0).fit(Xa)
    print(f"GMM {cov:<9}  BIC {gmm.bic(Xa):7.1f}  ARI {adjusted_rand_score(ya, gmm.predict(Xa)):.3f}")

# ---- 代码 10 ----
bics = [GaussianMixture(n_components=k, random_state=0).fit(Xa).bic(Xa) for k in range(1, 7)]
print("BIC：" + "  ".join(f"k={k} {b:.1f}" for k, b in zip(range(1, 7), bics)))

gmm = GaussianMixture(n_components=3, random_state=0).fit(Xv)
proba = gmm.predict_proba(Xv)
print(f"大小悬殊的数据：GMM ARI {adjusted_rand_score(yv, gmm.predict(Xv)):.3f}，"
      f"最大概率不到 0.9 的点 {(proba.max(axis=1) < 0.9).sum()} 个")
print(np.round(proba[proba.max(axis=1).argsort()[:3]], 3))

# ---- 代码 11 ----
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

for name, lab in [("真实划分 (两个月牙)", ym),
                  ("K-Means 的划分", KMeans(n_clusters=2, random_state=0).fit_predict(Xm))]:
    print(f"{name}：轮廓 {silhouette_score(Xm, lab):.3f}  "
          f"CH {calinski_harabasz_score(Xm, lab):.1f}  DB {davies_bouldin_score(Xm, lab):.3f}")

# ---- 代码 12 ----
from sklearn.metrics import normalized_mutual_info_score

models = {
    "K-Means": KMeans(n_clusters=3, random_state=0),
    "层次 (ward)": AgglomerativeClustering(n_clusters=3),
    "GMM": GaussianMixture(n_components=3, random_state=0),
    "HDBSCAN": HDBSCAN(min_cluster_size=10, copy=True),
}
rows = []
for name, model in models.items():
    lab = model.fit_predict(X_std)
    rows.append({"算法": name, "簇数": lab.max() + 1, "噪声": (lab == -1).sum(),
                 "轮廓": silhouette_score(X_std, lab), "CH": calinski_harabasz_score(X_std, lab),
                 "DB": davies_bouldin_score(X_std, lab),
                 "ARI": adjusted_rand_score(y, lab), "NMI": normalized_mutual_info_score(y, lab)})
print(pd.DataFrame(rows).round(3).to_string(index=False))
