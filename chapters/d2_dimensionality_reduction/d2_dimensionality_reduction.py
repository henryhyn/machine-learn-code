"""D2 降维：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/d2_dimensionality_reduction/d2_dimensionality_reduction.py
"""

# ---- 代码 1 ----
import numpy as np

rng = np.random.default_rng(0)
X2 = rng.multivariate_normal([0, 0], [[3, 2], [2, 2]], size=500)
Xc = X2 - X2.mean(axis=0)                      # 先中心化

for deg in [0, 38, 60, 90, 128]:
    u = np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])   # 单位方向
    z = Xc @ u                                 # 每个点在 u 上的坐标
    var = z.var()                              # 投影后的方差
    err = ((Xc - np.outer(z, u)) ** 2).sum(axis=1).mean()   # 平均重构误差
    print(f"方向 {deg:>3}°: 投影方差 {var:.3f}，重构误差 {err:.3f}，两者之和 {var + err:.3f}")

# ---- 代码 2 ----
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True, as_frame=True)
Xs = StandardScaler().fit_transform(X)            # 178 × 13，每列均值 0、方差 1

C = np.cov(Xs, rowvar=False)                      # 13 × 13 协方差矩阵
eigvals, eigvecs = np.linalg.eigh(C)              # 对称矩阵的特征分解，特征值从小到大
eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
U, S, Vt = np.linalg.svd(Xs - Xs.mean(axis=0), full_matrices=False)

pca = PCA().fit(Xs)
print("协方差矩阵的特征值:", np.round(eigvals[:4], 3))
print("奇异值² / (n - 1): ", np.round(S[:4] ** 2 / (len(Xs) - 1), 3))
print("explained_variance_:", np.round(pca.explained_variance_[:4], 3))
print("特征向量与 components_ 点积的绝对值:",
      np.round(np.abs((eigvecs[:, :4] * pca.components_[:4].T).sum(axis=0)), 3))

# ---- 代码 3 ----
ratio = pca.explained_variance_ratio_
cum = np.cumsum(ratio)
for k in range(8):
    print(f"PC{k + 1}: 解释方差比 {ratio[k]:.3f}，累计 {cum[k]:.3f}")
print("PC1 载荷最大的 3 个特征:",
      list(X.columns[np.argsort(-np.abs(pca.components_[0]))[:3]]))

# ---- 代码 4 ----
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

Xd, yd = load_digits(return_X_y=True)             # 1797 张 8×8 手写数字，64 个像素
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    Xd, yd, test_size=0.25, stratify=yd, random_state=42)

cum = np.cumsum(PCA().fit(Xd_train).explained_variance_ratio_)
for k in [2, 5, 10, 20, 30, 40]:
    p = PCA(n_components=k).fit(Xd_train)
    recon = p.inverse_transform(p.transform(Xd_test))       # 压到 k 维再还原回 64 维
    mse = ((Xd_test - recon) ** 2).mean()
    print(f"k={k:>2}: 累计解释方差 {cum[k - 1]:.3f}，测试集重构误差 (每像素 MSE) {mse:.3f}")

for nc in [0.90, 0.95, "mle"]:
    print(f"n_components={nc!r}: 保留 {PCA(n_components=nc).fit(Xd_train).n_components_} 个主成分")

# ---- 代码 5 ----
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

pipe = Pipeline([("pca", PCA()), ("knn", KNeighborsClassifier())])
search = GridSearchCV(pipe, {"pca__n_components": [2, 5, 10, 15, 20, 30, 40, None]}, cv=5)
search.fit(Xd_train, yd_train)
for k, s in zip(search.cv_results_["param_pca__n_components"], search.cv_results_["mean_test_score"]):
    print(f"n_components={str(k):>4}: 交叉验证准确率 {s:.3f}")
print(f"选中 {search.best_params_['pca__n_components']}，测试集准确率 {search.score(Xd_test, yd_test):.3f}")
knn_raw = KNeighborsClassifier().fit(Xd_train, yd_train)
print(f"不降维的 kNN，测试集准确率 {knn_raw.score(Xd_test, yd_test):.3f}")

# ---- 代码 6 ----
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

for name, Z in [("原始 64 维", Xd), ("PCA 20 维", PCA(n_components=20).fit_transform(Xd))]:
    labels = KMeans(n_clusters=10, n_init=10, random_state=0).fit_predict(Z)
    print(f"{name}: K-Means 与真实数字的 ARI {adjusted_rand_score(yd, labels):.3f}")

# ---- 代码 7 ----
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score

def separation(Z, y):
    """类中心两两之间的平均距离 ÷ 样本到本类中心的平均距离，越大说明各类看起来分得越开"""
    centers = np.array([Z[y == c].mean(axis=0) for c in np.unique(y)])
    within = np.mean([np.linalg.norm(Z[y == c] - centers[c], axis=1).mean() for c in np.unique(y)])
    between = np.mean([np.linalg.norm(a - b) for i, a in enumerate(centers) for b in centers[i + 1:]])
    return between / within

print(f"原始 64 维: kNN 准确率 {cross_val_score(KNeighborsClassifier(), Xd, yd, cv=5).mean():.2f}")
emb_pca = PCA(n_components=2).fit_transform(Xd)
emb_tsne = TSNE(n_components=2, random_state=0).fit_transform(Xd)
for name, Z in [("PCA 2 维", emb_pca), ("t-SNE 2 维", emb_tsne)]:
    acc = cross_val_score(KNeighborsClassifier(), Z, yd, cv=5).mean()
    print(f"{name}: kNN 准确率 {acc:.2f}，分离度 {separation(Z, yd):.1f}")

# ---- 代码 8 ----
for perp in [5, 100]:
    Z = TSNE(n_components=2, perplexity=perp, random_state=0).fit_transform(Xd)
    print(f"perplexity={perp:>3}: 坐标范围约 {np.ptp(Z[:, 0]):.0f}，分离度 {separation(Z, yd):.1f}")
print("TSNE 有 transform 方法吗？", hasattr(TSNE(), "transform"))

# ---- 代码 9 ----
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

raw = PCA().fit(X)                                   # 不缩放，直接对原始的葡萄酒数据做 PCA
print(f"不缩放: PC1 解释方差比 {raw.explained_variance_ratio_[0]:.3f}，"
      f"proline 在 PC1 上的载荷 {raw.components_[0][X.columns.get_loc('proline')]:.4f}")
print("各列标准差最大的两个:", X.std().round(1).nlargest(2).to_dict())
for name, model in [("不缩放", make_pipeline(PCA(n_components=2), LogisticRegression(max_iter=5000))),
                    ("先标准化", make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(max_iter=5000)))]:
    print(f"{name} + PCA 2 维 + 逻辑回归: 交叉验证准确率 {cross_val_score(model, X, y, cv=5).mean():.3f}")

# ---- 代码 10 ----
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA

Xr, yr = make_circles(n_samples=500, factor=0.3, noise=0.05, random_state=0)   # 两个同心圆
for name, reducer in [("PCA", PCA(n_components=1)),
                      ("KernelPCA rbf gamma=1", KernelPCA(n_components=1, kernel="rbf", gamma=1)),
                      ("KernelPCA rbf gamma=5", KernelPCA(n_components=1, kernel="rbf", gamma=5)),
                      ("KernelPCA rbf gamma=15", KernelPCA(n_components=1, kernel="rbf", gamma=15))]:
    acc = cross_val_score(make_pipeline(reducer, LogisticRegression()), Xr, yr, cv=5).mean()
    print(f"{name:<22} → 1 维 + 逻辑回归: {acc:.3f}")

# ---- 代码 11 ----
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

rng = np.random.default_rng(0)
n = 1000
z = rng.normal(size=n)
x1 = z + 0.1 * rng.normal(size=n)             # x1、x2 高度相关
x2 = z + 0.1 * rng.normal(size=n)
Xv = np.column_stack([x1, x2, rng.normal(size=(n, 3))])   # 再加 3 列噪声
yv = (x1 > x2).astype(int)                    # 标签只取决于 x1 - x2

p = make_pipeline(StandardScaler(), PCA()).fit(Xv)[-1]
print("各主成分的解释方差比:", np.round(p.explained_variance_ratio_, 3))
print("最后一个主成分:", np.round(p.components_[-1], 2))
for name, step in [("不降维", "passthrough"),
                   ("PCA 保留 95% 方差", PCA(n_components=0.95)),
                   ("LDA 1 维", LinearDiscriminantAnalysis(n_components=1))]:
    model = make_pipeline(StandardScaler(), step, LogisticRegression())
    print(f"{name}: 交叉验证准确率 {cross_val_score(model, Xv, yv, cv=5).mean():.3f}")

# ---- 代码 12 ----
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif

Xb, yb = load_breast_cancer(return_X_y=True)
Xb_noisy = np.hstack([Xb, np.random.default_rng(0).normal(size=(len(Xb), 100))])   # 30 列真特征 + 100 列噪声

# 像素同单位，只去掉恒为 0 的像素；乳腺癌各列量纲不同，先标准化
for data_name, prep, Xk, yk in [("digits (64 个相关的像素)", VarianceThreshold(), Xd, yd),
                                ("乳腺癌 + 100 列噪声", StandardScaler(), Xb_noisy, yb)]:
    base = cross_val_score(make_pipeline(prep, LogisticRegression(max_iter=5000)), Xk, yk, cv=5).mean()
    print(f"{data_name}，全部特征 {base:.3f}")
    for k in [5, 10, 20, 30]:
        pca_s, skb_s = [cross_val_score(make_pipeline(prep, step, LogisticRegression(max_iter=5000)),
                                        Xk, yk, cv=5).mean()
                        for step in [PCA(n_components=k), SelectKBest(f_classif, k=k)]]
        print(f"  k={k:>2}: PCA {pca_s:.3f}，SelectKBest {skb_s:.3f}")
