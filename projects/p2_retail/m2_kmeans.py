"""m2 K-Means 分群与画像：在 RFM 上做 K-Means，选 k、画像、起名字，再和 m1 的 RFM 打分分群对比。

- Pipeline(log1p + StandardScaler + KMeans)：RFM 三列强右偏，取对数压住长尾，再标准化让三列一样重 (书中 B2、D1)；
- k 用肘部 + 轮廓系数缩小范围，再看画像选业务上讲得通的那个；
- 稳定性：不同 random_state 的结果和参考结果的 ARI；
- 对比：m1 的 RFM 打分分群 (交叉表 + ARI)、高斯混合；
- 离群大客户：不取对数时，少数大客户自己占一个簇。

运行：uv run python projects/p2_retail/m2_kmeans.py
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from common import SEED, compute_rfm, load_clean, report
from m1_rfm import SEGMENT_ORDER, profile, rfm_scores, rfm_segments

K = 4
# 按各簇净消费额中位数从低到高排好之后的名字 (名字来自下面打印的画像)
CLUSTER_NAMES = ["沉睡低价值", "新近低频", "中坚老客", "核心大客户"]


def make_pipeline_k(k, random_state=SEED, n_init=10):
    return Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=n_init, random_state=random_state)),
    ])


def fit_segments(rfm):
    """拟合 k=4 的 Pipeline，返回 (pipeline, 每个客户的群名)。m3、m4 也用它。"""
    pipe = make_pipeline_k(K).fit(rfm)
    labels = pipe["kmeans"].labels_
    order = rfm.groupby(labels)["Monetary"].median().sort_values().index   # 簇编号按消费额中位数排序
    names = dict(zip(order, CLUSTER_NAMES))
    return pipe, pd.Series([names[c] for c in labels], index=rfm.index, name="cluster")


def main():
    rfm = compute_rfm(load_clean())
    print(f"{len(rfm)} 个客户\n")

    # 1. 选 k：肘部 + 轮廓系数 (在 log1p + 标准化后的空间里算)
    Z = make_pipeline_k(2)[:-1].fit_transform(rfm)
    scan = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        scan.append({"k": k, "inertia": round(float(km.inertia_), 1),
                     "silhouette": round(float(silhouette_score(Z, km.labels_)), 3)})
    print(pd.DataFrame(scan).to_string(index=False), "\n")

    # 2. k=4 的画像
    pipe, cluster = fit_segments(rfm)
    prof = profile(rfm, cluster, CLUSTER_NAMES)
    print(f"k={K} 的画像：")
    print(prof.to_string(), "\n")

    # 3. 和 m1 的 RFM 打分分群对比
    seg = rfm_segments(rfm_scores(rfm))
    cross = pd.crosstab(cluster, seg).loc[CLUSTER_NAMES, SEGMENT_ORDER]
    ari_rfm = adjusted_rand_score(seg, cluster)
    print(cross.to_string())
    print(f"K-Means 与 RFM 打分分群的 ARI {ari_rfm:.3f}\n")

    # 4. 初始化稳定性：n_init=1 和 n_init=10 各换 10 个 random_state，和参考结果比 ARI
    labels = pipe["kmeans"].labels_
    stability = {}
    for n_init in [1, 10]:
        aris = [adjusted_rand_score(labels, KMeans(n_clusters=K, n_init=n_init, random_state=s).fit_predict(Z))
                for s in range(10)]
        stability[f"n_init_{n_init}"] = {"min": round(min(aris), 3), "mean": round(float(np.mean(aris)), 3)}
        print(f"n_init={n_init:<2} 换 10 个 random_state：与参考结果的 ARI 最小 {min(aris):.3f}，平均 {np.mean(aris):.3f}")

    # 5. 高斯混合 (同样的特征空间、同样 4 个分量)
    gmm = GaussianMixture(n_components=K, random_state=SEED).fit(Z)
    ari_gmm = adjusted_rand_score(labels, gmm.predict(Z))
    print(f"高斯混合与 K-Means 的 ARI {ari_gmm:.3f}\n")

    # 6. 不取对数：只标准化，离群的大客户会自己占一个簇
    raw_km = make_pipeline(StandardScaler(), KMeans(n_clusters=K, n_init=10, random_state=SEED)).fit(rfm)
    raw_labels = raw_km[-1].labels_
    sizes = np.bincount(raw_labels)
    small = int(np.argmin(sizes))
    small_share = rfm["Monetary"][raw_labels == small].sum() / rfm["Monetary"].sum()
    raw_sil = silhouette_score(raw_km[0].transform(rfm), raw_labels)
    print(f"不取对数 (只标准化)：簇大小 {sorted(sizes.tolist())}，最小的簇 {sizes[small]} 个客户、"
          f"占净消费额 {small_share:.3f}；轮廓系数 {raw_sil:.3f} (在只标准化的空间里算)")

    report("m2", {
        "customers": len(rfm),
        "k_scan": scan,
        "k": K,
        "profile": [{"cluster": c, **r} for c, r in prof.to_dict("index").items()],
        "crosstab_vs_rfm_segments": {c: r for c, r in cross.to_dict("index").items()},
        "ari_vs_rfm_segments": round(float(ari_rfm), 3),
        "stability": stability,
        "ari_gmm_vs_kmeans": round(float(ari_gmm), 3),
        "no_log": {"cluster_sizes": sorted(sizes.tolist()), "smallest_cluster": int(sizes[small]),
                   "smallest_cluster_monetary_share": round(float(small_share), 3),
                   "silhouette": round(float(raw_sil), 3)},
    })


if __name__ == "__main__":
    main()
