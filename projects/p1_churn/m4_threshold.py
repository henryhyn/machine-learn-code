"""m4 类别不平衡与阈值：按业务代价决定“联系谁”。

m3 的模型排序能力已经很好 (AP 0.95)，但默认阈值 0.5 是随手定的。这里假设一个简单的业务账：
- 给一个客户打挽留电话，成本 1；
- 打给一个真会流失的客户并挽留成功，收益 10 (成本已另算)。
于是利润 = 10 × 联系到的流失客户数 − 1 × 联系的总人数。

对比下面几种决策方式 (都在同一个测试集上评估)：
1. m3 模型 + 默认阈值 0.5；
2. 训练时加 class_weight="balanced"，阈值仍是 0.5；
3. 理论阈值：概率 p 的客户，联系的期望利润是 GAIN × p − COST，p > COST / GAIN = 0.1 就该联系
   (前提是概率校准得好，见书中 C9)；
4. TunedThresholdClassifierCV 在训练集上用交叉验证按利润选阈值；
5. 两个参照：谁都不联系、所有人都联系。

运行：uv run python projects/p1_churn/m4_threshold.py
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV

from common import CV, SEED, load_data, report, split
from m3_compare import tree_pipeline

GAIN, COST = 10, 1
# m3 随机搜索选出的参数 (见 results/m3.json 对应的打印)，固定下来免得每次重新搜索
M3_PARAMS = dict(learning_rate=0.034, max_iter=289, max_leaf_nodes=47,
                 min_samples_leaf=25, l2_regularization=0.0023)


def profit(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return GAIN * tp - COST * (tp + fp)


def m3_model(**extra):
    return tree_pipeline(HistGradientBoostingClassifier(random_state=SEED, **M3_PARAMS, **extra))


def row(name, y_test, pred, proba=None, threshold=None):
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "threshold": None if threshold is None else round(float(threshold), 3),
        "contacted": int(tp + fp),
        "caught": int(tp),
        "recall": round(tp / (tp + fn), 3),
        "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
        "profit": int(profit(y_test, pred)),
        "test_ap": None if proba is None else round(float(average_precision_score(y_test, proba)), 3),
    }


def main():
    X_train, X_test, y_train, y_test = split(load_data())
    n_churn = int(y_test.sum())
    print(f"测试集 {len(y_test)} 人，其中流失 {n_churn} 人；联系 1 人成本 {COST}，挽留 1 个流失客户收益 {GAIN}\n")

    rows = [
        row("谁都不联系", y_test, np.zeros_like(y_test)),
        row("所有人都联系", y_test, np.ones_like(y_test)),
    ]

    base = m3_model().fit(X_train, y_train)
    proba = base.predict_proba(X_test)[:, 1]
    rows.append(row("m3 模型 + 阈值 0.5", y_test, (proba >= 0.5).astype(int), proba, 0.5))

    t = COST / GAIN
    rows.append(row(f"m3 模型 + 理论阈值 {t}", y_test, (proba >= t).astype(int), proba, t))

    balanced = m3_model(class_weight="balanced").fit(X_train, y_train)
    proba_b = balanced.predict_proba(X_test)[:, 1]
    rows.append(row("class_weight=balanced + 阈值 0.5", y_test, (proba_b >= 0.5).astype(int), proba_b, 0.5))

    tuned = TunedThresholdClassifierCV(m3_model(), scoring=make_scorer(profit), cv=CV,
                                       thresholds=np.linspace(0.01, 0.99, 99))
    tuned.fit(X_train, y_train)
    proba_t = tuned.estimator_.predict_proba(X_test)[:, 1]
    rows.append(row("按利润调阈值 (交叉验证)", y_test, tuned.predict(X_test), proba_t, tuned.best_threshold_))

    report("m4", rows)


if __name__ == "__main__":
    main()
