"""m3 多模型对比 + 交叉验证调参。

1. 同一套交叉验证，对比逻辑回归 (m2 的 Pipeline)、决策树、随机森林、HistGradientBoosting；
2. 只在训练集上用 RandomizedSearchCV 给 HistGradientBoosting 调参 (按 AP 选)；
3. 选出的模型在测试集上评估一次。

树模型不需要缩放，只丢掉冗余的 age 列。

运行：uv run python projects/p1_churn/m3_compare.py
"""
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common import CV, SEED, evaluate, load_data, report, split
from m2_pipeline import DROP_COLS, make_model


def tree_pipeline(clf):
    drop_age = ColumnTransformer([("drop", "drop", DROP_COLS)], remainder="passthrough")
    return Pipeline([("prep", drop_age), ("clf", clf)])


def main():
    X_train, X_test, y_train, y_test = split(load_data())
    candidates = {
        "逻辑回归 (m2)": make_model(),
        "决策树 (max_depth=5)": tree_pipeline(DecisionTreeClassifier(max_depth=5, random_state=SEED)),
        "随机森林": tree_pipeline(RandomForestClassifier(random_state=SEED, n_jobs=-1)),
        "HistGradientBoosting": tree_pipeline(HistGradientBoostingClassifier(random_state=SEED)),
    }
    rows = [evaluate(name, model, X_train, X_test, y_train, y_test) for name, model in candidates.items()]

    search = RandomizedSearchCV(
        tree_pipeline(HistGradientBoostingClassifier(random_state=SEED)),
        {
            "clf__learning_rate": loguniform(0.02, 0.3),
            "clf__max_leaf_nodes": randint(8, 64),
            "clf__min_samples_leaf": randint(5, 50),
            "clf__l2_regularization": loguniform(1e-3, 10),
            "clf__max_iter": randint(100, 600),
        },
        n_iter=40, scoring="average_precision", cv=CV, random_state=SEED, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best = {k.removeprefix("clf__"): (round(float(v), 4) if isinstance(v, float) else int(v))
            for k, v in sorted(search.best_params_.items())}
    print(f"调参：40 组候选，交叉验证 AP 最好 {search.best_score_:.3f}，参数 {best}\n")
    rows.append(evaluate("HistGradientBoosting (调参后)", search.best_estimator_,
                         X_train, X_test, y_train, y_test))
    report("m3", rows)


if __name__ == "__main__":
    main()
