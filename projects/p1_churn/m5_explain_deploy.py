"""m5 模型解释与上线：哪些特征在起作用，以及怎么把模型交出去。

1. 置换重要性 (permutation_importance)：在测试集上打乱一列，看 AP 掉多少；
   和基于不纯度的重要性不同，它衡量的是“模型在新数据上依赖这一列多少” (见书中 C7)。
2. 把 m4 选出的整个模型 (预处理 + HistGradientBoosting + 阈值) 用 joblib 存成一个文件，
   predict.py 只需加载它就能对新的 CSV 打分。

运行：uv run python projects/p1_churn/m5_explain_deploy.py
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV

from common import CV, HERE, RESULTS, SEED, load_data, split
from m4_threshold import m3_model, profit

MODEL_PATH = HERE / "models" / "churn_model.joblib"


def main():
    X_train, X_test, y_train, y_test = split(load_data())
    model = TunedThresholdClassifierCV(m3_model(), scoring=make_scorer(profit), cv=CV,
                                       thresholds=np.linspace(0.01, 0.99, 99))
    model.fit(X_train, y_train)

    result = permutation_importance(model.estimator_, X_test, y_test, scoring="average_precision",
                                    n_repeats=20, random_state=SEED)
    imp = (pd.DataFrame({"feature": X_test.columns,
                         "ap_drop_mean": result.importances_mean.round(3) + 0.0,   # + 0.0 去掉 -0.0
                         "ap_drop_std": result.importances_std.round(3) + 0.0})
           .sort_values("ap_drop_mean", ascending=False).reset_index(drop=True))
    print("置换重要性 (测试集上打乱该列后 AP 的下降，重复 20 次)：")
    print(imp.to_string(index=False))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "m5.json").write_text(
        json.dumps(imp.to_dict(orient="records"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    reloaded = joblib.load(MODEL_PATH)
    same = bool((reloaded.predict(X_test) == model.predict(X_test)).all())
    print(f"\n模型已保存到 {MODEL_PATH.relative_to(HERE)}，重新加载后预测一致：{same}，"
          f"阈值 {reloaded.best_threshold_:.2f}")


if __name__ == "__main__":
    main()
