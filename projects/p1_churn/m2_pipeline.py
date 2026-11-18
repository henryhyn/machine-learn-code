"""m2 预处理 Pipeline：按列的含义分别处理，整个关进 Pipeline 交给交叉验证。

列的处理 (依据数据说明和 m1 的探索)：
- 计数 / 金额类 (通话次数、秒数、短信数……)：右偏严重，先 log1p 再标准化；
- 有序类别 (charge_amount、age_group)：本身就是 0–10、1–5 的等级，直接标准化；
- 二值类别 (complains、tariff_plan、status)：独热编码 (drop="if_binary" 保留一列)；
- age：和 age_group 一一对应 (15/25/30/45/55)，是冗余列，丢掉。

运行：uv run python projects/p1_churn/m2_pipeline.py
"""
import warnings

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from common import evaluate, load_data, report, split

COUNT_COLS = ["call_failure", "subscription_length", "seconds_of_use", "frequency_of_use",
              "frequency_of_sms", "distinct_called_numbers", "customer_value"]
ORDINAL_COLS = ["charge_amount", "age_group"]
BINARY_COLS = ["complains", "tariff_plan", "status"]
DROP_COLS = ["age"]


def make_preprocessor():
    return ColumnTransformer([
        ("count", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                StandardScaler()), COUNT_COLS),
        ("ordinal", StandardScaler(), ORDINAL_COLS),
        ("binary", OneHotEncoder(drop="if_binary"), BINARY_COLS),
        ("drop", "drop", DROP_COLS),
    ])


def make_model():
    return Pipeline([("prep", make_preprocessor()), ("clf", LogisticRegression(max_iter=1000))])


def main():
    X_train, X_test, y_train, y_test = split(load_data())
    scaled_only = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        rows = [
            evaluate("逻辑回归 + 只做标准化", scaled_only, X_train, X_test, y_train, y_test),
            evaluate("逻辑回归 + 按列预处理", make_model(), X_train, X_test, y_train, y_test),
        ]
    n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    print(f"收敛警告 {n_warn} 次\n")
    report("m2", rows)


if __name__ == "__main__":
    main()
