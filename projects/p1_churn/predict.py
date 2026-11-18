"""用 m5 保存的模型给新客户打分。

用法 (先运行一次 m5_explain_deploy.py 生成模型文件)：
    uv run python projects/p1_churn/predict.py 输入.csv [输出.csv]

输入 CSV 的列和原始数据一致 (列名里的多余空格会被规范化，可以不含 Churn 列)。
输出在输入的基础上加两列：churn_proba (流失概率) 和 contact (1 表示建议联系)。

注意：joblib / pickle 文件加载时会执行其中的代码，只加载自己生成或可信来源的模型文件。
"""
import sys

import joblib
import pandas as pd

from common import HERE

MODEL_PATH = HERE / "models" / "churn_model.joblib"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    df = pd.read_csv(sys.argv[1])
    df.columns = ["_".join(c.lower().split()) for c in df.columns]
    X = df.drop(columns=["churn"], errors="ignore")

    model = joblib.load(MODEL_PATH)
    df["churn_proba"] = model.estimator_.predict_proba(X)[:, 1].round(4)
    df["contact"] = model.predict(X)

    if len(sys.argv) > 2:
        df.to_csv(sys.argv[2], index=False)
        print(f"已写入 {sys.argv[2]}：{len(df)} 人，建议联系 {int(df['contact'].sum())} 人")
    else:
        print(df[["churn_proba", "contact"]].to_string())


if __name__ == "__main__":
    main()
