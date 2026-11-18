"""m1 基线：先知道“什么都不学”能得多少分，再看一个最朴素的模型能比它好多少。

- 多数类基线 (DummyClassifier)：永远预测“不流失”；
- 逻辑回归：原始特征直接喂进去，不做任何预处理。
另外统计不去重时有多少测试样本在训练集里有完全相同的副本，说明为什么 load_data 默认去重。

运行：uv run python projects/p1_churn/m1_baseline.py
"""
import warnings

from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from common import evaluate, load_data, report, split


def main():
    raw = load_data(drop_duplicates=False)
    df = load_data()
    print(f"原始 {len(raw)} 行，完全重复 {raw.duplicated().sum()} 行，去重后 {len(df)} 行，流失率 {df['churn'].mean():.3f}\n")

    # 不去重时，同一条记录可能同时落进训练集和测试集：测试集等于部分“背过答案”
    Xr_train, Xr_test, _, _ = split(raw)
    seen = Xr_test.merge(Xr_train.drop_duplicates(), how="inner").shape[0]
    print(f"不去重时，测试集 {len(Xr_test)} 行中有 {seen} 行在训练集里有一模一样的副本\n")

    X_train, X_test, y_train, y_test = split(df)
    rows = [evaluate("多数类基线", DummyClassifier(strategy="prior"), X_train, X_test, y_train, y_test)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        rows.append(evaluate("逻辑回归 (原始特征)", LogisticRegression(max_iter=5000),
                             X_train, X_test, y_train, y_test))
    n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    print(f"逻辑回归训练了 6 次 (5 折 + 最终一次)，其中 {n_warn} 次迭代 5000 步仍未收敛：原始特征量纲差几个数量级\n")
    report("m1", rows)


if __name__ == "__main__":
    main()
