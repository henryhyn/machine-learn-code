"""P1 客户流失预测：各里程碑共用的数据加载、切分和评估。

数据来自 UCI Iranian Churn 数据集 (CC BY 4.0)，见 data/README.md。
"""
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data" / "customer_churn.csv"
RESULTS = HERE / "results"
TARGET = "Churn"
SEED = 42

# 交叉验证和最终测试共用的切分，所有里程碑都一样，分数才可比
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = {"ap": "average_precision", "roc_auc": "roc_auc"}


def load_data(drop_duplicates=True):
    """读入原始 CSV，规范列名 (原文件里有双空格)，默认去掉完全重复的行。"""
    df = pd.read_csv(DATA)
    df.columns = ["_".join(c.lower().split()) for c in df.columns]
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def split(df):
    """固定的训练 / 测试切分：测试集 25%，按标签分层。测试集只在每个里程碑最后用一次。"""
    X = df.drop(columns=TARGET.lower())
    y = df[TARGET.lower()]
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)


def evaluate(name, model, X_train, X_test, y_train, y_test):
    """训练集上 5 折交叉验证 + 测试集评估，返回一行结果。"""
    cv = cross_validate(model, X_train, y_train, cv=CV, scoring=SCORING)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "cv_ap": round(float(np.mean(cv["test_ap"])), 3),
        "cv_ap_std": round(float(np.std(cv["test_ap"])), 3),
        "test_ap": round(float(average_precision_score(y_test, proba)), 3),
        "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
        "test_precision": round(float(precision_score(y_test, pred, zero_division=0)), 3),
        "test_recall": round(float(recall_score(y_test, pred)), 3),
        "test_f1": round(float(f1_score(y_test, pred)), 3),
    }


def report(milestone, rows):
    """打印结果表并写入 results/<里程碑>.json。"""
    table = pd.DataFrame(rows).set_index("model")
    print(table.to_string())
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{milestone}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return table
