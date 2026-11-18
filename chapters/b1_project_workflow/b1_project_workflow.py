"""B1 一个项目的完整流程：正文里的全部代码，按出现顺序排列。

用 P1 客户流失预测的数据演示问题定义、多数类基线、人工规则基线和模型的对比。

运行：uv run python chapters/b1_project_workflow/b1_project_workflow.py
"""

# ---- 代码 1 ----
import pathlib

import pandas as pd

def find_churn_csv(rel="projects/p1_churn/data/customer_churn.csv"):
    """从当前目录往上找 P1 的数据：配套仓库根目录、或与它并列的目录下运行都能找到。"""
    here = pathlib.Path.cwd().resolve()
    for base in [here, *here.parents]:
        for cand in [base / rel, base / "machine-learn-code" / rel]:
            if cand.exists():
                return cand
    raise FileNotFoundError(f"找不到 {rel}，请在配套仓库根目录运行")

df = pd.read_csv(find_churn_csv())
df.columns = ["_".join(c.lower().split()) for c in df.columns]
print(f"原始 {len(df)} 行，完全重复 {df.duplicated().sum()} 行")
df = df.drop_duplicates().reset_index(drop=True)
print(f"去重后 {len(df)} 行 × {df.shape[1]} 列，流失率 {df['churn'].mean():.3f}")
print(df[["complains", "status", "subscription_length", "seconds_of_use", "churn"]].head(3).to_string())

# ---- 代码 2 ----
from sklearn.model_selection import train_test_split

X, y = df.drop(columns="churn"), df["churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
train = X_train.assign(churn=y_train)
for col, value, label in [("complains", 0, "没投诉"), ("complains", 1, "投诉过"),
                          ("status", 1, "活跃"), ("status", 2, "不活跃")]:
    group = train[train[col] == value]
    print(f"{label}：{len(group)} 人，流失率 {group['churn'].mean():.3f}")

# ---- 代码 3 ----
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

def report(name, pred):
    print(f"{name}：判为流失 {pred.sum()} 人，精确率 {precision_score(y_test, pred, zero_division=0):.3f}，"
          f"召回率 {recall_score(y_test, pred):.3f}，F1 {f1_score(y_test, pred):.3f}")

dummy = DummyClassifier().fit(X_train, y_train)
report("多数类基线", dummy.predict(X_test))
report("规则 1 (投诉过)", (X_test["complains"] == 1).astype(int))
rule2 = ((X_test["complains"] == 1) | (X_test["status"] == 2)).astype(int)
report("规则 2 (投诉或不活跃)", rule2)

# ---- 代码 4 ----
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(random_state=42).fit(X_train.drop(columns="age"), y_train)
proba = hgb.predict_proba(X_test.drop(columns="age"))[:, 1]
report("HistGradientBoosting (阈值 0.5)", (proba >= 0.5).astype(int))

k = int(rule2.sum())
top_k = pd.Series(proba, index=X_test.index).nlargest(k).index
print(f"同样联系 {k} 人：规则 2 抓到 {y_test[rule2 == 1].sum()} 个流失客户，"
      f"模型取概率最高的 {k} 人抓到 {y_test[top_k].sum()} 个 (测试集共 {y_test.sum()} 个)")
