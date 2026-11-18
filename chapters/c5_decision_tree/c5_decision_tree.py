"""C5 决策树：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/c5_decision_tree/c5_decision_tree.py
"""

# ---- 代码 1 ----
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "用户":   [f"u{i:02d}" for i in range(1, 15)],
    "活跃度": ["高"] * 5 + ["中"] * 4 + ["低"] * 5,
    "学生":   ["是", "否", "否", "是", "否", "是", "是", "否", "否", "是", "否", "否", "是", "否"],
    "优惠券": ["有", "有", "无", "无", "无", "有", "无", "无", "有", "无", "有", "无", "有", "有"],
    "续费":   ["是"] * 4 + ["否"] + ["是"] * 3 + ["否"] * 6,
})
print(pd.crosstab(df["活跃度"], df["续费"]))

# ---- 代码 2 ----
def entropy(labels):
    p = labels.value_counts(normalize=True).to_numpy()
    return float(-(p * np.log2(p)).sum())

def info_gain(df, feature, target="续费"):
    weights = df[feature].value_counts(normalize=True)
    cond = sum(w * entropy(df.loc[df[feature] == v, target]) for v, w in weights.items())
    return entropy(df[target]) - cond

print(f"H(D) = {entropy(df['续费']):.3f}")
for f in ["活跃度", "学生", "优惠券"]:
    print(f"{f}: 信息增益 = {info_gain(df, f):.3f}")

# ---- 代码 3 ----
def gain_ratio(df, feature, target="续费"):
    return info_gain(df, feature, target) / entropy(df[feature])

for f in ["活跃度", "学生", "优惠券", "用户"]:
    print(f"{f}: 信息增益 = {info_gain(df, f):.3f}, 固有值 = {entropy(df[f]):.3f}, 增益率 = {gain_ratio(df, f):.3f}")

# ---- 代码 4 ----
def gini(labels):
    p = labels.value_counts(normalize=True).to_numpy()
    return float(1 - (p ** 2).sum())

for level in ["高", "中", "低"]:
    left = df[df["活跃度"] == level]["续费"]
    right = df[df["活跃度"] != level]["续费"]
    g = (len(left) * gini(left) + len(right) * gini(right)) / len(df)
    print(f"活跃度 == {level} / != {level}: 加权基尼 = {g:.3f}")

# ---- 代码 5 ----
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
print(f"深度 {tree.get_depth()}，叶子 {tree.get_n_leaves()}")
print(f"训练集准确率 {tree.score(X_train, y_train):.3f}，测试集准确率 {tree.score(X_test, y_test):.3f}")

# ---- 代码 6 ----
from sklearn.tree import export_text

small = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
print(f"测试集准确率 {small.score(X_test, y_test):.3f}")
print(export_text(small, feature_names=list(X.columns), decimals=2))

# ---- 代码 7 ----
from sklearn.model_selection import cross_validate

for depth in [1, 2, 3, 4, 5, 6, 8, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    r = cross_validate(model, X_train, y_train, cv=5, return_train_score=True)
    print(f"max_depth={str(depth):>4}  训练 {r['train_score'].mean():.3f}  验证 {r['test_score'].mean():.3f}")

# ---- 代码 8 ----
from sklearn.model_selection import GridSearchCV

path = tree.cost_complexity_pruning_path(X_train, y_train)
search = GridSearchCV(DecisionTreeClassifier(random_state=42),
                      {"ccp_alpha": path.ccp_alphas[:-1]}, cv=5)
search.fit(X_train, y_train)
best = search.best_estimator_
print(f"候选 alpha {len(path.ccp_alphas) - 1} 个，选中 {search.best_params_['ccp_alpha']:.4f}")
print(f"叶子 {best.get_n_leaves()}，测试集准确率 {best.score(X_test, y_test):.3f}")

# ---- 代码 9 ----
rng = np.random.default_rng(0)
roots = []
for _ in range(20):
    idx = rng.choice(len(X_train), size=len(X_train), replace=True)
    m = DecisionTreeClassifier(random_state=0).fit(X_train.iloc[idx], y_train.iloc[idx])
    roots.append(X.columns[m.tree_.feature[0]])
print(pd.Series(roots).value_counts())

# ---- 代码 10 ----
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(1)
P = rng.uniform(0, 1, size=(1000, 2))
label = (P[:, 0] > P[:, 1]).astype(int)    # 边界是对角线 x1 = x2

for name, model in [("决策树", DecisionTreeClassifier(random_state=0)),
                    ("逻辑回归", LogisticRegression())]:
    acc = cross_val_score(model, P, label, cv=5).mean()
    print(f"{name}: 交叉验证准确率 {acc:.3f}")
print(f"决策树叶子数 {DecisionTreeClassifier(random_state=0).fit(P, label).get_n_leaves()}")

# ---- 代码 11 ----
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(2)
x = rng.uniform(0, 5, size=80)
y_reg = 2 * x + rng.normal(0, 0.5, size=80)    # 真实规律 y = 2x

reg = DecisionTreeRegressor(max_depth=4, random_state=0).fit(x.reshape(-1, 1), y_reg)
for v in [2.5, 4.9, 8, 100]:
    print(f"x = {v:>5}: 预测 {reg.predict([[v]])[0]:.2f}，真实规律 {2 * v:.1f}")

# ---- 代码 12 ----
X_missing = X_train.copy()
X_missing.iloc[::7, 0] = np.nan        # 每 7 行抹掉一个 mean radius
print(f"缺失 {X_missing.iloc[:, 0].isna().sum()} 个值")
m = DecisionTreeClassifier(random_state=42).fit(X_missing, y_train)
print(f"测试集准确率 {m.score(X_test, y_test):.3f}")
