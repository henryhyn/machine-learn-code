"""C7 集成学习 (一)：Bagging 与随机森林：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/c7_random_forest/c7_random_forest.py
"""

# ---- 代码 1 ----
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n_models, n_trials = 25, 20000
for rho in [0.0, 0.3, 0.6, 0.9]:
    common = rng.normal(size=(n_trials, 1))           # 所有模型共有的误差
    own = rng.normal(size=(n_trials, n_models))       # 每个模型自己的误差
    errors = np.sqrt(rho) * common + np.sqrt(1 - rho) * own
    avg = errors.mean(axis=1)
    theory = rho + (1 - rho) / n_models
    print(f"相关系数 {rho:.1f}: 单个模型方差 {errors[:, 0].var():.3f}，"
          f"{n_models} 个平均后 {avg.var():.3f} (公式 {theory:.3f})")

# ---- 代码 2 ----
for n in [1, 5, 25, 101]:
    correct = rng.random((10000, n)) < 0.6           # 每个模型各自有 60% 的概率答对
    vote = (correct.sum(axis=1) > n / 2).mean()
    print(f"{n:>3} 个独立模型投票: 准确率 {vote:.3f}")

# ---- 代码 3 ----
for n in [10, 100, 426, 10000]:
    ratio = [len(np.unique(rng.choice(n, size=n, replace=True))) / n for _ in range(1000)]
    print(f"n = {n:>5}: 抽中的不同样本平均占 {np.mean(ratio):.3f}")
print(f"1 - 1/e = {1 - 1 / np.e:.3f}")

# ---- 代码 4 ----
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

models = {
    "单棵决策树": DecisionTreeClassifier(random_state=42),
    "Bagging (100 棵树)": BaggingClassifier(DecisionTreeClassifier(), n_estimators=100,
                                          random_state=42, n_jobs=-1),
    "随机森林": RandomForestClassifier(random_state=42, n_jobs=-1),
    "ExtraTrees": ExtraTreesClassifier(random_state=42, n_jobs=-1),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: 训练集 {model.score(X_train, y_train):.3f}，测试集 {model.score(X_test, y_test):.3f}")

# ---- 代码 5 ----
scores = {"单棵决策树": [], "随机森林": []}
for seed in range(20):
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    scores["单棵决策树"].append(DecisionTreeClassifier(random_state=0).fit(Xa, ya).score(Xb, yb))
    scores["随机森林"].append(RandomForestClassifier(random_state=0, n_jobs=-1).fit(Xa, ya).score(Xb, yb))
for name, s in scores.items():
    print(f"{name}: 平均 {np.mean(s):.3f}，标准差 {np.std(s):.3f}，最低 {min(s):.3f}，最高 {max(s):.3f}")

# ---- 代码 6 ----
rng = np.random.default_rng(0)
preds = {"单棵决策树": [], "随机森林": []}
for _ in range(20):
    idx = rng.choice(len(X_train), size=len(X_train), replace=True)
    Xb, yb = X_train.iloc[idx], y_train.iloc[idx]
    preds["单棵决策树"].append(DecisionTreeClassifier(random_state=0).fit(Xb, yb).predict(X_test))
    preds["随机森林"].append(RandomForestClassifier(random_state=0, n_jobs=-1).fit(Xb, yb).predict(X_test))
for name, p in preds.items():
    p = np.array(p)
    flipped = (p.min(axis=0) != p.max(axis=0)).mean()
    print(f"{name}: 20 次重训中预测结果变过的测试样本占 {flipped:.1%}")

# ---- 代码 7 ----
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(oob_score=True, random_state=42, n_jobs=-1).fit(X_train, y_train)
in_bag = np.mean([len(np.unique(s)) / len(X_train) for s in rf.estimators_samples_])
oob_trees = np.zeros(len(X_train))
for s in rf.estimators_samples_:
    oob_trees[np.setdiff1d(np.arange(len(X_train)), s)] += 1
print(f"每棵树用到的不同样本平均占 {in_bag:.3f}，每个样本平均是 {oob_trees.mean():.1f} 棵树的袋外样本")

cv = cross_val_score(RandomForestClassifier(random_state=42, n_jobs=-1), X_train, y_train, cv=5)
print(f"OOB 准确率 {rf.oob_score_:.3f}")
print(f"5 折交叉验证 {cv.mean():.3f} ± {cv.std():.3f}")
print(f"测试集准确率 {rf.score(X_test, y_test):.3f}")

# ---- 代码 8 ----
from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

Xr, yr = make_friedman1(n_samples=800, n_features=10, noise=1.0, random_state=0)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.25, random_state=0)

print(f"单棵回归树: 测试集 R² {DecisionTreeRegressor(random_state=0).fit(Xr_train, yr_train).score(Xr_test, yr_test):.3f}")
print(f"RandomForestRegressor 默认 max_features = {RandomForestRegressor().max_features}")
for mf in [1.0, 0.5, "sqrt"]:
    reg = RandomForestRegressor(max_features=mf, oob_score=True, random_state=0, n_jobs=-1)
    reg.fit(Xr_train, yr_train)
    print(f"随机森林 max_features={mf!s:>4}: OOB R² {reg.oob_score_:.3f}，测试集 R² {reg.score(Xr_test, yr_test):.3f}")

# ---- 代码 9 ----
from sklearn.inspection import permutation_importance

rng = np.random.default_rng(0)
n = 1000
data = pd.DataFrame({
    "x_cont": rng.normal(size=n),             # 有用的连续特征
    "x_bin": rng.integers(0, 2, size=n),      # 有用的二值特征
    "noise_cont": rng.normal(size=n),         # 纯噪声，连续
    "noise_bin": rng.integers(0, 2, size=n),  # 纯噪声，二值
})
logit = 1.0 * data["x_cont"] + 1.5 * data["x_bin"] - 0.75
target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
D_train, D_test, t_train, t_test = train_test_split(
    data, target, test_size=0.3, stratify=target, random_state=0)

forest = RandomForestClassifier(random_state=0, n_jobs=-1).fit(D_train, t_train)
print(f"训练集 {forest.score(D_train, t_train):.3f}，测试集 {forest.score(D_test, t_test):.3f}")
perm_test = permutation_importance(forest, D_test, t_test, n_repeats=10, random_state=0, n_jobs=-1)
perm_train = permutation_importance(forest, D_train, t_train, n_repeats=10, random_state=0, n_jobs=-1)
print(pd.DataFrame({
    "不纯度重要性": forest.feature_importances_,
    "置换(测试集)": perm_test.importances_mean,
    "置换(训练集)": perm_train.importances_mean,
}, index=data.columns).round(3))

# ---- 代码 10 ----
from sklearn.datasets import make_classification

Xc, yc = make_classification(n_samples=1000, n_features=40, n_informative=8, n_redundant=10,
                             flip_y=0.05, random_state=0)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.3, random_state=0)

for mf in [None, 0.5, "sqrt", 1]:
    m = RandomForestClassifier(max_features=mf, random_state=0, n_jobs=-1).fit(Xc_train, yc_train)
    right = np.array([t.predict(Xc_test) == yc_test for t in m.estimators_])   # 每棵树对不对
    corr = np.corrcoef(right)[np.triu_indices(len(right), k=1)].mean()
    print(f"max_features={mf!s:>4}: 单棵树平均准确率 {right.mean():.3f}，"
          f"树与树对错的相关系数 {corr:.3f}，森林准确率 {m.score(Xc_test, yc_test):.3f}")

# ---- 代码 11 ----
from sklearn.metrics import log_loss

for n_trees in [1, 5, 10, 25, 50, 100, 200, 500]:
    m = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1).fit(X_train, y_train)
    print(f"{n_trees:>3} 棵树: 测试集准确率 {m.score(X_test, y_test):.3f}，"
          f"对数损失 {log_loss(y_test, m.predict_proba(X_test)):.3f}")

# ---- 代码 12 ----
from sklearn.linear_model import LogisticRegression

for leaf in [1, 5, 20, 50]:
    m = RandomForestClassifier(min_samples_leaf=leaf, random_state=0, n_jobs=-1).fit(D_train, t_train)
    print(f"min_samples_leaf={leaf:>2}: 训练集 {m.score(D_train, t_train):.3f}，测试集 {m.score(D_test, t_test):.3f}，"
          f"noise_cont 的不纯度重要性 {m.feature_importances_[2]:.3f}")
print(f"逻辑回归: 测试集 {LogisticRegression().fit(D_train, t_train).score(D_test, t_test):.3f}")

# ---- 代码 13 ----
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(2)
x = rng.uniform(0, 5, size=80).reshape(-1, 1)
y_line = 2 * x.ravel() + rng.normal(0, 0.5, size=80)    # 真实规律 y = 2x

forest_reg = RandomForestRegressor(random_state=0, n_jobs=-1).fit(x, y_line)
linear = LinearRegression().fit(x, y_line)
for v in [2.5, 4.9, 8, 100]:
    print(f"x = {v:>5}: 随机森林 {forest_reg.predict([[v]])[0]:.2f}，"
          f"线性回归 {linear.predict([[v]])[0]:.2f}，真实规律 {2 * v:.1f}")

# ---- 代码 14 ----
single = models["单棵决策树"]
print(f"单棵树 {single.tree_.node_count} 个节点")
print(f"随机森林 {len(rf.estimators_)} 棵树，共 {sum(t.tree_.node_count for t in rf.estimators_)} 个节点")
