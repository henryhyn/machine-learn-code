"""A2 机器学习在“学”什么：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/a2_what_ml_learns/a2_what_ml_learns.py
"""

# ---- 代码 1 ----
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
print(X.shape)
print(X.head(3))
print(iris.target_names.tolist(), y.value_counts().sort_index().tolist())

# ---- 代码 2 ----
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

km = KMeans(n_clusters=3, random_state=0).fit(X)            # 只给 X，不给 y
species = y.map(dict(enumerate(iris.target_names)))
print(pd.crosstab(species, km.labels_, rownames=["真实品种"], colnames=["簇"]))

pca = PCA(n_components=2).fit(X)                             # 4 维压到 2 维
print(f"2 个主成分保留的方差比例: {pca.explained_variance_ratio_.sum():.3f}")

# ---- 代码 3 ----
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
x = rng.uniform(0, 1, 100)
t = 2 * x + 1 + rng.normal(0, 0.1, 100)       # 真实规律 t = 2x + 1，加一点噪声

# 算法一：梯度下降，沿着损失下降最快的方向一小步一小步地走
w, b = 0.0, 0.0
for _ in range(2000):
    err = w * x + b - t
    w -= 0.1 * 2 * (err * x).mean()
    b -= 0.1 * 2 * err.mean()
print(f"梯度下降:   w = {w:.3f}, b = {b:.3f}, 经验风险 = {((w * x + b - t) ** 2).mean():.4f}")

# 算法二：最小二乘的解析解 (sklearn 的 LinearRegression)
lr = LinearRegression().fit(x.reshape(-1, 1), t)
print(f"最小二乘:   w = {lr.coef_[0]:.3f}, b = {lr.intercept_:.3f}")

# ---- 代码 4 ----
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(1)

def sample(n):
    x = rng.uniform(0, 1, n)
    return x.reshape(-1, 1), np.sin(2 * np.pi * x) + rng.normal(0, 0.2, n)

X_tr, y_tr = sample(12)       # 训练集：只有 12 个点
X_te, y_te = sample(1000)     # 测试集：同一个规律下的 1000 个新点

for degree in [1, 3, 6, 9, 11]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression()).fit(X_tr, y_tr)
    train_mse = mean_squared_error(y_tr, model.predict(X_tr))
    test_mse = mean_squared_error(y_te, model.predict(X_te))
    print(f"{degree:>2} 阶  训练误差 {train_mse:.3f}  测试误差 {test_mse:.3f}")

# ---- 代码 5 ----
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
print(f"训练集 {X_train.shape}，测试集 {X_test.shape}")
print(f"测试集各类数量 {y_test.value_counts().sort_index().tolist()}")

# ---- 代码 6 ----
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)            # 学习：只看训练集
y_pred = knn.predict(X_test)         # 预测：测试集的花它从没见过
print("预测:", y_pred[:10])
print("真实:", y_test.to_numpy()[:10])

# ---- 代码 7 ----
from sklearn.metrics import accuracy_score, confusion_matrix

print(f"测试集准确率 {accuracy_score(y_test, y_pred):.3f}")
print(f"训练集准确率 {knn.score(X_train, y_train):.3f}")
print(confusion_matrix(y_test, y_pred))

# ---- 代码 8 ----
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier().fit(X_train, y_train)    # 不看特征，永远猜训练集里最多的类
print(f"基线准确率 {dummy.score(X_test, y_test):.3f}")

# ---- 代码 9 ----
new_flowers = pd.DataFrame([[5.0, 3.4, 1.5, 0.2],
                            [6.0, 2.8, 4.9, 1.6]], columns=X.columns)
print(iris.target_names[knn.predict(new_flowers)])
print(knn.predict_proba(new_flowers))
