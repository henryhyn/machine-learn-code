"""A1 从人工智能到机器学习：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/a1_ai_to_ml/a1_ai_to_ml.py
"""

# ---- 代码 1 ----
from sklearn.datasets import load_digits

digits = load_digits()
print(digits.images.shape)          # 1797 张图，每张 8×8 像素
print(digits.images[0].astype(int)) # 每个像素是 0–16 的墨迹深浅
print("这张图的标签:", digits.target[0])

# ---- 代码 2 ----
import numpy as np
from sklearn.model_selection import train_test_split

images_train, images_test, y_train, y_test = train_test_split(
    digits.images, digits.target, test_size=0.25, stratify=digits.target, random_state=42)
is_zero = y_test == 0

def looks_like_zero(img, hole_max=4, side_min=8):
    """人写的规则：中间是空洞，左右两侧都有笔画。"""
    hole = img[3:5, 3:5].mean()     # 中心 2×2 区域的平均墨迹
    left = img[2:6, 1:3].max()      # 左侧有没有笔画
    right = img[2:6, 5:7].max()     # 右侧有没有笔画
    return hole < hole_max and left > side_min and right > side_min

print(f"测试集共 {len(y_test)} 张，其中 0 有 {is_zero.sum()} 张")
for hole_max in [1, 2, 4, 6]:
    pred = np.array([looks_like_zero(img, hole_max) for img in images_test])
    print(f"空洞阈值 {hole_max}：认出 0 {(pred & is_zero).sum():>2} 张，误判 {(pred & ~is_zero).sum():>2} 张")

# ---- 代码 3 ----
pred = np.array([looks_like_zero(img) for img in images_test])
print("数字 0–9 各有几张被误判成 0:", np.bincount(y_test[pred & ~is_zero], minlength=10))

# ---- 代码 4 ----
from sklearn.linear_model import LogisticRegression

X_train = images_train.reshape(len(images_train), -1)   # 8×8 拉平成 64 个特征
X_test = images_test.reshape(len(images_test), -1)

clf = LogisticRegression(max_iter=1000).fit(X_train, y_train == 0)
pred = clf.predict(X_test)
print(f"认出 0 {(pred & is_zero).sum()} 张，误判 {(pred & ~is_zero).sum()} 张")

# ---- 代码 5 ----
from sklearn.tree import DecisionTreeClassifier

model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print(f"10 个数字一起分，逻辑回归测试集准确率 {model.score(X_test, y_test):.3f}")

tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
print(f"决策树学出 {tree.get_n_leaves()} 条 if/else 规则，测试集准确率 {tree.score(X_test, y_test):.3f}")
