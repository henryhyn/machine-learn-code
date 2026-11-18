"""A3 够用的数学：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/a3_math_essentials/a3_math_essentials.py
"""

# ---- 代码 1 ----
import numpy as np

X = np.array([[89, 3, 12],      # 每行一套房子：面积 (平方米)、房间数、房龄 (年)
              [120, 4, 5],
              [60, 2, 20],
              [95, 3, 8]])
w = np.array([1.2, 5.0, -0.8])  # 每个特征的权重
b = 10.0

x0 = X[0]
print(f"第一套房子 x = {x0}，形状 {x0.shape}；数据集形状 {X.shape}")
print(f"逐项相乘再相加: {(x0 * w).sum() + b:.1f}，点积: {x0 @ w + b:.1f}")
print("一次算完 4 套:", X @ w + b)
print("X 的转置形状:", X.T.shape, "，XᵀX 的形状:", (X.T @ X).shape)
print("XᵀX 左上角 = 面积列与自己的点积:", (X.T @ X)[0, 0], "=", X[:, 0] @ X[:, 0])

# ---- 代码 2 ----
v = np.array([3.0, -4.0, 0.0])
print(f"L2 范数 {np.linalg.norm(v):.1f}，L1 范数 {np.linalg.norm(v, 1):.1f}")

# ---- 代码 3 ----
rng = np.random.default_rng(0)
x1 = rng.normal(size=50)
for noise in [1.0, 0.1, 0.01]:
    A = np.column_stack([x1, x1 + rng.normal(0, noise, 50)])     # 第二列 ≈ 第一列
    y = A @ np.array([2.0, 2.0]) + rng.normal(0, 0.5, 50)          # 真实系数 (2, 2)
    c1 = np.linalg.lstsq(A, y)[0]
    c2 = np.linalg.lstsq(A, y + rng.normal(0, 0.01, 50))[0]        # y 只抖动一点点
    print(f"噪声 {noise:<4}: cond(A) {np.linalg.cond(A):6.1f}，cond(AᵀA) {np.linalg.cond(A.T @ A):8.1f}，"
          f"系数 ({c1[0]:5.2f}, {c1[1]:5.2f}) → 抖动后 ({c2[0]:5.2f}, {c2[1]:5.2f})")

# ---- 代码 4 ----
f = lambda x: (x - 3) ** 2
for h in [0.1, 0.001]:
    print(f"h = {h:<5}: (f(1+h) - f(1)) / h = {(f(1 + h) - f(1)) / h:.4f}")
print(f"公式 f'(x) = 2(x - 3)，在 x = 1 处为 {2 * (1 - 3)}")

# ---- 代码 5 ----
def f2(w):
    return w[0] ** 2 + 3 * w[1] ** 2

w = np.array([1.0, 1.0])
g = np.array([2 * w[0], 6 * w[1]])           # 梯度
print("梯度:", g)
for name, d in [("沿梯度", g), ("沿 w1 轴", np.array([1.0, 0.0])),
                ("沿 w2 轴", np.array([0.0, 1.0])), ("沿负梯度", -g)]:
    d = d / np.linalg.norm(d)                # 只比方向，步长都是 0.01
    print(f"{name:<6}走 0.01: f 变化 {f2(w + 0.01 * d) - f2(w):+.4f}")

# ---- 代码 6 ----
for lr in [0.01, 0.1, 0.5, 0.9, 1.1]:
    x, path = 0.0, []
    for step in range(20):
        x -= lr * 2 * (x - 3)                # 导数是 2(x - 3)
        path.append(x)
    print(f"学习率 {lr:<4}: 前 3 步 {', '.join(f'{p:.2f}' for p in path[:3])}，第 20 步 x = {x:.4f}")

# ---- 代码 7 ----
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z, h = 0.5, 1e-6
loss = lambda z: -np.log(sigmoid(z))         # 标签为 1 时的交叉熵
print(f"σ 的数值导数 {(sigmoid(z + h) - sigmoid(z - h)) / (2 * h):.6f}，公式 σ(1-σ) = {sigmoid(z) * (1 - sigmoid(z)):.6f}")
print(f"损失的数值导数 {(loss(z + h) - loss(z - h)) / (2 * h):.6f}，链式法则 σ(z) - 1 = {sigmoid(z) - 1:.6f}")

# ---- 代码 8 ----
f4 = lambda x: x ** 4 - 3 * x ** 2 + x
df4 = lambda x: 4 * x ** 3 - 6 * x + 1
for x0 in [-2.0, 0.5, 2.0]:
    x = x0
    for _ in range(1000):
        x -= 0.01 * df4(x)
    print(f"从 {x0:4.1f} 出发: 停在 x = {x:6.3f}，f(x) = {f4(x):6.3f}")

# ---- 代码 9 ----
rng = np.random.default_rng(42)
faces = np.arange(1, 7)
E = faces.mean()                             # 六个面等概率，期望就是简单平均
Var = ((faces - E) ** 2).mean()
rolls = rng.integers(1, 7, size=100_000)     # 掷 10 万次
print(f"理论: 期望 {E:.3f}，方差 {Var:.3f}，标准差 {np.sqrt(Var):.3f}")
print(f"模拟: 期望 {rolls.mean():.3f}，方差 {rolls.var():.3f}，标准差 {rolls.std():.3f}")

data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
print(f"标准差 除以 n: {data.std():.3f}，除以 n-1: {data.std(ddof=1):.3f}")

# ---- 代码 10 ----
def p_sick_given_positive(prior, sens=0.99, fpr=0.05):
    p_pos = sens * prior + fpr * (1 - prior)          # 全概率公式：阳性的两种来源
    return sens * prior / p_pos                         # 贝叶斯公式

for prior in [0.001, 0.01, 0.1]:
    print(f"患病率 {prior:>5.1%}: 阳性者真患病的概率 {p_sick_given_positive(prior):.3f}")

n = 1_000_000
sick = rng.random(n) < 0.01
positive = np.where(sick, rng.random(n) < 0.99, rng.random(n) < 0.05)
print(f"模拟 {n} 人: 阳性 {positive.sum()} 人，其中真患病 {(sick & positive).sum()} 人，占 {sick[positive].mean():.3f}")

# ---- 代码 11 ----
p = np.linspace(0.001, 0.999, 999)
heads, n = 7, 10
lik = p ** heads * (1 - p) ** (n - heads)
loglik = heads * np.log(p) + (n - heads) * np.log(1 - p)
print(f"似然最大处 p = {p[lik.argmax()]:.3f}，对数似然最大处 p = {p[loglik.argmax()]:.3f}")

flips = rng.random(2000) < 0.3                         # 一枚正面概率 0.3 的硬币抛 2000 次
h = flips.sum()
lik2 = p ** h * (1 - p) ** (2000 - h)
loglik2 = h * np.log(p) + (2000 - h) * np.log(1 - p)
print(f"2000 次中 {h} 次正面：似然的最大值 {lik2.max()}，对数似然最大处 p = {p[loglik2.argmax()]:.3f}")

# ---- 代码 12 ----
import math

z = rng.normal(0, 1, 1_000_000)
for k in [1, 2, 3]:
    print(f"±{k}σ 以内: 模拟 {np.mean(np.abs(z) < k):.4f}，理论 {math.erf(k / math.sqrt(2)):.4f}")

# ---- 代码 13 ----
for n in [10, 100, 10_000]:
    print(f"n = {n:>6}: 某个样本一次都没被抽中的概率 (1 - 1/n)^n = {(1 - 1 / n) ** n:.4f}")
print(f"1/e = {1 / np.e:.4f}")

# ---- 代码 14 ----
def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]                             # 约定 0·log0 = 0
    return float((p * np.log2(1 / p)).sum())

for dist in [[0.5, 0.5], [0.9, 0.1], [1.0, 0.0], [0.25] * 4]:
    print(f"{str(dist):<26} 熵 {entropy(dist):.3f} 比特")

# ---- 代码 15 ----
for q in [0.99, 0.9, 0.5, 0.1, 0.01]:
    print(f"真实标签 1，预测概率 {q:<4}: 交叉熵 {-np.log(q):.3f}")

y_true = np.array([1, 0, 1, 1, 0])
p_pred = np.array([0.9, 0.2, 0.6, 0.95, 0.1])
ll = -np.mean(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred))
print(f"5 个样本的平均交叉熵 (对数损失): {ll:.4f}")

# ---- 代码 16 ----
def cosine(u, v):
    return u @ v / (np.linalg.norm(u) * np.linalg.norm(v))

a = np.array([3.0, 0.0, 1.0])
docs = {"b": np.array([6.0, 0.0, 2.0]),     # a 抄两遍：方向相同，长度翻倍
        "c": np.array([0.0, 3.0, 1.0])}     # 讲的是别的
for name, v in docs.items():
    print(f"a 与 {name}: 欧氏 {np.linalg.norm(a - v):.3f}，曼哈顿 {np.abs(a - v).sum():.1f}，余弦相似度 {cosine(a, v):.3f}")

# ---- 代码 17 ----
people = np.array([[25, 8000], [26, 12000], [45, 8200]], dtype=float)   # 年龄、月收入
def dist_from_A(M):
    return np.linalg.norm(M[1:] - M[0], axis=1).round(2)

print("原始单位，A 到 B、C:", dist_from_A(people))
print("收入换成千元:      ", dist_from_A(people / [1, 1000]))
print("各除以典型标准差:  ", dist_from_A(people / [10, 3000]))      # 假设年龄标准差 10、收入 3000
