"""C4 朴素贝叶斯：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/c4_naive_bayes/c4_naive_bayes.py
"""

# ---- 代码 1 ----
from collections import Counter

train = [
    ("免费 领取 红包", "垃圾"),
    ("恭喜 中奖 免费 领取 大奖", "垃圾"),
    ("点击 链接 领取 优惠", "垃圾"),
    ("明天 开会 记得 带 电脑", "正常"),
    ("晚上 一起 吃饭", "正常"),
    ("红包 收到 了 谢谢", "正常"),
]
counts = {"垃圾": Counter(), "正常": Counter()}
for text, label in train:
    counts[label].update(text.split())
vocab = set(counts["垃圾"]) | set(counts["正常"])
print(f"词表 {len(vocab)} 个词；垃圾类共 {sum(counts['垃圾'].values())} 个词，正常类共 {sum(counts['正常'].values())} 个词")

def score(words, label, alpha):
    total = sum(counts[label].values())
    s, parts = 0.5, []                                   # 先验：两类各 3 条，都是 0.5
    for w in words:
        num, den = counts[label][w] + alpha, total + alpha * len(vocab)
        parts.append(f"{num}/{den}")
        s *= num / den
    return s, parts

msg = "明天 领取 红包".split()
for alpha in [0, 1]:
    print(f"alpha = {alpha}")
    for label in ["垃圾", "正常"]:
        s, parts = score(msg, label, alpha)
        print(f"  {label}: 0.5 × " + " × ".join(parts) + f" = {s:.6f}")

# ---- 代码 2 ----
import numpy as np

p_spam = np.full(300, 0.010)      # 一条 300 词的长消息，每个词在垃圾类下的概率
p_ham = np.full(300, 0.008)       # 同样 300 个词在正常类下的概率
print(f"直接连乘：垃圾 {np.prod(p_spam)}，正常 {np.prod(p_ham)}")
print(f"取对数求和：垃圾 {np.log(p_spam).sum():.1f}，正常 {np.log(p_ham).sum():.1f}")

# ---- 代码 3 ----
import pandas as pd

spam_phrases = ["恭喜 您 中奖 了", "点击 链接 领取 红包", "限时 优惠 名额 有限", "回复 TD 退订",
                "免费 领取 话费", "积分 即将 过期 请 兑换", "低息 贷款 当天 到账", "会员 专享 五折 优惠",
                "点击 链接 查看 详情", "免费 试用 三天", "您 的 账户 有 一笔 奖金 待 领取", "转发 本 短信 领 红包"]
ham_phrases = ["明天 上午 开会", "晚上 一起 吃饭 吗", "快递 放 门口 了", "记得 交 作业",
               "我 到家 了", "周末 去 看 电影 吧", "妈妈 让 你 回 电话", "路上 堵车 晚 点 到",
               "老师 说 明天 考试", "下班 帮 我 买 牛奶", "会议 改 到 下午 三点", "收到 谢谢"]
shared_phrases = ["你好", "今天 有空 吗", "请 尽快 回复", "详情 见 群 里", "明天 截止", "群 里 抢 红包 了",
                  "这家 店 有 优惠", "点击 看 照片", "免费 停车", "账户 密码 别 告诉 别人"]

rng = np.random.default_rng(42)

def make_msg(is_spam):
    own = spam_phrases if is_spam else ham_phrases
    parts = [own[rng.integers(len(own))] if rng.random() < 0.7
             else shared_phrases[rng.integers(len(shared_phrases))]
             for _ in range(rng.integers(1, 4))]          # 每条 1–3 个短语
    return " ".join(parts)

y_text = (rng.random(400) < 0.4).astype(int)              # 1 = 垃圾短信，约占 40%
docs = [make_msg(s) for s in y_text]
for d, s in list(zip(docs, y_text))[3:7]:
    print("垃圾" if s else "正常", "|", d)
print(pd.Series(y_text).value_counts().to_dict())

# ---- 代码 4 ----
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

print(f"默认 token_pattern：词表 {len(CountVectorizer().fit(docs).vocabulary_)} 个词")
d_train, d_test, yt_train, yt_test = train_test_split(
    docs, y_text, test_size=0.3, stratify=y_text, random_state=0)
cv = CountVectorizer(token_pattern=r"(?u)\S+")          # 按空格切，单字词也保留
Xt_train = cv.fit_transform(d_train)
Xt_test = cv.transform(d_test)
density = Xt_train.nnz / (Xt_train.shape[0] * Xt_train.shape[1])
print(f"按空格切：词表 {Xt_train.shape[1]} 个词，训练矩阵 {Xt_train.shape}，{type(Xt_train).__name__}，非零占 {density:.1%}")

# ---- 代码 5 ----
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB().fit(Xt_train, yt_train)
print(f"测试集准确率 {nb.score(Xt_test, yt_test):.3f}")
print("分错的消息：")
for d, t, p in zip(d_test, yt_test, nb.predict(Xt_test)):
    if t != p:
        print(f"  真实 {'垃圾' if t else '正常'}，预测 {'垃圾' if p else '正常'} | {d}")

# ---- 代码 6 ----
words = cv.get_feature_names_out()
ratio = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]     # log P(词|垃圾) - log P(词|正常)
order = np.argsort(ratio)
print("最像垃圾短信的词：", "  ".join(f"{words[i]} {ratio[i]:+.2f}" for i in order[::-1][:6]))
print("最像正常短信的词：", "  ".join(f"{words[i]} {ratio[i]:+.2f}" for i in order[:6]))

# ---- 代码 7 ----
from scipy.special import logsumexp

class MyMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        counts = np.array([np.asarray(X[y == c].sum(axis=0)).ravel() for c in self.classes_])
        smoothed = counts + self.alpha                                  # 拉普拉斯平滑
        self.feature_log_prob_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
        self.class_log_prior_ = np.log(np.array([np.mean(y == c) for c in self.classes_]))
        return self

    def joint_log_likelihood(self, X):
        return X @ self.feature_log_prob_.T + self.class_log_prior_     # 对数空间里连乘变求和

    def predict(self, X):
        return self.classes_[self.joint_log_likelihood(X).argmax(axis=1)]

    def predict_proba(self, X):
        jll = self.joint_log_likelihood(X)
        return np.exp(jll - logsumexp(jll, axis=1, keepdims=True))     # 归一化也在对数空间做

mine = MyMultinomialNB().fit(Xt_train, yt_train)
print(f"条件概率一致：{np.allclose(mine.feature_log_prob_, nb.feature_log_prob_)}")
print(f"预测一致：{(mine.predict(Xt_test) == nb.predict(Xt_test)).all()}")
print(f"概率最大差：{np.abs(mine.predict_proba(Xt_test) - nb.predict_proba(Xt_test)).max():.1e}")

# ---- 代码 8 ----
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, ComplementNB
from sklearn.linear_model import LogisticRegression

for name, model in [("MultinomialNB", MultinomialNB()), ("BernoulliNB", BernoulliNB()),
                    ("ComplementNB", ComplementNB()), ("LogisticRegression", LogisticRegression())]:
    acc = cross_val_score(model, Xt_train, yt_train, cv=5).mean()
    print(f"{name:<18} 5 折交叉验证准确率 {acc:.3f}")

# ---- 代码 9 ----
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
gnb = GaussianNB().fit(X_train, y_train)
print(f"测试集准确率 {gnb.score(X_test, y_test):.3f}")
print(f"第 1 个特征 (mean radius) 各类的均值 {gnb.theta_[:, 0].round(2)}，方差 {gnb.var_[:, 0].round(2)}")
print(f"var_smoothing 加到每个方差上的 epsilon_ = {gnb.epsilon_:.2e}")

# ---- 代码 10 ----
for alpha in [0.01, 1, 10, 100]:
    m = MultinomialNB(alpha=alpha)
    acc = cross_val_score(m, Xt_train, yt_train, cv=5).mean()
    conf = m.fit(Xt_train, yt_train).predict_proba(Xt_test).max(axis=1).mean()
    print(f"alpha = {alpha:>6}: 交叉验证准确率 {acc:.3f}，测试集平均最大概率 {conf:.3f}")

# ---- 代码 11 ----
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def report(name, model, Xtr, Xte):
    p = model.fit(Xtr, y_train).predict_proba(Xte)[:, 1]
    extreme = np.mean((p < 0.01) | (p > 0.99))
    wrong_sure = np.sum(((p > 0.5) != y_test) & ((p < 0.01) | (p > 0.99)))
    print(f"准确率 {np.mean((p > 0.5) == y_test):.3f}  Brier {brier_score_loss(y_test, p):.4f}  "
          f"对数损失 {log_loss(y_test, p):.3f}  极端概率 {extreme:>4.0%}  极端且错 {wrong_sure} | {name}")

report("GaussianNB", GaussianNB(), X_train, X_test)
report("GaussianNB 特征复制 3 份", GaussianNB(), np.tile(X_train, 3), np.tile(X_test, 3))
report("GaussianNB + sigmoid", CalibratedClassifierCV(GaussianNB(), cv=5), X_train, X_test)
report("GaussianNB + isotonic", CalibratedClassifierCV(GaussianNB(), cv=5, method="isotonic"), X_train, X_test)
report("逻辑回归", make_pipeline(StandardScaler(), LogisticRegression()), X_train, X_test)

# ---- 代码 12 ----
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(0)
P = rng.uniform(-1, 1, size=(1000, 2))
label = (P[:, 0] * P[:, 1] > 0).astype(int)          # 一、三象限为正类：异或
for name, model in [("GaussianNB", GaussianNB()), ("逻辑回归", LogisticRegression()),
                    ("kNN (k=15)", KNeighborsClassifier(n_neighbors=15))]:
    print(f"{name:<11} 交叉验证准确率 {cross_val_score(model, P, label, cv=5).mean():.3f}")
for c in [0, 1]:
    print(f"类 {c}：x1 均值 {P[label == c, 0].mean():+.3f}，x2 均值 {P[label == c, 1].mean():+.3f}")

# ---- 代码 13 ----
def learning_curve_nb_lr(X, y, nb_model, lr_model, sizes, n_rep=20):
    X_pool, X_te, y_pool, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    rng = np.random.default_rng(0)
    for n in sizes:
        acc_nb, acc_lr = [], []
        for _ in range(n_rep):                       # 每个样本量随机抽 n_rep 次取平均
            idx = rng.choice(len(X_pool), n, replace=False)
            while len(np.unique(y_pool[idx])) < 2:
                idx = rng.choice(len(X_pool), n, replace=False)
            acc_nb.append(nb_model.fit(X_pool[idx], y_pool[idx]).score(X_te, y_te))
            acc_lr.append(lr_model.fit(X_pool[idx], y_pool[idx]).score(X_te, y_te))
        print(f"  n = {n:>5}  朴素贝叶斯 {np.mean(acc_nb):.3f}  逻辑回归 {np.mean(acc_lr):.3f}")

print("乳腺癌数据 (GaussianNB)：")
learning_curve_nb_lr(X, y, GaussianNB(), make_pipeline(StandardScaler(), LogisticRegression()),
                     [10, 20, 40, 80, 160, 398])

rng = np.random.default_rng(1)
y_bin = rng.integers(0, 2, 12000)
p0 = rng.uniform(0.05, 0.3, 200)                     # 200 个二值特征，各自独立
p1 = p0.copy()
p1[:20] *= 1.8                                       # 只有前 20 个在两类间有差别
X_bin = (rng.random((12000, 200)) < np.where(y_bin[:, None] == 1, p1, p0)).astype(float)
print("200 个条件独立的二值特征 (BernoulliNB)：")
learning_curve_nb_lr(X_bin, y_bin, BernoulliNB(), LogisticRegression(max_iter=1000),
                     [20, 50, 200, 500, 2000, 8000])
