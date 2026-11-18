"""D3 关联规则：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/d3_association_rules/d3_association_rules.py
"""

# ---- 代码 1 ----
from collections import Counter

baskets = [
    "面包 牛奶",
    "面包 尿布 啤酒 鸡蛋",
    "牛奶 尿布 啤酒 可乐",
    "面包 牛奶 尿布 啤酒",
    "面包 牛奶 尿布 可乐",
    "面包 牛奶 鸡蛋",
    "尿布 啤酒",
    "面包 牛奶 尿布",
    "面包 鸡蛋",
    "牛奶 尿布 啤酒",
    "面包 牛奶 可乐",
    "面包 尿布 啤酒",
]
transactions = [frozenset(b.split()) for b in baskets]
n = len(transactions)
item_counts = Counter(item for t in transactions for item in t)
for item, c in sorted(item_counts.items(), key=lambda x: (-x[1], x[0])):
    print(f"{item}: 出现 {c} 次，支持度 {c / n:.3f}")

# ---- 代码 2 ----
def support(itemset):
    return sum(frozenset(itemset) <= t for t in transactions) / n

def rule_metrics(lhs, rhs):
    s = support(lhs | rhs)
    conf = s / support(lhs)
    lift = conf / support(rhs)
    return s, conf, lift

for lhs, rhs in [({"尿布"}, {"啤酒"}), ({"啤酒"}, {"尿布"}), ({"尿布"}, {"面包"})]:
    s, conf, lift = rule_metrics(lhs, rhs)
    print(f"{lhs} -> {rhs}: 支持度 {s:.3f}，置信度 {conf:.3f}，提升度 {lift:.3f}")

# ---- 代码 3 ----
from itertools import combinations

items = sorted(item_counts)
all_itemsets = [frozenset(c) for k in range(1, len(items) + 1) for c in combinations(items, k)]
brute = [s for s in all_itemsets if support(s) >= 0.25]
print(f"{len(items)} 种商品，非空项集 {len(all_itemsets)} 个，频繁项集 {len(brute)} 个")
print(f"50 种商品的项集数：{2 ** 50 - 1:,}")

# ---- 代码 4 ----
import pandas as pd

def apriori(transactions, min_support):
    min_count = min_support * len(transactions)
    counts = Counter(item for t in transactions for item in t)
    level = {frozenset([i]): c for i, c in counts.items() if c >= min_count}
    frequent = dict(level)
    stats = [{"k": 1, "连接得到": len(counts), "剪枝后": len(counts), "频繁": len(level)}]
    k = 2
    while level:
        prev = sorted(tuple(sorted(s)) for s in level)
        # 连接：前 k-2 个商品相同的两个频繁 (k-1)-项集合并成一个 k-项集
        joined = {frozenset(a + b[-1:]) for a, b in combinations(prev, 2) if a[:-1] == b[:-1]}
        # 剪枝：只要有一个 (k-1)-子集不频繁，就不必再数
        cands = [c for c in joined if all(frozenset(s) in level for s in combinations(c, k - 1))]
        counts = Counter(c for t in transactions for c in cands if c <= t)   # 扫一遍数据库
        level = {c: counts[c] for c in cands if counts[c] >= min_count}
        if joined:
            stats.append({"k": k, "连接得到": len(joined), "剪枝后": len(cands), "频繁": len(level)})
        frequent.update(level)
        k += 1
    return frequent, pd.DataFrame(stats)

frequent, stats = apriori(transactions, min_support=0.25)
print(stats.to_string(index=False))
for s, c in sorted(frequent.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
    if len(s) > 1:
        print("、".join(sorted(s)), c)
print(set(frequent) == set(brute))

# ---- 代码 5 ----
def gen_rules(frequent, n, min_conf):
    rows = []
    for itemset, cnt in frequent.items():
        for r in range(1, len(itemset)):
            for lhs in map(frozenset, combinations(itemset, r)):
                rhs = itemset - lhs
                conf = cnt / frequent[lhs]          # 子集一定也在 frequent 里
                if conf >= min_conf:
                    rows.append({"前件": "、".join(sorted(lhs)), "后件": "、".join(sorted(rhs)),
                                 "支持度": cnt / n, "置信度": conf,
                                 "提升度": conf / (frequent[rhs] / n)})
    rules = pd.DataFrame(rows, columns=["前件", "后件", "支持度", "置信度", "提升度"])
    return rules.sort_values(["提升度", "置信度", "前件", "后件"],
                             ascending=[False, False, True, True]).reset_index(drop=True)

rules = gen_rules(frequent, n, min_conf=0.6)
print(rules.round(3))

# ---- 代码 6 ----
import numpy as np
from math import comb

rng = np.random.default_rng(42)
n_items, n_trans = 50, 2000
names = np.array([f"g{i:02d}" for i in range(n_items)])
popularity = 0.3 * 0.92 ** np.arange(n_items)          # g00 最热门，越往后越冷门
M = rng.random((n_trans, n_items)) < popularity        # 各商品独立地被买
bundles = [[5, 12, 23], [8, 30]]                       # 埋进去的两组“搭配购买”
for b, prob in zip(bundles, [0.08, 0.06]):
    hit = rng.random(n_trans) < prob
    M[np.ix_(hit, b)] = True
big = [frozenset(names[row]) for row in M]
print(f"{len(big)} 条交易，{n_items} 种商品，平均每单 {M.sum(axis=1).mean():.2f} 件")

big_frequent, big_stats = apriori(big, min_support=0.02)
big_stats["暴力枚举"] = [comb(n_items, k) for k in big_stats["k"]]
print(big_stats.to_string(index=False))

# ---- 代码 7 ----
bundle_sets = [frozenset(names[b]) for b in bundles]

def only_bundle(row):
    s = set(row["前件"].split("、")) | set(row["后件"].split("、"))
    return any(s <= b for b in bundle_sets)

for ms in [0.05, 0.02, 0.01, 0.005]:
    f, _ = apriori(big, ms)
    r = gen_rules(f, len(big), min_conf=0.5)
    print(f"最小支持度 {ms:<5}: 频繁项集 {len(f):>3}，规则 {len(r):>3}，"
          f"其中只含搭配商品的 {r.apply(only_bundle, axis=1).sum()}")

# ---- 代码 8 ----
order = sorted((i for i, c in item_counts.items() if c >= 3), key=lambda i: (-item_counts[i], i))
print(" > ".join(order))
paths = [tuple(i for i in order if i in t) for t in transactions]
nodes = Counter(p[:d] for p in paths for d in range(1, len(p) + 1))
for prefix, c in sorted(nodes.items()):
    print("    " * (len(prefix) - 1) + f"{prefix[-1]}:{c}")
print(f"{sum(len(p) for p in paths)} 次商品出现，压缩成 {len(nodes)} 个节点")
