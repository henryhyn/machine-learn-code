"""手写 Apriori 和规则生成，算法和书中 D3 一致 (连接 → 剪枝 → 扫描计数 → 逐层)，
区别只在“扫描计数”这一步用 numpy 布尔矩阵实现，几万张发票、几千种商品也能在几秒内跑完。

交易用一个布尔矩阵 X 表示：X[i, j] 为 True 表示第 i 张发票里有第 j 种商品。
项集用排好序的列号元组表示，例如 (3, 17)。
"""
from itertools import combinations

import numpy as np
import pandas as pd


def apriori(X, min_support, max_len=None):
    """返回 (frequent, stats)。frequent 是 {项集元组: 出现次数}，stats 记录每一层的候选数。"""
    n_trans, n_items = X.shape
    min_count = min_support * n_trans
    counts = X.sum(axis=0)
    level = {(j,): int(counts[j]) for j in np.flatnonzero(counts >= min_count)}
    frequent = dict(level)
    stats = [{"k": 1, "joined": n_items, "pruned": n_items, "frequent": len(level)}]
    k = 2
    while level and (max_len is None or k <= max_len):
        # 连接：前 k-2 项相同的两个频繁 (k-1)-项集合并。按前缀分组后，组内两两组合即可
        groups = {}
        for s in sorted(level):
            groups.setdefault(s[:-1], []).append(s[-1])
        joined = [p + (a, b) for p, lasts in groups.items() for a, b in combinations(lasts, 2)]
        # 剪枝：任何一个 (k-1)-子集不频繁，就不必再数 (反单调性)
        cands = [c for c in joined if all(sub in level for sub in combinations(c, k - 1))]
        # 扫描计数：按“前 k-1 项”分组，先找出含这 k-1 项的发票，再在这些行里数最后一项
        by_prefix = {}
        for c in cands:
            by_prefix.setdefault(c[:-1], []).append(c[-1])
        new_level = {}
        for prefix, lasts in by_prefix.items():
            rows = X[:, prefix].all(axis=1)
            cnt = X[rows][:, lasts].sum(axis=0)
            for j, c in zip(lasts, cnt):
                if c >= min_count:
                    new_level[prefix + (j,)] = int(c)
        if joined:
            stats.append({"k": k, "joined": len(joined), "pruned": len(cands), "frequent": len(new_level)})
        level = new_level
        frequent.update(level)
        k += 1
    return frequent, pd.DataFrame(stats)


def gen_rules(frequent, n_trans, min_conf):
    """和 D3 一样：对每个频繁项集枚举前件，置信度 = count(Z) / count(X)，直接查表。"""
    rows = []
    for itemset, cnt in frequent.items():
        for r in range(1, len(itemset)):
            for lhs in combinations(itemset, r):
                rhs = tuple(i for i in itemset if i not in lhs)
                conf = cnt / frequent[lhs]                      # 子集一定也在 frequent 里
                if conf >= min_conf:
                    rows.append({"lhs": lhs, "rhs": rhs, "support": cnt / n_trans, "confidence": conf,
                                 "lift": conf / (frequent[rhs] / n_trans)})
    rules = pd.DataFrame(rows, columns=["lhs", "rhs", "support", "confidence", "lift"])
    return rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
