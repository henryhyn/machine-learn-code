"""m3 购物篮分析：从清洗后的交易构建“发票 × 商品”的篮子，用手写 Apriori (apriori.py) 挖关联规则。

- 篮子：m1 里那 4320 个客户的全部购买发票 (退货发票不算篮子)，所有国家都用 (九成客户在英国)；
  一张发票一个篮子，只记“买没买”某件商品 (StockCode)，显示时用该 StockCode 最常见的 Description；
- 主设置：min_support = 0.01 (至少 184 张发票)，min_conf = 0.5；
- 看三件事：提升度最高的规则；热门商品让置信度虚高 (按置信度排 vs 按提升度排)；阈值敏感性；
- 最后按 m2 的客户群分别挖一遍，看不同群的规则有什么不同。

运行：uv run python projects/p2_retail/m3_basket.py
"""
import time

import numpy as np
import pandas as pd

from apriori import apriori, gen_rules
from common import compute_rfm, load_clean, report
from m2_kmeans import CLUSTER_NAMES, fit_segments

MIN_SUPPORT, MIN_CONF = 0.01, 0.5
SEGMENT_MIN_SUPPORT = 0.02
SEASON_START = "2011-09-01"


def build_baskets(buy):
    """返回 (X, invoice_ids, names)：X[i, j] 表示第 i 张发票里有第 j 种商品。"""
    inv = buy["InvoiceNo"].astype("category")
    item = buy["StockCode"].astype("category")
    X = np.zeros((len(inv.cat.categories), len(item.cat.categories)), dtype=bool)
    X[inv.cat.codes, item.cat.codes] = True
    desc = buy.groupby("StockCode")["Description"].agg(lambda s: s.value_counts().index[0].strip())
    return X, inv.cat.categories, desc.loc[item.cat.categories].to_numpy()


def readable(rules, names):
    out = rules.copy()
    out["lhs_name"] = out["lhs"].map(lambda t: " + ".join(names[list(t)]))
    out["rhs_name"] = out["rhs"].map(lambda t: " + ".join(names[list(t)]))
    return out


def rule_rows(rules, n):
    return [{"lhs": r.lhs_name, "rhs": r.rhs_name, "support": round(float(r.support), 3),
             "confidence": round(float(r.confidence), 3), "lift": round(float(r.lift), 1)} for r in rules.head(n).itertuples()]


def main():
    df = load_clean()
    rfm = compute_rfm(df)
    buy = df[~df["IsReturn"] & df["CustomerID"].isin(rfm.index)]
    X, invoices, names = build_baskets(buy)
    n = len(X)
    sizes = X.sum(axis=1)
    sup = X.mean(axis=0)
    print(f"{n} 张发票 (篮子)，{X.shape[1]} 种商品；每篮商品数中位数 {np.median(sizes):.0f}，"
          f"平均 {sizes.mean():.1f}，90% 分位数 {np.percentile(sizes, 90):.0f}\n")
    hot = np.argsort(-sup)[:10]
    print("最热门的 5 种商品：")
    for j in hot[:5]:
        print(f"  {sup[j]:.3f}  {names[j]}")

    # 1. 主设置下的频繁项集和规则
    t0 = time.perf_counter()
    frequent, stats = apriori(X, MIN_SUPPORT)
    rules = readable(gen_rules(frequent, n, MIN_CONF), names)
    print(f"\nmin_support={MIN_SUPPORT}、min_conf={MIN_CONF}：频繁项集 {len(frequent)} 个，规则 {len(rules)} 条，"
          f"用时 {time.perf_counter() - t0:.1f} 秒")
    print(stats.to_string(index=False))
    # 自检：2-项集的计数和矩阵乘法 X^T X 的结果一致
    Xf = X.astype(np.int32)
    assert all(frequent[s] == int(Xf[:, s[0]] @ Xf[:, s[1]]) for s in frequent if len(s) == 2)

    # 同一个项集拆出的规则 (A⇒B 和 B⇒A) 提升度相同，展示时每个项集只留置信度最高的一条
    itemset = rules.apply(lambda r: tuple(sorted(r.lhs + r.rhs)), axis=1)
    top_rules = rules.assign(itemset=itemset).sort_values(["lift", "confidence"], ascending=False) \
        .drop_duplicates("itemset")
    print("\n提升度最高的规则 (每个项集一条)：")
    print(top_rules.head(8)[["lhs_name", "rhs_name", "support", "confidence", "lift"]].round(3).to_string(index=False))

    pairs = [(s, c) for s, c in frequent.items() if len(s) == 2]
    pair_lifts = np.array([c * n / (frequent[s[:1]] * frequent[s[1:]]) for s, c in pairs])
    print(f"\n{len(pairs)} 个频繁 2-项集，提升度最小 {pair_lifts.min():.2f}，小于 1 的 {(pair_lifts < 1).sum()} 个")

    # 2. 热门商品让置信度虚高：单个后件的全部规则 (不设 min_conf)，分别按置信度和提升度取前 50
    single = readable(gen_rules(frequent, n, 0.0), names)
    single = single[single["rhs"].map(len) == 1].copy()
    single["rhs_support"] = single["rhs"].map(lambda t: sup[t[0]])
    single["rhs_hot"] = single["rhs"].map(lambda t: t[0] in set(hot))
    hot_stats = {}
    for col, zh in [("confidence", "置信度"), ("lift", "提升度")]:
        top = single.sort_values([col, "support"], ascending=False).head(50)
        hot_stats[f"top50_by_{col}"] = {"hot_rhs": int(top["rhs_hot"].sum()),
                                        "rhs_support_median": round(float(top["rhs_support"].median()), 3)}
        print(f"按{zh}取前 50 条：后件是前 10 热门商品的 {int(top['rhs_hot'].sum())} 条，"
              f"后件支持度中位数 {top['rhs_support'].median():.3f}")
    # 一个具体例子：同一个前件，按置信度和按提升度选出的后件不同，前者是热门商品
    examples = []
    for lhs, grp in single[single["lhs"].map(len) == 1].groupby("lhs"):
        by_conf = grp.sort_values("confidence").iloc[-1]
        by_lift = grp.sort_values("lift").iloc[-1]
        if by_conf["rhs_hot"] and by_conf["rhs"] != by_lift["rhs"]:
            examples.append((by_lift["lift"], by_conf, by_lift))
    n_diff = sum(g.sort_values("confidence")["rhs"].iloc[-1] != g.sort_values("lift")["rhs"].iloc[-1]
                 for _, g in single.groupby("lhs"))
    _, by_conf, by_lift = max(examples, key=lambda e: e[0])
    example = {"lhs": by_conf["lhs_name"],
               "best_by_confidence": {"rhs": by_conf["rhs_name"], "confidence": round(float(by_conf["confidence"]), 3),
                                      "lift": round(float(by_conf["lift"]), 1),
                                      "rhs_support": round(float(by_conf["rhs_support"]), 3)},
               "best_by_lift": {"rhs": by_lift["rhs_name"], "confidence": round(float(by_lift["confidence"]), 3),
                                "lift": round(float(by_lift["lift"]), 1),
                                "rhs_support": round(float(by_lift["rhs_support"]), 3)}}
    print(f"{single['lhs'].nunique()} 个前件里，有 {n_diff} 个按置信度和按提升度选出的最佳后件不同。例：")
    for key in ["best_by_confidence", "best_by_lift"]:
        e = example[key]
        print(f"  {example['lhs']} ⇒ {e['rhs']}：置信度 {e['confidence']:.3f}，提升度 {e['lift']:.1f}，"
              f"后件支持度 {e['rhs_support']:.3f}")

    # 3. 阈值敏感性
    print()
    sensitivity = []
    for ms in [0.03, 0.02, 0.015, 0.01, 0.0075]:
        t0 = time.perf_counter()
        f, st = apriori(X, ms)
        r = gen_rules(f, n, MIN_CONF)
        sensitivity.append({"min_support": ms, "min_count": int(np.ceil(ms * n)), "items": int(st["frequent"].iloc[0]),
                            "counted": int(st["pruned"].sum()), "frequent": len(f),
                            "max_len": max(len(s) for s in f), "rules": len(r)})
        print(f"min_support={ms:<6}: 频繁单品 {sensitivity[-1]['items']:>4}，数过的候选 {st['pruned'].sum():>6}，"
              f"频繁项集 {len(f):>5}，最长 {sensitivity[-1]['max_len']}，规则 {len(r):>5}，"
              f"用时 {time.perf_counter() - t0:.1f} 秒")

    # 4. 按 m2 的客户群分别挖
    _, cluster = fit_segments(rfm)
    first = buy.drop_duplicates("InvoiceNo").set_index("InvoiceNo")
    inv_cluster = first["CustomerID"].map(cluster).loc[invoices].to_numpy()
    inv_recent = (first["InvoiceDate"] >= SEASON_START).loc[invoices].to_numpy()
    print(f"\n按客户群分别挖 (min_support={SEGMENT_MIN_SUPPORT}、min_conf={MIN_CONF})：")
    segments = []
    for c in CLUSTER_NAMES:
        m = inv_cluster == c
        f, _ = apriori(X[m], SEGMENT_MIN_SUPPORT)
        r = readable(gen_rules(f, int(m.sum()), MIN_CONF), names)
        xmas = r.apply(lambda x: "CHRISTMAS" in x.lhs_name + x.rhs_name, axis=1).mean() if len(r) else 0.0
        segments.append({"cluster": c, "invoices": int(m.sum()),
                         "share_since_sep": round(float(inv_recent[m].mean()), 3),
                         "frequent": len(f), "rules": len(r), "christmas_rule_share": round(float(xmas), 3),
                         "top_rule": rule_rows(r, 1)[0]})
        print(f"  {c}: 发票 {m.sum()}，其中 9 月以后 {inv_recent[m].mean():.3f}；频繁项集 {len(f)}，"
              f"规则 {len(r)}，含 CHRISTMAS 的规则占 {xmas:.3f}")
        print(f"      提升度最高：{r.iloc[0].lhs_name} ⇒ {r.iloc[0].rhs_name} (lift {r.iloc[0].lift:.1f})")

    report("m3", {
        "invoices": n,
        "items": int(X.shape[1]),
        "basket_size_median": float(np.median(sizes)),
        "basket_size_mean": round(float(sizes.mean()), 1),
        "basket_size_p90": float(np.percentile(sizes, 90)),
        "hot_items": [{"item": names[j], "support": round(float(sup[j]), 3)} for j in hot[:5]],
        "min_support": MIN_SUPPORT,
        "min_conf": MIN_CONF,
        "level_stats": stats.to_dict("records"),
        "frequent": len(frequent),
        "rules": len(rules),
        "top_rules": rule_rows(top_rules, 8),
        "frequent_pairs": len(pairs),
        "frequent_pairs_min_lift": round(float(pair_lifts.min()), 2),
        "frequent_pairs_lift_below_1": int((pair_lifts < 1).sum()),
        "hot_rhs": hot_stats,
        "antecedents": int(single["lhs"].nunique()),
        "antecedents_conf_lift_disagree": int(n_diff),
        "hot_example": example,
        "sensitivity": sensitivity,
        "segments": segments,
    })


if __name__ == "__main__":
    main()
