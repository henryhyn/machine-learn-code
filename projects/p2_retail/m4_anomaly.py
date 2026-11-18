"""m4 异常交易检测：在发票级特征上用孤立森林和 LOF 打分。没有真实标签，所以分三步评估：

(a) 人工检查：真实数据上分数最高的 20 张发票，按透明的规则归类，看它们是“问题”还是“少见但正常”；
(b) 注入异常：随机挑 1% 的发票，把其中一行的数量乘以 100 (或 10)，模拟多敲了零的录入错误，
    用 AP 和“复核前 1% 能抓到多少”量化各方法，并和单个特征的简单规则比；
(c) contamination 只动阈值：同一份注入数据上换 contamination，看标记数量和 AP (书中 D4)。

发票级特征 (只用购买发票，m1 的 4320 个客户)：商品种数、总件数、金额、单价中位数、最大一行的件数、
最大一行占金额的比例。前五个右偏，取对数；然后统一标准化。数据里没有深夜和周六的交易
(下单时间都在 6 点到 20 点之间)，所以不用“是否深夜”这类特征。

运行：uv run python projects/p2_retail/m4_anomaly.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from common import SEED, compute_rfm, load_clean, load_raw, report
from m2_kmeans import fit_segments

FEATURES = ["n_items", "qty", "amount", "med_price", "max_line_qty", "max_line_share"]
LOG_FEATURES = FEATURES[:5]
INJECT_RATE = 0.01
N_REPEATS = 5
TOP_N = 20


def invoice_features(lines):
    """每张发票一行。max_qty_vs_item 是“上下文”特征：最大的 log(该行件数 / 这件商品所有行的件数中位数)。"""
    lines = lines.assign(qty_vs_item=np.log(
        lines["Quantity"] / lines.groupby("StockCode")["Quantity"].transform("median")))
    g = lines.groupby("InvoiceNo")
    return pd.DataFrame({
        "customer": g["CustomerID"].first(),
        "date": g["InvoiceDate"].first(),
        "n_items": g["StockCode"].nunique(),
        "qty": g["Quantity"].sum(),
        "amount": g["Amount"].sum(),
        "med_price": g["UnitPrice"].median(),
        "max_line_qty": g["Quantity"].max(),
        "max_line_share": g["Amount"].max() / g["Amount"].sum(),
        "max_qty_vs_item": g["qty_vs_item"].max(),
    })


def preprocess():
    """前五列 log 后和最后一列一起标准化。用 FunctionTransformer 按列取对数，便于放进 Pipeline。"""
    def log_cols(X):
        X = np.asarray(X, dtype=float).copy()
        X[:, :5] = np.log(X[:, :5])
        return X
    return make_pipeline(FunctionTransformer(log_cols), StandardScaler())


def detectors():
    return {
        "孤立森林": make_pipeline(preprocess(), IsolationForest(random_state=SEED)),
        "LOF (n_neighbors=50)": make_pipeline(preprocess(), LocalOutlierFactor(n_neighbors=50)),
    }


def anomaly_scores(f):
    """统一成“越大越异常”。"""
    out = {}
    for name, model in detectors().items():
        model.fit(f[FEATURES])
        last = model[-1]
        out[name] = -(last.score_samples(model[:-1].transform(f[FEATURES])) if hasattr(last, "score_samples")
                      else last.negative_outlier_factor_)
    out["规则：金额"] = f["amount"].to_numpy()
    out["规则：某行件数 / 该商品常见件数"] = f["max_qty_vs_item"].to_numpy()
    return out


def categorize(f, top, returns):
    """把分数最高的发票按透明的规则归类 (按顺序，命中第一条为止)。"""
    cats = []
    for inv, r in f.loc[top].iterrows():
        later = returns[(returns["CustomerID"] == r["customer"]) & (returns["InvoiceDate"] >= r["date"]) &
                        (returns["InvoiceDate"] <= r["date"] + pd.Timedelta(days=30))]
        reversed_ = (-later["Amount"] >= 0.9 * r["amount"]).any() and r["amount"] >= 100
        if reversed_:
            cats.append("30 天内被同一客户冲销")
        elif r["n_items"] <= 2 and r["qty"] >= 1000:
            cats.append("单品批量 (1–2 种商品，≥1000 件)")
        elif r["qty"] <= 2:
            cats.append("极小单 (≤2 件)")
        elif r["n_items"] <= 2:
            cats.append("其他 1–2 种商品的单子")
        else:
            cats.append("多品类的单子")
    return cats


def inject(buy, factor, seed):
    """随机挑 1% 的发票，每张随机一行的数量乘以 factor。返回 (新的交易行, 被注入的发票号)。"""
    rng = np.random.default_rng(seed)
    invoices = buy["InvoiceNo"].unique()
    picked = rng.choice(invoices, size=int(INJECT_RATE * len(invoices)), replace=False)
    out = buy.copy()
    rows = out[out["InvoiceNo"].isin(picked)].groupby("InvoiceNo").sample(1, random_state=seed).index
    out.loc[rows, "Quantity"] *= factor
    out["Amount"] = out["Quantity"] * out["UnitPrice"]
    return out, set(picked)


def main():
    df = load_clean()
    rfm = compute_rfm(df)
    _, cluster = fit_segments(rfm)
    buy = df[~df["IsReturn"] & df["CustomerID"].isin(rfm.index)]
    f = invoice_features(buy)
    print(f"{len(f)} 张购买发票；下单时间 {f['date'].dt.hour.min()} 点到 {f['date'].dt.hour.max()} 点\n")

    # (a) 真实数据上分数最高的 20 张发票
    raw = load_raw()
    returns = raw[raw["InvoiceNo"].str.startswith("C") & raw["CustomerID"].notna()].drop_duplicates()
    returns = returns.assign(Amount=returns["Quantity"] * returns["UnitPrice"])
    scores = anomaly_scores(f)
    inspect = {}
    for name in ["孤立森林", "LOF (n_neighbors=50)"]:
        top = f.index[np.argsort(-scores[name])[:TOP_N]]
        cats = categorize(f, top, returns)
        view = f.loc[top, ["n_items", "qty", "amount", "max_line_qty"]].assign(
            cluster=f.loc[top, "customer"].map(cluster), category=cats)
        print(f"{name} 分数最高的 {TOP_N} 张发票：")
        print(view.round(2).to_string(), "\n")
        counts = pd.Series(cats).value_counts()
        inspect[name] = {"categories": counts.to_dict(),
                         "single_item_share": round(float((f.loc[top, "n_items"] <= 2).mean()), 2),
                         "examples": [{"invoice": inv, **{k: (round(float(v), 2) if k != "cluster" and k != "category"
                                                              else v) for k, v in row.items()}}
                                      for inv, row in view.head(5).iterrows()]}
    big = f.loc["581483"]
    line = buy[buy["InvoiceNo"] == "581483"].iloc[0]
    print(f"最大的一张：发票 581483，{line['Description'].strip()} × {big['qty']}，金额 {big['amount']:.2f}")

    # (b) 注入异常
    inj = []
    for factor in [100, 10]:
        res = {}
        for rep in range(N_REPEATS):
            lines, picked = inject(buy, factor, seed=rep)
            f2 = invoice_features(lines)
            y = f2.index.isin(picked).astype(int)
            k = int(y.sum())
            for name, s in anomaly_scores(f2).items():
                res.setdefault(name, []).append((average_precision_score(y, s), y[np.argsort(-s)[:k]].mean()))
        print(f"\n注入 ×{factor} (每次 {k} 张，占 {y.mean():.3f}；{N_REPEATS} 次平均)：")
        for name, v in res.items():
            ap, rec = np.mean(v, axis=0)
            inj.append({"factor": factor, "method": name, "ap": round(float(ap), 3),
                        "ap_std": round(float(np.std([a for a, _ in v])), 3), "recall_at_k": round(float(rec), 3)})
            print(f"  {name:<24} AP {ap:.3f}  复核前 {k} 张抓到 {rec:.3f}")

    # (c) contamination 只动阈值：在第 0 次 ×100 注入的数据上
    lines, picked = inject(buy, 100, seed=0)
    f2 = invoice_features(lines)
    y = f2.index.isin(picked).astype(int)
    print()
    cont = []
    for c in ["auto", 0.001, 0.01, 0.05]:
        model = make_pipeline(preprocess(), IsolationForest(contamination=c, random_state=SEED)).fit(f2[FEATURES])
        flagged = model.predict(f2[FEATURES]) == -1
        s = -model.score_samples(f2[FEATURES])
        cont.append({"contamination": c, "offset": round(float(model[-1].offset_), 3),
                     "flagged": int(flagged.sum()), "caught": int((flagged & (y == 1)).sum()),
                     "ap": round(float(average_precision_score(y, s)), 3)})
        print(f"contamination={c!s:<5}: offset_ {model[-1].offset_:.3f}，标为异常 {flagged.sum():>5} 张，"
              f"其中注入的 {(flagged & (y == 1)).sum():>3} 张，AP {average_precision_score(y, s):.3f}")

    report("m4", {
        "invoices": len(f),
        "hour_min": int(f["date"].dt.hour.min()),
        "hour_max": int(f["date"].dt.hour.max()),
        "inject_rate": INJECT_RATE,
        "injected_per_repeat": k,
        "repeats": N_REPEATS,
        "inspect_top": TOP_N,
        "inspect": inspect,
        "largest_invoice": {"invoice": "581483", "description": line["Description"].strip(),
                            "qty": int(big["qty"]), "amount": round(float(big["amount"]), 2)},
        "injection": inj,
        "contamination": cont,
    })


if __name__ == "__main__":
    main()
