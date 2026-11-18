"""m1 RFM 特征：清洗交易数据，按客户算 Recency、Frequency、Monetary，再用传统的 RFM 打分分群当基线。

- 清洗规则见 common.clean (每一步打印删掉多少行)；
- RFM 的口径见 common.compute_rfm (参照日 = 数据最后一天 +1，Frequency 数购买发票，Monetary 用净额)；
- 三列都强右偏，后面 K-Means 要先 log1p 再标准化 (书中 B2)；
- 基线：每列按五分位打 1–5 分 (R 越近分越高)，再按 R 分和 F、M 平均分的高低分成四群。

运行：uv run python projects/p2_retail/m1_rfm.py
"""
import numpy as np
import pandas as pd

from common import REFERENCE_DATE, clean, compute_rfm, load_raw, report

SEGMENT_ORDER = ["高价值活跃", "高价值渐冷", "低价值新近", "低价值沉睡"]


def rfm_scores(rfm):
    """五分位打分。Frequency 有大量并列 (三分之一的客户只买过一次)，先按排名打破并列再切分。"""
    def quintile(s, ascending=True):
        labels = [1, 2, 3, 4, 5] if ascending else [5, 4, 3, 2, 1]
        return pd.qcut(s.rank(method="first"), 5, labels=labels).astype(int)
    return pd.DataFrame({
        "R": quintile(rfm["Recency"], ascending=False),   # 越近分越高
        "F": quintile(rfm["Frequency"]),
        "M": quintile(rfm["Monetary"]),
    })


def rfm_segments(scores):
    """经典的 2×2 分法：R 分 >= 4 算“新近”，F、M 平均分 >= 3 算“高价值”。"""
    recent = scores["R"] >= 4
    valuable = (scores["F"] + scores["M"]) / 2 >= 3
    seg = np.select([recent & valuable, ~recent & valuable, recent & ~valuable],
                    SEGMENT_ORDER[:3], default=SEGMENT_ORDER[3])
    return pd.Series(seg, index=scores.index, name="segment")


def profile(rfm, labels, order=None):
    """每群的人数、R/F/M 中位数、占总净消费额的比例。"""
    g = rfm.groupby(labels)
    table = pd.DataFrame({
        "customers": g.size(),
        "recency_median": g["Recency"].median(),
        "frequency_median": g["Frequency"].median(),
        "monetary_median": g["Monetary"].median().round(0),
        "monetary_share": (g["Monetary"].sum() / rfm["Monetary"].sum()).round(3),
    })
    return table.loc[order] if order is not None else table


def main():
    raw = load_raw()
    is_c = raw["InvoiceNo"].str.startswith("C")
    facts = {
        "rows": len(raw),
        "date_min": str(raw["InvoiceDate"].min()),
        "date_max": str(raw["InvoiceDate"].max()),
        "missing_customer_share": round(float(raw["CustomerID"].isna().mean()), 3),
        "duplicated_rows": int(raw.duplicated().sum()),
        "return_rows": int(is_c.sum()),
        "return_rows_all_negative_qty": bool((raw.loc[is_c, "Quantity"] < 0).all()),
        "negative_qty_not_return": int(((raw["Quantity"] < 0) & ~is_c).sum()),
        "zero_price_rows": int((raw["UnitPrice"] == 0).sum()),
        "negative_price_rows": int((raw["UnitPrice"] < 0).sum()),
        "uk_row_share": round(float((raw["Country"] == "United Kingdom").mean()), 3),
        "countries": int(raw["Country"].nunique()),
        "saturday_rows": int((raw["InvoiceDate"].dt.day_of_week == 5).sum()),
        "hour_min": int(raw["InvoiceDate"].dt.hour.min()),
        "hour_max": int(raw["InvoiceDate"].dt.hour.max()),
    }
    print("原始数据：")
    for k, v in facts.items():
        print(f"  {k}: {v}")
    print()

    df, steps = clean(raw)
    rfm = compute_rfm(df)
    n_cust = df["CustomerID"].nunique()
    print(f"参照日 {REFERENCE_DATE.date()}；有交易的客户 {n_cust} 个，去掉净消费额 <= 0 的 "
          f"{n_cust - len(rfm)} 个，剩 {len(rfm)} 个\n")

    desc = rfm.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["skew"] = rfm.skew()
    desc["skew_log1p"] = np.log1p(rfm).skew()
    print(desc.round(2).to_string(), "\n")
    one_time = float((rfm["Frequency"] == 1).mean())
    top10 = rfm["Monetary"].nlargest(int(len(rfm) * 0.1)).sum() / rfm["Monetary"].sum()
    print(f"只买过一次的客户占 {one_time:.3f}；消费额前 10% 的客户贡献了 {top10:.3f} 的净消费额\n")

    scores = rfm_scores(rfm)
    seg = rfm_segments(scores)
    seg_table = profile(rfm, seg, SEGMENT_ORDER)
    print("RFM 打分分群 (基线)：")
    print(seg_table.to_string())

    report("m1", {
        "raw": facts,
        "cleaning": [{"step": name, "rows_removed": n} for name, n in steps],
        "clean_rows": len(df),
        "clean_return_rows": int(df["IsReturn"].sum()),
        "clean_customers": n_cust,
        "clean_products": int(df["StockCode"].nunique()),
        "purchase_invoices": int(df.loc[~df["IsReturn"], "InvoiceNo"].nunique()),
        "return_invoices": int(df.loc[df["IsReturn"], "InvoiceNo"].nunique()),
        "rfm_customers": len(rfm),
        "dropped_nonpositive_monetary": n_cust - len(rfm),
        "one_time_share": round(one_time, 3),
        "top10pct_monetary_share": round(float(top10), 3),
        "rfm_describe": [
            {"feature": f, "mean": round(float(r["mean"]), 1), "median": round(float(r["50%"]), 1),
             "max": round(float(r["max"]), 1), "skew": round(float(r["skew"]), 2),
             "skew_log1p": round(float(r["skew_log1p"]), 2)}
            for f, r in desc.iterrows()],
        "segments": [{"segment": s, **r} for s, r in seg_table.to_dict("index").items()],
    })


if __name__ == "__main__":
    main()
