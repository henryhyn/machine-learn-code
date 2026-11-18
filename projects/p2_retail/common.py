"""P2 用户分群与购物篮分析：各里程碑共用的数据读取、清洗、RFM 计算和结果记录。

数据来自 UCI Online Retail 数据集 (CC BY 4.0)，见 data/README.md。
"""
import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data" / "online_retail.csv.gz"
RESULTS = HERE / "results"
SEED = 42

# 数据覆盖 2010-12-01 到 2011-12-09。RFM 的参照日取“数据最后一天的次日”：
# 最后一天下过单的客户 Recency = 1，而不是 0，含义是“距参照日还有几天”。
REFERENCE_DATE = pd.Timestamp("2011-12-10")


def load_raw():
    """原样读入 541909 行。InvoiceNo、StockCode 必须读成字符串 (有 C536379、85123A 这类值)。"""
    return pd.read_csv(DATA, dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": "Int64"},
                       parse_dates=["InvoiceDate"])


def clean(df, verbose=True):
    """统一的清洗，每一步打印删掉多少行。退货行 (发票号以 C 开头) 保留，由各里程碑自己决定怎么用。

    1. 完全重复的行：同一张发票里同一件商品、同样数量和单价出现两次，按导出重复处理；
    2. 没有 CustomerID 的行：无法归到客户，RFM 和分群都用不上 (约四分之一的行)；
    3. 非商品行：StockCode 不以 5 位数字开头的是邮费 (POST、DOT)、运费 (C2)、手工调整 (M)、
       折扣 (D)、银行手续费、平台佣金、坏账调整等，不是顾客买的商品；
    4. 单价 <= 0 的行：赠品、库存调整和坏账冲销，没有交易金额。
    """
    steps = []

    def step(name, keep):
        nonlocal df
        steps.append((name, int((~keep).sum())))
        df = df[keep]

    n0 = len(df)
    step("完全重复的行", ~df.duplicated())
    step("CustomerID 缺失", df["CustomerID"].notna())
    step("非商品行 (邮费、手工调整、折扣等)", df["StockCode"].str.match(r"^\d{5}"))
    step("单价 <= 0", df["UnitPrice"] > 0)
    df = df.copy()
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["IsReturn"] = df["InvoiceNo"].str.startswith("C")
    df["Amount"] = df["Quantity"] * df["UnitPrice"]
    if verbose:
        print(f"原始 {n0} 行")
        for name, n in steps:
            print(f"  去掉 {name}: {n} 行")
        print(f"清洗后 {len(df)} 行 (其中退货 {int(df['IsReturn'].sum())} 行)，"
              f"客户 {df['CustomerID'].nunique()} 个，商品 {df['StockCode'].nunique()} 种\n")
    return df.reset_index(drop=True), steps


def load_clean(verbose=False):
    return clean(load_raw(), verbose=verbose)[0]


def compute_rfm(df):
    """按客户算 RFM。

    - Recency：参照日 (最后一天 +1) 减去最后一次购买的日期，单位天，越小越近；
    - Frequency：不同的购买发票数 (退货发票不算一次购买)；
    - Monetary：净消费额 = 购买金额 - 退货金额 (退货行的金额本来就是负数，直接相加)。
    净消费额 <= 0 的客户 (只有退货、或退的比买的多，多半是退了数据开始之前买的东西) 去掉。
    """
    buy = df[~df["IsReturn"]]
    last = buy.groupby("CustomerID")["InvoiceDate"].max().dt.normalize()
    rfm = pd.DataFrame({
        "Recency": (REFERENCE_DATE - last).dt.days,
        "Frequency": buy.groupby("CustomerID")["InvoiceNo"].nunique(),
        "Monetary": df.groupby("CustomerID")["Amount"].sum().round(2),
    })
    return rfm[rfm["Monetary"] > 0].dropna().astype({"Recency": int, "Frequency": int})


def report(milestone, result):
    """把一个里程碑的关键数字写进 results/<里程碑>.json。"""
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{milestone}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_to_builtin) + "\n", encoding="utf-8")
    print(f"\n结果已写入 results/{milestone}.json")


def _to_builtin(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    raise TypeError(type(x))
