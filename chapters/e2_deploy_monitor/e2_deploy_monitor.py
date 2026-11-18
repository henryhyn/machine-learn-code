"""E2 模型上线与监控：正文里的全部代码，按出现顺序排列。

运行：uv run python chapters/e2_deploy_monitor/e2_deploy_monitor.py
"""

# ---- 代码 1 ----
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_customers(n, seed, spend_scale=1.0, drift=0.0):
    """构造流失数据。spend_scale 让月消费整体变化 (数据漂移)；drift 从 0 到 1，让特征和流失的关系变化 (概念漂移)。"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "在网月数": rng.integers(1, 73, n),
        "月消费": np.round(rng.lognormal(np.log(80), 0.4, n) * spend_scale, 1),   # 单位：元
        "通话分钟": np.round(rng.gamma(2.0, 150, n)),
        "投诉次数": rng.poisson(0.5, n),
        "套餐": rng.choice(["基础", "畅享", "尊享"], n, p=[0.5, 0.35, 0.15]),
    })
    logit = (-2.3 - 0.04 * (df["在网月数"] - 36) + (1.0 - drift) * df["投诉次数"]
             - 0.002 * (df["通话分钟"] - 300) + (0.012 - 0.032 * drift) * (df["月消费"] - 80)
             + df["套餐"].map({"基础": 0.3, "畅享": 0.0, "尊享": -0.5}))
    df["流失"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


features = ["在网月数", "月消费", "通话分钟", "投诉次数", "套餐"]
train, test = make_customers(5000, seed=0), make_customers(5000, seed=1)
X_train, y_train, X_test, y_test = train[features], train["流失"], test[features], test["流失"]

prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), features[:4]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["套餐"]),
])
model = FixedThresholdClassifier(make_pipeline(prep, LogisticRegression()), threshold=0.3)
model.fit(X_train, y_train)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"训练集流失率 {y_train.mean():.3f}，测试集 AUC {auc:.3f}，标为高风险 {model.predict(X_test).mean():.1%}")

# ---- 代码 2 ----
import json
import tempfile
from pathlib import Path

import joblib
import sklearn

meta = {
    "模型": "流失预测 v1",
    "训练数据": "2025-01 至 2025-06 的 5000 个客户",
    "特征": {c: str(X_train[c].dtype) for c in features},
    "sklearn 版本": sklearn.__version__,
    "测试集 AUC": round(auc, 3),
    "阈值": model.threshold,
}
with tempfile.TemporaryDirectory() as d:
    joblib.dump(model, Path(d) / "churn_v1.joblib")
    (Path(d) / "churn_v1.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded = joblib.load(Path(d) / "churn_v1.joblib")
    print((Path(d) / "churn_v1.json").read_text(encoding="utf-8"))
print("加载后的预测和原模型完全一致：", np.array_equal(loaded.predict_proba(X_test), model.predict_proba(X_test)))

# ---- 代码 3 ----
import pickle


class NotAModel:
    def __reduce__(self):    # pickle 用它记录“加载时调用哪个函数来重建对象”
        return (print, ("加载时执行了 print；换成别的函数，就能在加载者的机器上做别的事",))


obj = pickle.loads(pickle.dumps(NotAModel()))
print("pickle.loads 得到的对象：", obj)
with tempfile.TemporaryDirectory() as d:
    joblib.dump(NotAModel(), Path(d) / "model.joblib")
    joblib.load(Path(d) / "model.joblib")

# ---- 代码 4 ----
import warnings

from sklearn.exceptions import InconsistentVersionWarning

old = pickle.dumps(model).replace(b"1.9.1", b"1.8.0")    # 假装这个模型是用 1.8.0 保存的
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    old_model = pickle.loads(old)
ws = [w.message for w in caught if w.category is InconsistentVersionWarning]
print(f"{len(ws)} 条 InconsistentVersionWarning，涉及 {len({w.estimator_name for w in ws})} 种估计器")
print(f"保存时的版本 {ws[0].original_sklearn_version}，当前版本 {ws[0].current_sklearn_version}")
print("照样能预测：", old_model.predict(X_test[:5]))

# ---- 代码 5 ----
def try_predict(name, X):
    try:
        p = model.predict_proba(X)[:, 1]
    except ValueError as e:
        print(f"{name}：报错 {str(e).splitlines()[0]}")
        return
    print(f"{name}：平均概率 {p.mean():.3f}，高风险 {model.predict(X).mean():.1%}，AUC {roc_auc_score(y_test, p):.3f}")


try_predict("原样", X_test)
try_predict("列顺序打乱、多一列", X_test[features[::-1]].assign(用户ID=1))
try_predict("列名改了", X_test.rename(columns={"月消费": "月消费_元"}))
try_predict("月消费传成字符串", X_test.assign(月消费=X_test["月消费"].astype(str)))
try_predict("月消费全部缺失", X_test.assign(月消费=np.nan))
try_predict("全是新套餐", X_test.assign(套餐="校园"))
try_predict("月消费单位变成分", X_test.assign(月消费=X_test["月消费"] * 100))

# ---- 代码 6 ----
def make_schema(X):
    """从训练数据记下每列的类型、范围 (或取值集合) 和缺失率。"""
    schema = {}
    for c in X.columns:
        rule = {"缺失率": X[c].isna().mean()}
        if pd.api.types.is_numeric_dtype(X[c]):
            rule.update(类型="数值", 范围=(X[c].min(), X[c].max()))
        else:
            rule.update(类型="类别", 取值=set(X[c].dropna()))
        schema[c] = rule
    return schema


def validate(X, schema, max_bad=0.01, missing_tol=0.05):
    problems = []
    for c, rule in schema.items():
        if c not in X.columns:
            problems.append(f"{c}：缺少这一列")
            continue
        if X[c].isna().mean() > rule["缺失率"] + missing_tol:
            problems.append(f"{c}：缺失率 {X[c].isna().mean():.0%}，训练时 {rule['缺失率']:.0%}")
        col = X[c].dropna()
        if rule["类型"] == "数值":
            if not pd.api.types.is_numeric_dtype(col):
                problems.append(f"{c}：应为数值，收到 {col.dtype}")
                continue
            lo, hi = rule["范围"]
            bad = ((col < lo) | (col > hi)).mean()
            if bad > max_bad:
                problems.append(f"{c}：{bad:.0%} 的值超出训练时的范围 [{lo}, {hi}]")
        else:
            bad = (~col.isin(rule["取值"])).mean()
            if bad > max_bad:
                problems.append(f"{c}：{bad:.0%} 是训练时没见过的取值")
    return problems


schema = make_schema(X_train)
print("正常的一批：", validate(X_test, schema))
first = np.arange(len(X_test)) < 1000
bad_batch = X_test.assign(月消费=X_test["月消费"] * 100,
                          通话分钟=X_test["通话分钟"].astype(str),
                          投诉次数=X_test["投诉次数"].where(~first),
                          套餐=X_test["套餐"].where(~first, "校园"))
print("有问题的一批：")
for p in validate(bad_batch, schema):
    print(" -", p)

# ---- 代码 7 ----
scenarios = {
    "无漂移": make_customers(5000, seed=2),
    "数据漂移": make_customers(5000, seed=3, spend_scale=1.3),   # 涨价：月消费整体高 30%，关系不变
    "概念漂移": make_customers(5000, seed=4, drift=1.0),         # 分布不变，关系变了
}
rows = {}
for name, df in scenarios.items():
    p = model.predict_proba(df[features])[:, 1]
    rows[name] = {"月消费均值": df["月消费"].mean(), "平均预测概率": p.mean(),
                  "标为高风险": model.predict(df[features]).mean(),
                  "实际流失率": df["流失"].mean(), "AUC": roc_auc_score(df["流失"], p)}
print(pd.DataFrame(rows).T.round(3))

# ---- 代码 8 ----
def psi(expected, actual, bins=10):
    """群体稳定性指数。数值特征按 expected (训练数据) 的分位数分箱，类别特征按取值分组。"""
    expected, actual = pd.Series(expected).dropna(), pd.Series(actual).dropna()
    if pd.api.types.is_numeric_dtype(expected):
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    else:
        cats = sorted(set(expected) | set(actual))
        e = expected.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        a = actual.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)    # 空箱给一个很小的比例，避免 log(0)
    return float(np.sum((a - e) * np.log(a / e)))


train_score = model.predict_proba(X_train)[:, 1]
table = {}
for name, df in scenarios.items():
    table[name] = {c: psi(X_train[c], df[c]) for c in features}
    table[name]["预测概率"] = psi(train_score, model.predict_proba(df[features])[:, 1])
print(pd.DataFrame(table).round(3))

# ---- 代码 9 ----
from scipy.stats import ks_2samp

last = make_customers(200_000, seed=10)
this = make_customers(200_000, seed=11, spend_scale=1.01)    # 月消费只涨了 1%
ks = ks_2samp(last["月消费"], this["月消费"])
print(f"月消费均值 {last['月消费'].mean():.1f} → {this['月消费'].mean():.1f}")
print(f"20 万样本：KS 统计量 {ks.statistic:.4f}，p 值 {ks.pvalue:.1e}，PSI {psi(last['月消费'], this['月消费']):.4f}")
print(f"2000 样本：p 值 {ks_2samp(last['月消费'][:2000], this['月消费'][:2000]).pvalue:.2f}")
for name, df in [("上月", last), ("本月", this)]:
    print(f"{name} AUC {roc_auc_score(df['流失'], model.predict_proba(df[features])[:, 1]):.3f}")

# ---- 代码 10 ----
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder

adv = make_pipeline(
    ColumnTransformer([("cat", OrdinalEncoder(), ["套餐"])], remainder="passthrough"),
    RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=0, n_jobs=-1))
for name, df in scenarios.items():
    X_adv = pd.concat([X_train, df[features]], ignore_index=True)
    y_adv = np.r_[np.zeros(len(X_train)), np.ones(len(df))]      # 0 = 训练数据，1 = 新数据
    auc_adv = cross_val_score(adv, X_adv, y_adv, cv=5, scoring="roc_auc").mean()
    print(f"{name}：区分新旧数据的 AUC {auc_adv:.3f}")

# ---- 代码 11 ----
from sklearn.base import clone

months = [make_customers(2000, seed=100 + m, drift=m / 23) for m in range(24)]   # 关系在 24 个月里慢慢变


def fit_on(ms):
    d = pd.concat([months[m] for m in ms])
    return clone(model).fit(d[features], d["流失"])


never = fit_on(range(0, 4))          # 第 6 个月上线，标签延迟 3 个月，只有第 0–3 个月的标签
rows = {}
for t in range(6, 24):               # 第 t 个月打分时，有标签的是第 0 到 t-3 个月
    candidates = {"从不重训": never,
                  "每月重训 全部历史": fit_on(range(0, t - 2)),
                  "每月重训 最近6个月": fit_on(range(max(0, t - 8), t - 2))}
    rows[t] = {k: roc_auc_score(months[t]["流失"], m.predict_proba(months[t][features])[:, 1])
               for k, m in candidates.items()}
res = pd.DataFrame(rows).T
print(res.iloc[::3].round(3))
print("平均", res.mean().round(3).to_dict())
