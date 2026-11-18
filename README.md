# machine-learn-code

《讲透经典机器学习》的配套代码。

- Python 3.12，scikit-learn 1.9，依赖见 `pyproject.toml`，用 [uv](https://docs.astral.sh/uv/) 管理。
- `chapters/`：每章一个目录，内容和正文里的代码块一致，可以直接运行。
- `projects/p1_churn/`：配套项目 P1 客户流失预测，说明和结果见 [projects/p1_churn/README.md](projects/p1_churn/README.md)。
- `projects/p2_retail/`：配套项目 P2 用户分群与购物篮分析，说明和结果见 [projects/p2_retail/README.md](projects/p2_retail/README.md)。

```bash
uv sync                      # 国内网络慢时：UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple uv sync
uv run python chapters/c5_decision_tree/c5_decision_tree.py
```

章节代码的演示数据都是 sklearn 自带、可以离线加载的数据集或代码生成的数据；两个项目用的 UCI 数据集 (CC BY 4.0) 已放在各自的 `data/` 目录里，不需要联网。随机过程都固定了 `random_state`，输出应与正文逐字一致。
