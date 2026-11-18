# 数据来源

`customer_churn.csv` 是 UCI 机器学习仓库的 Iranian Churn 数据集原文件 (`Customer Churn.csv`，未做修改，只改了文件名)。

- 引用：Iranian Churn [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5JW3Z
- 页面：https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
- 许可证：Creative Commons Attribution 4.0 International (CC BY 4.0)

数据说明 (摘自 UCI 页面)：一家伊朗电信公司 12 个月内随机抽取的 3150 个客户，每行一个客户。除 Churn 外的所有列都是前 9 个月的汇总，Churn 是第 12 个月末的状态，中间 3 个月是预留的计划期。

| 列 | 含义 |
| - | - |
| Call Failure | 通话失败次数 |
| Complains | 是否投诉 (0 否，1 是) |
| Subscription Length | 订阅总月数 |
| Charge Amount | 消费等级 (有序，说明写 0–9，实际取值 0–10) |
| Seconds of Use | 通话总秒数 |
| Frequency of use | 通话总次数 |
| Frequency of SMS | 短信总条数 |
| Distinct Called Numbers | 拨打过的不同号码数 |
| Age Group | 年龄段 (有序，1 最年轻，5 最年长) |
| Tariff Plan | 资费类型 (1 按量付费，2 合约) |
| Status | 状态 (1 活跃，2 不活跃) |
| Age | 年龄 (只有 15/25/30/45/55 五个值，与 Age Group 一一对应) |
| Customer Value | 客户价值 (数据方计算) |
| Churn | 是否流失 (1 流失，0 未流失)，预测目标 |
