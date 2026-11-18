# 数据来源

`online_retail.csv.gz` 是 UCI 机器学习仓库的 Online Retail 数据集 (原文件 `Online Retail.xlsx`)。转换方式：用 pandas 读入 xlsx (InvoiceNo、StockCode 读成字符串，CustomerID 读成可空整数)，原样另存为 gzip 压缩的 CSV，没有删改任何行和列。共 541909 行。

- 引用：Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
- 页面：https://archive.ics.uci.edu/dataset/352/online+retail
- 许可证：Creative Commons Attribution 4.0 International (CC BY 4.0)
- 相关论文：Chen, D., Sain, S. L., Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management*, 19(3).

数据说明 (摘自 UCI 页面)：一家在英国注册、没有实体店的在线零售商在 2010-12-01 到 2011-12-09 之间的全部交易，主要卖各种场合用的礼品，很多客户是批发商。每行是一张发票里的一种商品。

| 列 | 含义 |
| - | - |
| InvoiceNo | 发票号，说明写“6 位整数”；以 C 开头的是取消 (退货)，另有 3 张以 A 开头的坏账调整 |
| StockCode | 商品编码，说明写“5 位整数”；实际有带字母后缀的 (如 85123A)，也有 POST、M、D、BANK CHARGES、AMAZONFEE 这类非商品编码 |
| Description | 商品名称 (有 1454 行缺失，同一编码偶尔有多个名称) |
| Quantity | 这一行的件数，退货为负 |
| InvoiceDate | 开票日期和时间 |
| UnitPrice | 单价，英镑 |
| CustomerID | 客户编号 (5 位整数)，约四分之一的行缺失 |
| Country | 客户所在国家 (38 个，九成以上的行是英国) |

读入时 InvoiceNo、StockCode 必须指定为字符串，CustomerID 用可空整数 (`"Int64"`)，见 `common.load_raw`。清洗规则见 `common.clean`。
