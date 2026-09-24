# Qlib 主体集成说明

本项目的研究主体是 [Microsoft Qlib](https://github.com/microsoft/qlib)：

- Qlib：数据 Provider、DataHandler、Dataset、因子/Alpha、模型、策略、回测、评估和实验记录。
- 本项目：A 股 PIT 数据、历史行业、交易状态、主板股票池、成本规则、策略晋级门禁和 Web UI 适配。
- Backtrader：仅用于订单级执行回放和交叉核验，不作为研究框架主体。

当前环境检查结果：Qlib 尚未安装，网络代理不可用，因此不能宣称 Qlib 工作流已经执行。安装后使用：

```bash
pip install pyqlib
python -m quant_system.qlib_adapter
```

先把现有长表转换为 Qlib 适配输入：

```bash
python -c "from quant_system.qlib_adapter import materialize_qlib_parquet; print(materialize_qlib_parquet())"
```

研究配置：`quant_system/configs/research/qlib_main_research.json`。

## 融合规则

当 Qlib 与自研结果不一致时，先检查并修正自研适配层，包括：

1. 信号时点和标签时点；
2. 调仓、持仓、成交价和成本口径；
3. 股票池和停牌/涨跌停过滤；
4. PIT 公告日连接；
5. 复权价格与原始执行价格。

只有在 Qlib 主流程和订单级执行回放完成对账后，才允许进入策略筛选和纸面执行。
