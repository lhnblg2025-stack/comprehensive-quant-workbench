# 量化策略开发工作流

## 目标

策略开发和数据补录分离。新数据只进入 `data_warehouse/staging/`，研究运行器读取冻结输入并输出不可变 run bundle；任何缺少 PIT 股票池、交易状态或公告可用时间的结果只能是 `RESEARCH_ONLY`。

## 一次完整流程

1. **登记数据缺口**
   ```bash
   PYTHONPATH=. python3 scripts/build_public_data_gap_catalog.py
   ```
   先看 `data_warehouse/staging/public_data_gap_catalog_20260923/manifest.json` 的 `admission_blockers`。

2. **采集公开数据**

   为一个数据域准备 JSON spec，只填写官方 URL、参数和预期格式：
   ```json
   [
     {
       "key": "example_calendar",
       "url": "https://www.sse.com.cn/...",
       "data_domain": "exchange_calendar",
       "params": {},
       "expected_format": "json"
     }
   ]
   ```
   使用：
   ```bash
   PYTHONPATH=. python3 scripts/collect_public_data_staging.py \
     --spec specs/public_data.json \
     --output-root data_warehouse/staging/public_data_collect_20260923
   ```
   原始响应、规范化结果、哈希、日期范围和失败原因都保存在同一 manifest。采集器不会写入主面板。

3. **开发信号**

   价格/成交量信号应加入 `quant_system/public_strategies.py`，每个信号必须：
   - 只使用当日及更早字段；
   - 通过 `feature_asof` 因果检查；
   - 注册到 `EXPANDED_STRATEGIES`；
   - 指定所属 family，便于门禁和分层诊断。

4. **运行研究诊断**
   ```bash
   PYTHONPATH=. python3 scripts/run_strategy_research.py \
     --strategies mom_20 \
     --horizons 1,5,21 \
     --start 2018-01-01 \
     --end 2026-09-18
   ```
   输出包含信号、未来收益标签、Rank IC、分层收益、换手代理、市场 regime 分层和输入哈希。

5. **晋级条件**

   只有在逐日 PIT 股票池、停牌/ST/涨跌停、公司行为、公告可用时间和执行回放均通过独立审计后，才允许新增正式回测入口。当前门禁固定为：
   `RESEARCH_ONLY=true`、`formal_backtest_admitted=false`、`survivorship_bias_risk=true`。

## 不能做的事情

- 不能把报告期日期当公告可用日期；
- 不能用当前成分快照重建历史成分；
- 不能把上市前日期填成空值或前值；
- 不能把免费源缺失或连接失败改写成完整覆盖；
- 不能触发真实下单。
