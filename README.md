# 综合量化工作台 · 公开核心

从量化工作台拆分的可独立运行研究核心，用于展示回测基础设施、数据校验和策略扩展接口。**这是经过筛选的公开核心版，并非原工作台全部源码或全部功能。**

仓库：https://github.com/lhnblg2025-stack/comprehensive-quant-workbench

## 能力与边界

- 保留通用回测引擎：交易费用、滑点、次日撮合、组合与因子分组回测接口。
- 保留统一 Sharpe 指标、DataFrame 校验、请求节流和 CPU 节流工具。
- 私有策略由公开均线模板替代，参数仅为演示假设，不是生产参数。
- 提供纯合成价格和回环地址 Web 演示，无真实账户、无自动下单、无数据下载。
- 不包含私有 Alpha、模型、持仓、历史研究结果、未授权报告、外部数据或原 Git 历史。

仅供研究，不构成投资建议。合成回测结果不是历史业绩或收益承诺。

## 安装与运行

Python 3.10 或更高版本，建议使用独立环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash start_quant.sh
```

打开 http://127.0.0.1:8600 ，点击运行离线模板回测。依赖安装需要联网或本地 wheel 缓存；安装后演示不需要联网。
使用 `QUANT_PYTHON=.venv/bin/python bash start_quant.sh` 显式选择解释器；`QUANT_WEB_PORT=8601` 可改变端口。按 Ctrl+C 停止服务。

```bash
python -m unittest discover -s tests -v
python scripts/check_release.py
```

## 模块

| 目录 | 内容 |
|---|---|
| quant_system | 原通用回测与校验工具、公开配置和信号模板 |
| quant_platform | 合成数据与离线回测门面 |
| quant_web | 新增公开演示服务，不加载原工作台私有后台 |
| tests | 离线运行、指标与信号时序测试 |
| scripts | 发布文件静态检查 |
| docs | 发布范围和限制 |
| reports | 报告授权规则；未包含真实研究报告 |

修改 `quant_system/signals.py` 可实现自己的策略：输出 signal = 1、0、-1，并保留引擎所需的 ma_slow、ma_trend 列。只能使用当前及此前数据；生产策略请在私有仓库管理。

## 限制

原完整工作台的 Web 页面、行情接口、私有预测模块和运维集成未发布。回测引擎的市场规则及公司行动可选模块未纳入，本版会走引擎原有回退路径；演示不验证完整交易所规则、公司行动、真实市场成交或实盘适用性。仅测试合成演示路径，其余保留接口不承诺全部测试通过。

原工作台前端源码现已加入 `quant_web/static/`（内嵌知识资料已清空，第三方图表压缩包暂不分发）。还包括过拟合检验、任务 DAG、跨进程锁与快照归档工具。前端对应的完整后台尚未迁移，这些页面不能通过公开演示入口实现全部功能。

完整范围见 [发布说明](docs/RELEASE_SCOPE.md)。

## 授权

发布代码采用 [MIT](LICENSE)。仅涵盖本仓库有权授权的代码，不重新授权第三方依赖。NumPy、pandas 是独立安装的第三方包，遵循各自 BSD 许可。研究报告如独立发布采用 CC BY 4.0，见 [报告说明](reports/README.md)。
