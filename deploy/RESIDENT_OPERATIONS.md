# 常驻量化节点操作手册

该模式让常开云主机承担数据更新、盘后报告、数据发布、纸面账本、健康监控和飞书汇报。本机关闭后不影响这些任务；本机只用于开发、查看控制台和手工成交回填。

## 一次安装

在常开 Ubuntu/Debian 节点执行：

```bash
sudo bash deploy/install_tencent_cloud.sh /path/to/checkout
sudoedit /etc/quant/quant.env
sudo systemctl restart quant-web.service
bash deploy/verify_unattended.sh
```

`/etc/quant/quant.env` 必填：

```text
QUANT_WEB_API_KEY=<随机长密钥>
FEISHU_APP_ID=<飞书应用ID>
FEISHU_APP_SECRET=<飞书应用密钥>
FEISHU_CHAT_ID=<接收群或用户ID>
```

安装后会启用：

- `quant-after-close.timer`：交易日盘后主链
- `quant-data-update.timer`：日频数据更新
- `quant-resident-recovery.timer`：主机重启后检查并补跑最近完成交易日
- `quant-resident-digest.timer`：交易日19:10飞书运行摘要

## 日常只看两处

1. 飞书：19:10固定收到数据发布、任务状态、健康告警和纸面执行摘要。严重失败会在重试耗尽后单独告警，重复故障自动去重。
2. 控制台：`https://<你的域名>/ops_console.html`。首屏显示 `release_id`、警告、最近盘后链、纸面成交率、数据状态和需要处理事项。

无需查看服务器日志，也无需手工启动日常任务。

## 纸面成交回填

每日把手工或券商成交导出为两张CSV：

```text
orders.csv
symbol,direction,shares,suggested_price,order_type,notes

fills.csv
order_id,filled_shares,filled_price,commission,stamp_tax,filled_at
```

在节点执行：

```bash
python scripts/import_paper_execution.py \
  --release-id <控制台显示的release_id> \
  --orders orders.csv
python scripts/import_paper_execution.py \
  --release-id <控制台显示的release_id> \
  --fills fills.csv
```

系统会累计部分成交、成交率、未成交和建议价偏差。每周或每月导出实际持仓CSV：

```text
symbol,shares
600519,100
```

并对账：

```bash
python scripts/paper_position_reconciliation.py \
  --actual actual_positions.csv \
  --output generated/paper_reconciliation
```

退出码为0代表持仓一致；2代表有差异，需在控制台或飞书中处理。

## 验收与故障处理

```bash
bash deploy/verify_unattended.sh
systemctl --failed
systemctl status quant-after-close.service --no-pager
```

- 单次数据源失败：release 写入 warning；海外评分自动不参与，不会伪造信号。
- 核心数据、报告或发布门禁失败：盘后链自动重试两次；仍失败则飞书告警并保留最后成功 release。
- 主机重启：恢复 timer 会按真实交易日历补跑最近完成交易日。
- release 清理：仅可执行 `python -m quant_system.data_release --prune 90`；被复盘引用的 release 不会删除。

## 运行边界

该系统可无人值守生成研究报告和维护纸面账本，但不自动向券商下单。连续三个月纸面运行记录是评估执行偏差和是否进入真实实盘的必要证据。
