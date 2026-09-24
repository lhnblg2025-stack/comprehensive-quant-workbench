# 腾讯云真实验收清单

在真实 CVM 上执行前，先确认以下外部条件已经就绪：

- CVM：Ubuntu 22.04/Debian 12，4 vCPU / 8 GB 以上
- Python 3.10 可用，且可以创建 `/opt/quant/venv`
- 域名解析到 CVM，或使用公网 IP 但仅限内部验收
- 腾讯云安全组只开放 22、80、443；不要开放 8600
- Nginx 证书：腾讯云 SSL 或 Let's Encrypt
- SSH 私钥和 `known_hosts` 已正确配置

## 首装

```bash
sudo bash deploy/install_tencent_cloud.sh /path/to/checkout
sudoedit /etc/quant/quant.env
sudo systemctl restart quant-web.service
```

生产密钥必须填真实随机值，不能保留 `replace-with-a-long-random-secret`。

## 即时验收

```bash
bash deploy/verify_unattended.sh
curl -fsS http://127.0.0.1:8600/api/version
systemctl list-timers quant-after-close.timer quant-data-update.timer
```

## 故障演练

- 重启服务：

  ```bash
  sudo systemctl restart quant-web.service
  bash deploy/verify_unattended.sh
  ```

- 杀掉服务进程，观察 systemd 自动拉起：

  ```bash
  sudo systemctl kill -s KILL quant-web.service
  sleep 15
  systemctl is-active quant-web.service
  ```

- 发布失败回滚：用一个损坏 archive 运行 `deploy/remote_release_linux.sh`，确认 `current` 仍指向旧 release。

- 断网恢复：在非交易时段断开外网 10 分钟再恢复，观察 `freshness_gate` 和运行状态是否降级而不是崩溃。

## 7-14 天观测

每天检查：

```bash
bash deploy/verify_unattended.sh
systemctl --failed
systemctl status quant-web.service --no-pager
ls -l /opt/quant/generated/runtime_status 2>/dev/null
```

记录：

- Web 常驻是否发生重启
- 盘后任务是否每天在 15:35 后完成
- 数据更新 timer 是否正常
- 数据源是否出现 stale/missing
- 报告是否每天生成并通过 `artifact_guard`
- 磁盘、内存和 CPU 是否稳定
- 是否有异常外部告警或错误推送

只有连续通过上述观测后，才应把常驻稳定性评级从代码级 `B+` 提升为环境验证后的 `A-`。
