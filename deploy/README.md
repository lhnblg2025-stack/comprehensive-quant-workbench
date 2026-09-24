# 云端常驻部署

本项目支持同一份代码在本机和腾讯云 CVM 运行。云端建议使用 Ubuntu/Debian、systemd、Python 3.10+、至少 4 vCPU / 8 GB RAM，并将 Web 后端仅绑定到 `127.0.0.1`，由 Nginx 提供 HTTPS。

## 腾讯云一键安装

将仓库上传到 CVM 后，以 root 执行：

```bash
sudo bash deploy/install_tencent_cloud.sh /path/to/checkout
sudoedit /etc/quant/quant.env
sudo systemctl restart quant-web.service
```

安装脚本会创建 `quant` 系统用户、`/opt/quant/venv`、`/opt/quant/current` 代码软链、`/opt/quant/shared` 持久数据目录、systemd 服务和盘后 timer。真实密钥只写入 `/etc/quant/quant.env`，权限应为 `0600`。公网访问前必须设置随机的 `QUANT_WEB_API_KEY`，并在腾讯云安全组只开放 80/443；不要直接开放 8600。报告写入 `/opt/quant/shared/reports`，升级代码不会删除报告、缓存或数据仓库。

## 服务与验收

```bash
curl http://127.0.0.1:8600/api/livez
curl http://127.0.0.1:8600/api/readyz
curl http://127.0.0.1:8600/api/version
systemctl status quant-web.service
systemctl list-timers quant-after-close.timer
```

盘后 timer 在工作日上海时间 15:35 触发，`Persistent=true` 保证主机短暂离线后补跑；服务和流水线均有单实例保护。数据更新服务已安装但默认不强制立即执行，可按需执行 `systemctl start quant-data-update.service`。

Nginx 反向代理模板见 `nginx-quant.conf.example`，证书可用腾讯云证书服务或 Let's Encrypt。

## 凭据

凭据不得写入仓库：

- `FEISHU_*` 或 `config/report_delivery.json` 的 `secret_loader` 环境变量
- `YOUDAONOTE_API_KEY`
- 雪球/小红书合法 Cookie

缺少可选凭据时，系统应显示对应数据源不可用，而不是伪造成功。

## 本机统一托管

复制 `quant-web-local.service` 到 `~/.config/systemd/user/quant-web.service` 后执行：

```bash
mkdir -p ~/.config/systemd/user
cp deploy/quant-web-local.service ~/.config/systemd/user/quant-web.service
systemctl --user daemon-reload
systemctl --user enable --now quant-web.service
systemctl --user status quant-web.service
```

也可以直接运行：

```bash
bash start_quant.sh
```

验收以 `http://127.0.0.1:8600/api/version` 的 PID、版本和 `ready_phase` 为准；同一机器不要并行运行手工 `server.py` 或第二个 watchdog。
