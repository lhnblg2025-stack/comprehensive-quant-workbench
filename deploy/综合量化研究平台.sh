#!/usr/bin/env bash
# 本地量化平台启动器：启动唯一 Web 服务并打开运营控制台。
set -u
exec ${PROJECT_ROOT}/deploy/start_quant_console.sh
