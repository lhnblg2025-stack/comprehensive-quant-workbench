#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP="${XDG_DESKTOP_DIR:-$HOME/Desktop}"
mkdir -p "$DESKTOP"
cat > "$DESKTOP/综合量化研究平台.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=综合量化研究平台
Comment=启动本地量化运营控制台
Path=$ROOT
Exec=/bin/bash $ROOT/start_quant.sh
Terminal=true
Icon=utilities-system-monitor
Categories=Finance;Science;
StartupNotify=true
EOF
chmod +x "$DESKTOP/综合量化研究平台.desktop"
printf 'launcher=%s\n' "$DESKTOP/综合量化研究平台.desktop"
