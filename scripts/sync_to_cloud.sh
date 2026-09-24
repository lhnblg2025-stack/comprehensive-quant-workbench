#!/usr/bin/env bash
# Local-to-cloud code publication. No data or secrets are included.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLOUD_KIND="${CLOUD_KIND:-tencent}"
STAMP="${QUANT_RELEASE_ID:-$(date +%Y%m%d%H%M%S)}"
WORK_DIR="${QUANT_SYNC_WORK_DIR:-/tmp/cloud_migrate}"
LOG_DIR="${QUANT_SYNC_LOG_DIR:-/tmp/quant_logs}"
KNOWN_HOSTS="${QUANT_KNOWN_HOSTS:-$HOME/.ssh/known_hosts}"
DRY_RUN="${QUANT_SYNC_DRY_RUN:-0}"
ARCHIVE="$WORK_DIR/quant_code_${STAMP}.tar.gz"
VERSION_FILE="$WORK_DIR/VERSION"
mkdir -p "$WORK_DIR" "$LOG_DIR"
LOG="$LOG_DIR/sync_${CLOUD_KIND}_$(date +%Y%m%d).log"

for tool in tar ssh scp git; do
  command -v "$tool" >/dev/null || { echo "[sync][FATAL] missing $tool" | tee -a "$LOG"; exit 1; }
done
[[ -f "$KNOWN_HOSTS" ]] || { echo "[sync][FATAL] known_hosts not found: $KNOWN_HOSTS" | tee -a "$LOG"; exit 2; }

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$KNOWN_HOSTS" -o ConnectTimeout=15)
WS_HASH="$(git -C "$ROOT" rev-parse --short HEAD)"
QS_HASH="$(git -C "$ROOT/quant_system" rev-parse --short HEAD)"
printf 'workspace=%s quant_system=%s release=%s\n' "$WS_HASH" "$QS_HASH" "$STAMP" > "$VERSION_FILE"

cd "$ROOT"
STAGE="$(mktemp -d "$WORK_DIR/package.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
tar \
  --exclude='.git' --exclude='*/.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='data_warehouse' --exclude='generated' --exclude='logs' --exclude='tmp' \
  --exclude='*.parquet' --exclude='*.log' --exclude='*.sqlite' --exclude='*.sqlite3' \
  --exclude='.env' --exclude='.env.*' --exclude='*.pem' --exclude='*.key' \
  --exclude='credentials' --exclude='config/private_data_sources.json' \
  --exclude='config/external_api_keys.json' --exclude='config/design_proxy_keys.json' \
  --exclude='config/xueqiu_cookie.json' \
  -cf - quant_system quant_web quant_platform scripts config deploy requirements.lock.txt start_quant.sh \
  | tar -C "$STAGE" -xf -
cp "$VERSION_FILE" "$STAGE/VERSION"
tar -C "$STAGE" -czf "$ARCHIVE" .

if tar -tzf "$ARCHIVE" | grep -E '(^|/)(\.env($|\.)|[^/]*\.(pem|key)$|credentials/|private_data_sources\.json$|external_api_keys\.json$|design_proxy_keys\.json$|xueqiu_cookie\.json$)' >/dev/null; then
  echo "[sync][FATAL] archive contains forbidden secret paths" | tee -a "$LOG"
  exit 3
fi
[[ "$DRY_RUN" == 1 ]] && { echo "DRY_RUN_OK archive=$ARCHIVE release=$STAMP" | tee -a "$LOG"; exit 0; }

case "$CLOUD_KIND" in
  vultr|linux)
    HOST="${QUANT_CLOUD_HOST:-${VULTR_HOST:-}}"
    KEY="${QUANT_CLOUD_PEM:-${VULTR_KEY:-$HOME/.ssh/id_ed25519}}"
    APP_HOME="${QUANT_REMOTE_APP_HOME:-/opt/quant}"
    SERVICE="${QUANT_REMOTE_SERVICE:-quant-web.service}"
    [[ -n "$HOST" ]] || { echo "[sync][FATAL] QUANT_CLOUD_HOST required"; exit 2; }
    REMOTE_STAGE="/tmp/quant-release-${STAMP}"
    scp -i "$KEY" "${SSH_OPTS[@]}" "$ARCHIVE" "$ROOT/deploy/remote_release_linux.sh" "$HOST:/tmp/"
    OUT="$(ssh -i "$KEY" "${SSH_OPTS[@]}" "$HOST" \
      "set -e; trap 'rm -rf '\''$REMOTE_STAGE'\''' EXIT; mkdir -p '$REMOTE_STAGE'; mv '/tmp/$(basename "$ARCHIVE")' '$REMOTE_STAGE/release.tar.gz'; mv /tmp/remote_release_linux.sh '$REMOTE_STAGE/deploy.sh'; chmod 700 '$REMOTE_STAGE/deploy.sh'; QUANT_RELEASE_HOME='$APP_HOME' QUANT_RELEASE_SERVICE='$SERVICE' '$REMOTE_STAGE/deploy.sh' '$REMOTE_STAGE/release.tar.gz' '$STAMP'")"
    grep -q "DEPLOY_OK release=$STAMP" <<<"$OUT" || { echo "[sync][FATAL] missing deploy acknowledgement"; exit 1; }
    ;;
  tencent)
    echo "[sync][FATAL] Windows direct deployment is retired; use a Linux Tencent CVM or run install_tencent_cloud.sh on the host." >&2
    exit 4
    ;;
  *) echo "[sync][FATAL] unsupported CLOUD_KIND=$CLOUD_KIND" >&2; exit 2 ;;
esac
printf '[sync] completed release=%s workspace=%s quant_system=%s\n' "$STAMP" "$WS_HASH" "$QS_HASH" | tee -a "$LOG"
