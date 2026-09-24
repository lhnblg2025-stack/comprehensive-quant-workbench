#!/usr/bin/env bash
# Install the quant stack on a Tencent Cloud CVM running Debian/Ubuntu.
# The code is released atomically under APP_ROOT/current; mutable state lives
# under APP_ROOT/shared so a code release never replaces runtime data.
set -euo pipefail

APP_ROOT="${QUANT_ROOT:-/opt/quant}"
SERVICE_USER="${QUANT_SERVICE_USER:-quant}"
PYTHON_BIN="${QUANT_PYTHON:-$APP_ROOT/venv/bin/python}"
REPO_SOURCE="${1:-$(pwd)}"
RELEASE_ID="bootstrap-$(date +%Y%m%d%H%M%S)"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root: sudo bash deploy/install_tencent_cloud.sh /path/to/checkout" >&2
  exit 2
fi
if [[ ! -f "$REPO_SOURCE/quant_web/server.py" ]]; then
  echo "Checkout not found: $REPO_SOURCE" >&2
  exit 2
fi
REPO_SOURCE="$(cd "$REPO_SOURCE" && pwd)"

for tool in rsync python3 install systemctl flock; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "Required command not found: $tool" >&2
    exit 3
  }
done

RELEASES="$APP_ROOT/releases"
SHARED="$APP_ROOT/shared"
TARGET="$RELEASES/$RELEASE_ID"
STAGE=""
PREVIOUS=""
cleanup() {
  local rc=$?
  trap - EXIT ERR INT TERM
  if [[ -n "$PREVIOUS" && -e "$PREVIOUS" ]]; then
    rollback_link="$APP_ROOT/.current.bootstrap.rollback.$$"
    ln -s "$PREVIOUS" "$rollback_link"
    mv -Tf "$rollback_link" "$APP_ROOT/current" || true
    systemctl restart quant-web.service >/dev/null 2>&1 || true
  elif [[ -L "$APP_ROOT/current" ]]; then
    rm -f "$APP_ROOT/current"
  fi
  [[ -z "$STAGE" ]] || rm -rf "$STAGE"
  [[ ! -d "$TARGET" || "$(readlink -f "$APP_ROOT/current" 2>/dev/null || true)" == "$TARGET" ]] || rm -rf "$TARGET"
  exit "$rc"
}

install -d -m 0755 /etc/quant "$APP_ROOT" "$RELEASES" "$SHARED"
LOCK_FILE="$APP_ROOT/.release.lock"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "another release is being installed" >&2; exit 75; }
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  useradd --system --home-dir "$APP_ROOT" --shell /usr/sbin/nologin "$SERVICE_USER"
fi
STAGE="$(mktemp -d "$RELEASES/.install.XXXXXX")"
trap cleanup EXIT ERR INT TERM

if [[ ! -x "$PYTHON_BIN" ]]; then
  python3 -m venv "$APP_ROOT/venv"
  "$APP_ROOT/venv/bin/pip" install --upgrade pip
  "$APP_ROOT/venv/bin/pip" install -r "$REPO_SOURCE/requirements.lock.txt"
  PYTHON_BIN="$APP_ROOT/venv/bin/python"
fi
[[ -x "$PYTHON_BIN" ]] || { echo "Python runtime not executable: $PYTHON_BIN" >&2; exit 3; }

# Copy code only. Runtime databases, reports and market data are persistent state.
rsync -a --delete \
  --exclude='.git' --exclude='*/.git' --exclude='generated' \
  --exclude='data_warehouse' --exclude='logs' --exclude='tmp' \
  --exclude='tmp_tx' --exclude='.cache' --exclude='__pycache__' \
  --exclude='*.pyc' --exclude='*.parquet' --exclude='*.log' \
  --exclude='config/*.sqlite*' --exclude='config/*.db' \
  --exclude='config/*.lock' --exclude='config/.env.secrets' \
  "$REPO_SOURCE/" "$STAGE/"

# Migrate a legacy direct install once, then make mutable directories shared.
for name in config generated data_warehouse logs reports; do
  shared_dir="$SHARED/$name"
  install -d -m 0770 -o "$SERVICE_USER" -g "$SERVICE_USER" "$shared_dir"
  legacy_dir="$APP_ROOT/$name"
  if [[ -d "$legacy_dir" && ! -L "$legacy_dir" ]]; then
    rsync -a "$legacy_dir/" "$shared_dir/"
  fi
  if [[ -d "$STAGE/$name" && ! -L "$STAGE/$name" ]]; then
    rsync -a "$STAGE/$name/" "$shared_dir/"
  fi
  rm -rf "$STAGE/$name"
  ln -s "$shared_dir" "$STAGE/$name"
done

printf 'workspace=bootstrap quant_system=bootstrap release=%s\n' "$RELEASE_ID" > "$STAGE/VERSION"

# Prepare and validate the root-only environment before publishing current. A
# missing production key must not leave a half-installed active release.
if [[ ! -f /etc/quant/quant.env ]]; then
  install -m 0600 "$STAGE/deploy/quant.env.example" /etc/quant/quant.env
fi
sed -i "s#^QUANT_ROOT=.*#QUANT_ROOT=$APP_ROOT/current#; s#^QUANT_PYTHON=.*#QUANT_PYTHON=$PYTHON_BIN#; s#^QUANT_REPORT_ROOT=.*#QUANT_REPORT_ROOT=$APP_ROOT/shared/reports#" /etc/quant/quant.env
if ! grep -q '^QUANT_REPORT_ROOT=' /etc/quant/quant.env; then
  printf 'QUANT_REPORT_ROOT=%s/shared/reports\n' "$APP_ROOT" >> /etc/quant/quant.env
fi
if [[ -n "${QUANT_WEB_API_KEY:-}" ]]; then
  if grep -q '^QUANT_WEB_API_KEY=' /etc/quant/quant.env; then
    sed -i "s#^QUANT_WEB_API_KEY=.*#QUANT_WEB_API_KEY=$QUANT_WEB_API_KEY#" /etc/quant/quant.env
  else
    printf '\nQUANT_WEB_API_KEY=%s\n' "$QUANT_WEB_API_KEY" >> /etc/quant/quant.env
  fi
fi
chmod 0600 /etc/quant/quant.env
API_KEY="$(sed -n 's/^QUANT_WEB_API_KEY=//p' /etc/quant/quant.env | head -1)"
ALLOW_UNAUTH="$(sed -n 's/^QUANT_WEB_ALLOW_UNAUTH=//p' /etc/quant/quant.env | head -1)"
if [[ -z "$API_KEY" || "$API_KEY" == "replace-with-a-long-random-secret" ]] && [[ "$ALLOW_UNAUTH" != "1" && "$ALLOW_UNAUTH" != "true" && "$ALLOW_UNAUTH" != "yes" ]]; then
  echo "Set QUANT_WEB_API_KEY in /etc/quant/quant.env before enabling the cloud service." >&2
  exit 5
fi

[[ ! -e "$TARGET" ]] || { echo "Release already exists: $TARGET" >&2; exit 4; }
mv "$STAGE" "$TARGET"
STAGE=""

# Preserve an existing release or old direct install for rollback.
if [[ -L "$APP_ROOT/current" ]]; then
  PREVIOUS="$(readlink -f "$APP_ROOT/current" || true)"
elif [[ -e "$APP_ROOT/current" ]]; then
  PREVIOUS="$APP_ROOT/current.legacy.$RELEASE_ID"
  mv "$APP_ROOT/current" "$PREVIOUS"
fi
CURRENT_LINK="$APP_ROOT/.current.install.$$"
ln -s "$TARGET" "$CURRENT_LINK"
mv -Tf "$CURRENT_LINK" "$APP_ROOT/current"

chown -R root:"$SERVICE_USER" "$TARGET"
chown -R "$SERVICE_USER:$SERVICE_USER" "$SHARED"
chmod 0750 "$TARGET"
for unit in quant-web.service quant-after-close.service quant-after-close.timer quant-data-update.service quant-data-update.timer quant-resident-recovery.service quant-resident-recovery.timer quant-resident-digest.service quant-resident-digest.timer; do
  rendered="$(mktemp /tmp/quant-unit.XXXXXX)"
  sed -e "s#/opt/quant#$APP_ROOT#g" -e "s#/opt/quant/venv/bin/python#$PYTHON_BIN#g" "$TARGET/deploy/$unit" > "$rendered"
  install -m 0644 "$rendered" "/etc/systemd/system/$unit"
  rm -f "$rendered"
done
systemctl daemon-reload
systemctl enable --now quant-web.service
systemctl enable --now quant-after-close.timer
systemctl enable --now quant-data-update.timer
systemctl enable --now quant-resident-recovery.timer
systemctl enable --now quant-resident-digest.timer

systemctl --no-pager --full status quant-web.service || true
trap - EXIT ERR INT TERM
rm -f "$APP_ROOT/.release.lock"
printf '\nInstall complete. Local probe: curl http://127.0.0.1:8600/api/version\n'
