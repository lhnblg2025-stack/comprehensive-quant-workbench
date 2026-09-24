#!/usr/bin/env bash
# Atomic Linux release installer. Designed for local sandbox tests and remote SSH use.
set -euo pipefail

ARCHIVE="${1:?archive path required}"
RELEASE_ID="${2:?release id required}"
APP_HOME="${QUANT_RELEASE_HOME:-/opt/quant}"
SERVICE="${QUANT_RELEASE_SERVICE:-quant-web.service}"
HEALTH_URL="${QUANT_RELEASE_HEALTH_URL:-http://127.0.0.1:8600/api/livez}"
KEEP_RELEASES="${QUANT_RELEASE_KEEP:-3}"
SYSTEMCTL_BIN="${SYSTEMCTL_BIN:-systemctl}"
CURL_BIN="${CURL_BIN:-curl}"
SERVICE_USER="${QUANT_RELEASE_USER:-quant}"
CODE_OWNER="${QUANT_RELEASE_CODE_OWNER:-root:$SERVICE_USER}"
UNIT_DIR="${QUANT_RELEASE_UNIT_DIR:-/etc/systemd/system}"
CHOWN_BIN="${CHOWN_BIN:-chown}"

RELEASES="$APP_HOME/releases"
SHARED="$APP_HOME/shared"
CURRENT="$APP_HOME/current"
LOCK_FILE="$APP_HOME/.release.lock"
PREVIOUS=""
STAGE=""
TARGET=""
UNIT_BACKUP=""
UNITS=(quant-web.service quant-after-close.service quant-after-close.timer quant-data-update.service quant-data-update.timer quant-resident-recovery.service quant-resident-recovery.timer quant-resident-digest.service quant-resident-digest.timer)
REQUIRED_UNITS=(quant-web.service quant-after-close.service quant-after-close.timer quant-data-update.service quant-data-update.timer)

case "$RELEASE_ID" in
  (*[!A-Za-z0-9._-]*|'') echo "invalid release id" >&2; exit 2 ;;
esac
[[ -f "$ARCHIVE" ]] || { echo "archive not found: $ARCHIVE" >&2; exit 2; }
id "$SERVICE_USER" >/dev/null 2>&1 || { echo "service user not found: $SERVICE_USER" >&2; exit 3; }
for tool in tar flock readlink mv mkdir find sort awk install sed; do
  command -v "$tool" >/dev/null 2>&1 || { echo "required command not found: $tool" >&2; exit 3; }
done

install -d -m 0755 "$RELEASES" "$SHARED"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another release is being installed" >&2
  exit 75
fi

TARGET="$RELEASES/$RELEASE_ID"
[[ ! -e "$TARGET" ]] || { echo "release already exists: $TARGET" >&2; exit 2; }
STAGE="$(mktemp -d "$RELEASES/.stage-${RELEASE_ID}.XXXXXX")"

rollback() {
  local rc=$?
  trap - ERR INT TERM
  if [[ -n "$UNIT_BACKUP" && -d "$UNIT_BACKUP" ]]; then
    for unit in "${UNITS[@]}"; do
      if [[ -f "$UNIT_BACKUP/$unit" ]]; then
        install -m 0644 "$UNIT_BACKUP/$unit" "$UNIT_DIR/$unit"
      else
        rm -f "$UNIT_DIR/$unit"
      fi
    done
    "$SYSTEMCTL_BIN" daemon-reload >/dev/null 2>&1 || true
  fi
  if [[ -n "$PREVIOUS" && -e "$PREVIOUS" ]]; then
    local rollback_link="$APP_HOME/.current.rollback.$$"
    ln -s "$PREVIOUS" "$rollback_link"
    mv -Tf "$rollback_link" "$CURRENT"
    "$SYSTEMCTL_BIN" restart "$SERVICE" >/dev/null 2>&1 || true
  elif [[ -L "$CURRENT" ]]; then
    rm -f "$CURRENT"
  fi
  [[ -z "$UNIT_BACKUP" ]] || rm -rf "$UNIT_BACKUP"
  [[ -z "$STAGE" ]] || rm -rf "$STAGE"
  [[ -z "$TARGET" ]] || rm -rf "$TARGET"
  exit "$rc"
}
trap rollback ERR INT TERM

# Reject absolute and parent-traversal archive members before extraction.
if tar -tzf "$ARCHIVE" | awk 'BEGIN{bad=0} /^\// || /(^|\/)\.\.($|\/)/ {bad=1} END{exit bad}'; then
  :
else
  echo "unsafe archive member path" >&2
  exit 4
fi

tar -xzf "$ARCHIVE" -C "$STAGE"
[[ -f "$STAGE/quant_web/server.py" ]] || { echo "invalid release: missing quant_web/server.py" >&2; false; }
[[ -f "$STAGE/VERSION" ]] || { echo "invalid release: missing VERSION" >&2; false; }
if ! grep -q "release=$RELEASE_ID" "$STAGE/VERSION"; then
  echo "invalid release: VERSION does not match $RELEASE_ID" >&2
  false
fi

# Runtime state survives code releases. Merge release config without deleting
# locally managed secrets, then expose every mutable area through shared links.
for name in config generated data_warehouse logs reports; do
  shared_dir="$SHARED/$name"
  install -d -m 0770 -o "$SERVICE_USER" -g "$SERVICE_USER" "$shared_dir"
  if [[ -d "$STAGE/$name" && ! -L "$STAGE/$name" ]]; then
    tar -cf - -C "$STAGE/$name" . | tar -xf - -C "$shared_dir"
  fi
  rm -rf "$STAGE/$name"
  ln -s "$shared_dir" "$STAGE/$name"
done

if [[ -L "$CURRENT" ]]; then
  PREVIOUS="$(readlink -f "$CURRENT" || true)"
  [[ -z "$PREVIOUS" || -d "$PREVIOUS" ]] || { echo "current symlink target is invalid" >&2; false; }
elif [[ -e "$CURRENT" ]]; then
  echo "current must be a symlink; refusing to overwrite it" >&2
  false
fi

mv "$STAGE" "$TARGET"
STAGE=""
"$CHOWN_BIN" -R "$CODE_OWNER" "$TARGET"
chmod 0750 "$TARGET"

UNIT_BACKUP="$(mktemp -d "$APP_HOME/.units-backup.XXXXXX")"
install -d -m 0755 "$UNIT_DIR"
for unit in "${UNITS[@]}"; do
  if [[ ! -f "$TARGET/deploy/$unit" ]]; then
    if printf '%s\n' "${REQUIRED_UNITS[@]}" | grep -qx "$unit"; then
      echo "invalid release: missing deploy/$unit" >&2; false
    fi
    continue
  fi
  [[ ! -f "$UNIT_DIR/$unit" ]] || cp -a "$UNIT_DIR/$unit" "$UNIT_BACKUP/$unit"
  rendered="$UNIT_BACKUP/$unit.new"
  sed "s#/opt/quant#$APP_HOME#g" "$TARGET/deploy/$unit" > "$rendered"
  install -m 0644 "$rendered" "$UNIT_DIR/$unit"
done
"$SYSTEMCTL_BIN" daemon-reload
"$SYSTEMCTL_BIN" enable quant-after-close.timer quant-data-update.timer quant-resident-recovery.timer quant-resident-digest.timer

CURRENT_LINK="$APP_HOME/.current.release.$$"
ln -s "$TARGET" "$CURRENT_LINK"
mv -Tf "$CURRENT_LINK" "$CURRENT"
"$SYSTEMCTL_BIN" restart "$SERVICE"

healthy=false
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if "$CURL_BIN" -fsS --max-time 3 "$HEALTH_URL" >/dev/null; then
    healthy=true
    break
  fi
  sleep 1
done
[[ "$healthy" == true ]] || { echo "health check failed: $HEALTH_URL" >&2; false; }
"$SYSTEMCTL_BIN" is-active --quiet "$SERVICE"

trap - ERR INT TERM
rm -rf "$UNIT_BACKUP"
UNIT_BACKUP=""
rm -f "$ARCHIVE"
# Keep a bounded number of release directories; never remove current.
mapfile -t old < <(find "$RELEASES" -mindepth 1 -maxdepth 1 -type d ! -name '.stage-*' -printf '%T@ %p\n' | sort -nr | awk -v keep="$KEEP_RELEASES" 'NR>keep {sub(/^[^ ]+ /, ""); print}')
current_real="$(readlink -f "$CURRENT" || true)"
for path in "${old[@]:-}"; do
  [[ -n "$path" && "$path" != "$current_real" ]] && rm -rf "$path"
done
printf 'DEPLOY_OK release=%s previous=%s\n' "$RELEASE_ID" "${PREVIOUS:-none}"
