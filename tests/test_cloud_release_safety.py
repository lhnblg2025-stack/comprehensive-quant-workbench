from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_pull_module():
    spec = importlib.util.spec_from_file_location("pull_cloud_data_tested", ROOT / "scripts" / "pull_cloud_data.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pull_cloud_ssh_options_require_known_hosts(tmp_path, monkeypatch):
    module = _load_pull_module()
    known_hosts = tmp_path / "known_hosts"
    monkeypatch.setattr(module, "KNOWN_HOSTS", known_hosts)
    try:
        module._ssh_options()
        assert False, "missing known_hosts must fail closed"
    except RuntimeError:
        pass
    known_hosts.write_text("example ssh-ed25519 AAAATEST\n", encoding="utf-8")
    options = module._ssh_options()
    joined = " ".join(options)
    assert "StrictHostKeyChecking=yes" in joined
    assert "StrictHostKeyChecking=no" not in joined
    assert f"UserKnownHostsFile={known_hosts}" in joined


def test_cloud_sources_write_to_canonical_domains():
    module = _load_pull_module()
    assert module.SOURCES["hot_rank"]["local_dir"] == ROOT / "data_warehouse" / "hot_rank"
    assert module.SOURCES["fund_flow"]["local_dir"] == ROOT / "data_warehouse" / "fund_flow"


def test_failed_transfer_preserves_existing_file(tmp_path, monkeypatch):
    module = _load_pull_module()
    destination = tmp_path / "domain"
    destination.mkdir()
    existing = destination / "sample.json"
    existing.write_text('{"old": true}', encoding="utf-8")
    module.SOURCES["test"] = {
        "host": "example", "remote_dir": "/remote", "local_dir": destination,
        "pattern": ("sample", ".json"), "is_json": True, "compress": False,
    }
    monkeypatch.setattr(module, "_ls_remote", lambda *_: [("sample.json", 100)])
    monkeypatch.setattr(module, "_ssh_options", lambda: [])
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "failed"))
    assert module.pull_source("test", force=True) == 0
    assert existing.read_text(encoding="utf-8") == '{"old": true}'
    assert not list(destination.glob("*.part"))


def test_sync_dry_run_is_cwd_independent_and_excludes_secrets(tmp_path):
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("example ssh-ed25519 AAAATEST\n", encoding="utf-8")
    work = tmp_path / "work"
    logs = tmp_path / "logs"
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / "sync_to_cloud.sh")],
        cwd=tmp_path,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "QUANT_KNOWN_HOSTS": str(known_hosts),
            "QUANT_SYNC_WORK_DIR": str(work),
            "QUANT_SYNC_LOG_DIR": str(logs),
            "QUANT_SYNC_DRY_RUN": "1",
            "QUANT_RELEASE_ID": "test-release",
            "CLOUD_KIND": "linux",
        },
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    archive = work / "quant_code_test-release.tar.gz"
    assert archive.is_file()
    names = subprocess.check_output(["tar", "-tzf", str(archive)], text=True).splitlines()
    assert any(name.endswith("quant_web/server.py") for name in names)
    assert any(name.endswith("VERSION") for name in names)
    forbidden = (".env.secrets", ".pem", ".key", "credentials/", "xueqiu_cookie.json")
    assert not any(any(token in name for token in forbidden) for name in names)


def test_install_and_release_share_same_runtime_contract():
    install = (ROOT / "deploy" / "install_tencent_cloud.sh").read_text(encoding="utf-8")
    release = (ROOT / "deploy" / "remote_release_linux.sh").read_text(encoding="utf-8")
    service = (ROOT / "deploy" / "quant-web.service").read_text(encoding="utf-8")
    for text in (install, release):
        assert 'APP_ROOT="${QUANT_ROOT:-/opt/quant}"' in text or 'APP_HOME="${QUANT_RELEASE_HOME:-/opt/quant}"' in text
        assert 'SHARED="$APP_ROOT/shared"' in text or 'SHARED="$APP_HOME/shared"' in text
        assert 'flock' in text
        assert '0770' in text
    assert 'WorkingDirectory=/opt/quant/current' in service
    assert 'Environment=QUANT_REPORT_ROOT=/opt/quant/shared/reports' in service
    assert 'ReadWritePaths=/opt/quant/current /opt/quant/shared' in service


def test_release_updates_units_and_has_rollback_cleanup():
    release = (ROOT / "deploy" / "remote_release_linux.sh").read_text(encoding="utf-8")
    assert 'daemon-reload' in release
    assert 'UNIT_BACKUP' in release
    assert 'rm -rf "$UNIT_BACKUP"' in release
    assert 'mv -Tf "$CURRENT_LINK" "$CURRENT"' in release
    assert 'trap rollback ERR INT TERM' in release


def test_local_release_preserves_shared_state_and_rolls_back(tmp_path):
    import getpass
    import grp
    import os
    import pwd
    import tarfile

    user = getpass.getuser()
    group = grp.getgrgid(pwd.getpwnam(user).pw_gid).gr_name
    app = tmp_path / "app"
    units = tmp_path / "units"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    systemctl_log = tmp_path / "systemctl.log"
    systemctl = bin_dir / "systemctl"
    systemctl.write_text(f"#!/bin/sh\necho \"$@\" >> {systemctl_log}\nexit 0\n", encoding="utf-8")
    systemctl.chmod(0o755)
    curl = bin_dir / "curl"
    curl.write_text("#!/bin/sh\nexit \"${FAKE_CURL_RC:-0}\"\n", encoding="utf-8")
    curl.chmod(0o755)

    def make_archive(release_id: str, marker: str) -> Path:
        source = tmp_path / f"src-{release_id}"
        (source / "quant_web").mkdir(parents=True)
        (source / "quant_web" / "server.py").write_text(marker, encoding="utf-8")
        (source / "config").mkdir()
        (source / "config" / "defaults.json").write_text(marker, encoding="utf-8")
        (source / "deploy").mkdir()
        for unit in ("quant-web.service", "quant-after-close.service", "quant-after-close.timer", "quant-data-update.service", "quant-data-update.timer"):
            (source / "deploy" / unit).write_text(f"Description={marker}\nWorkingDirectory=/opt/quant/current\n", encoding="utf-8")
        (source / "VERSION").write_text(f"release={release_id}\n", encoding="utf-8")
        archive = tmp_path / f"{release_id}.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            for path in source.rglob("*"):
                tf.add(path, arcname=path.relative_to(source))
        return archive

    env = {
        **os.environ,
        "QUANT_RELEASE_HOME": str(app),
        "QUANT_RELEASE_UNIT_DIR": str(units),
        "QUANT_RELEASE_USER": user,
        "QUANT_RELEASE_CODE_OWNER": f"{user}:{group}",
        "SYSTEMCTL_BIN": str(systemctl),
        "CURL_BIN": str(curl),
    }
    script = ROOT / "deploy" / "remote_release_linux.sh"
    first = subprocess.run(["bash", str(script), str(make_archive("r1", "one")), "r1"], env=env, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    assert (app / "current" / "quant_web" / "server.py").read_text(encoding="utf-8") == "one"
    (app / "shared" / "generated" / "state.json").write_text("keep", encoding="utf-8")

    failed_env = {**env, "FAKE_CURL_RC": "1"}
    second = subprocess.run(["bash", str(script), str(make_archive("r2", "two")), "r2"], env=failed_env, text=True, capture_output=True)
    assert second.returncode != 0
    assert (app / "current" / "quant_web" / "server.py").read_text(encoding="utf-8") == "one"
    assert (app / "shared" / "generated" / "state.json").read_text(encoding="utf-8") == "keep"
    assert not (app / "releases" / "r2").exists()
    assert not list(app.glob(".units-backup.*"))
