"""Scan public files without printing matched secret values."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
patterns = [
    re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'),
    re.compile(r'AKIA[A-Z0-9]{16}'),
    re.compile(r'sk-[A-Za-z0-9_-]{24,}'),
    re.compile(r'(?:api_key|password|secret|token)\s*[:=]\s*[\"\'][A-Za-z0-9_\-/+=]{20,}[\"\']', re.I),
    re.compile(r'/home/(?:ethan|ubuntu|runner|user)/'),
    re.compile(r'(?:^|[\/])(?:private_recovery|recovery_backups|recovered_[A-Za-z0-9_-]+)(?:[\/]|$)'),
]

def scan():
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    failures = []
    for name in filter(None, tracked):
        path = ROOT / name
        if path.is_symlink() or '.git' in Path(name).parts:
            failures.append((name, 'link-or-nested-git'))
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            failures.append((name, 'oversized'))
        if path.suffix.lower() in {'.pem', '.key', '.db', '.sqlite', '.parquet', '.pkl'} or path.name == '.env':
            failures.append((name, 'excluded-file'))
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeError:
            failures.append((name, 'binary'))
            continue
        if name != 'scripts/check_release.py' and any(p.search(text) for p in patterns):
            failures.append((name, 'sensitive-pattern'))
    for name, reason in failures:
        print(name, reason)
    print('Scanned', len([x for x in tracked if x]), 'files;', len(failures), 'findings')
    return bool(failures)

if __name__ == '__main__':
    raise SystemExit(scan())
