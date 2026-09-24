"""Release gate: scan tracked files for credentials, PII and unpublishable assets.

This is a *helper* gate, not a proof of cleanliness. It never prints matched
secret values — only the path, line number and a category.

Usage:
    python3 scripts/check_release.py          # scan the tracked file set
    python3 scripts/check_release.py --all    # also scan untracked, non-ignored files
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

# Categories -> compiled patterns. Keep these conservative to avoid noise.
SENSITIVE = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |PGP )?PRIVATE KEY-----"),
    "github-token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "aws-key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "cloud-key": re.compile(r"\bLTAI[0-9A-Za-z]{12,}\b"),
    "openai-style-key": re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    "slack-token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "google-api-key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "assigned-secret": re.compile(
        r"""(?i)\b(?:api[_-]?key|apikey|access[_-]?key|secret|passwd|password|auth[_-]?token)\b"""
        r"""\s*[:=]\s*["'](?!(?:CHANGE_ME|REPLACE_ME|replace-me|your-|\$\{|env:|Bearer|\s*$)"""
        r""")[A-Za-z0-9_\-/+=]{16,}["']"""
    ),
    "connection-string": re.compile(r"(?i)\b(?:https?|ftp|ssh)://[^\s/:@]+:[^\s/@]+@"),
    "home-path": re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/"),
    "windows-user-path": re.compile(r"[A-Za-z]:\\+Users\\+[A-Za-z0-9._-]+"),
}

PII = {
    "cn-mobile": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "wechat-openid": re.compile(r"\bo[A-Za-z0-9_-]{25,}@im\.wechat\b"),
}

EXCLUDED_SUFFIXES = {
    ".pem", ".key", ".db", ".sqlite", ".sqlite3", ".parquet",
    ".pkl", ".pickle", ".joblib", ".pyc", ".log", ".zip", ".tar", ".gz", ".tgz",
}
EXCLUDED_NAMES = {".env", ".env.secrets", ":memory:"}

# Placeholder/documentation domains that must NOT be reported as real PII.
EMAIL_ALLOWLIST = re.compile(
    r"(?i)@(?:[\w.-]*\.)?example(?:\.com|\.org|\.net)?$"
    r"|@(?:company\.com|your-[a-z0-9.-]+|im\.wechat)$"
)
MAX_BYTES = 2 * 1024 * 1024

# Vendored third-party browser bundles: large by nature, reviewed once by hand.
# Their licences are documented in docs/RELEASE_SCOPE.md.
VENDORED_OK = {
    "quant_web/static/plotly.min.js",
    "quant_web/static/echarts.min.js",
}


def _tracked_files(include_untracked: bool) -> list[str]:
    args = ["git", "ls-files", "-z"]
    if include_untracked:
        args = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    out = subprocess.check_output(args, cwd=ROOT).decode("utf-8", "replace")
    return [n for n in out.split("\0") if n]


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def scan(include_untracked: bool = False) -> list[tuple[str, str, int, str]]:
    findings: list[tuple[str, str, int, str]] = []
    names = _tracked_files(include_untracked)
    for name in names:
        path = ROOT / name
        if not path.is_file():
            continue
        if path.is_symlink():
            findings.append((name, "link", 0, "symlink not allowed"))
            continue
        if ".git" in Path(name).parts:
            findings.append((name, "nested-git", 0, "nested git metadata"))
            continue
        size = path.stat().st_size
        if size > MAX_BYTES and name not in VENDORED_OK:
            findings.append((name, "oversized", 0, f"{size} bytes"))
        if path.suffix.lower() in EXCLUDED_SUFFIXES or path.name in EXCLUDED_NAMES:
            findings.append((name, "excluded-file", 0, path.suffix or path.name))
            continue
        if path.suffix.lower() in {".js", ".css"} and size > 512 * 1024:
            # vendored bundles are large but not scanned line-by-line
            continue
        text = _text(path)
        if text is None:
            findings.append((name, "binary", 0, "non-utf8"))
            continue
        if path.as_posix() == "scripts/check_release.py":
            continue
        for category, pattern in SENSITIVE.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append((name, category, line, "match"))
        for category, pattern in PII.items():
            for match in pattern.finditer(text):
                value = match.group(0)
                if category == "email" and EMAIL_ALLOWLIST.search(value):
                    continue
                line = text.count("\n", 0, match.start()) + 1
                findings.append((name, f"pii:{category}", line, "match"))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="include untracked, non-ignored files")
    args = parser.parse_args(argv)

    findings = scan(include_untracked=args.all)
    for name, category, line, detail in findings:
        location = f"{name}:{line}" if line else name
        print(f"{location}  {category}  ({detail})")
    total = len(_tracked_files(args.all))
    print(f"Scanned {total} files; {len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
