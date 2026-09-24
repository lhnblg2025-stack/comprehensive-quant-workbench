from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "git_history_audit.sh"


def test_history_audit_is_read_only():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "git count-objects" in text
    assert "git ls-files" in text
    assert "printf 'git filter-repo" in text
    executable_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not any(line.startswith("git filter-repo ") for line in executable_lines)
    assert "git gc" not in text
    assert "git prune" not in text
