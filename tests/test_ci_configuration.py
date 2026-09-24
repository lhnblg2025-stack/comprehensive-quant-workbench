from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = ROOT / "Makefile"


def test_ci_workflow_has_stable_test_tiers():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'PYTHON_VERSION: "3.10"' in text
    assert "requirements.lock.txt" in text
    assert "fast-contract:" in text
    assert "integration:" in text
    assert "slow-nightly:" in text
    assert "workflow_dispatch:" in text
    assert "run_integration" in text
    assert "schedule:" in text
    assert "make ci" in text
    assert "make integration" in text
    assert "make slow" in text


def test_makefile_covers_root_and_independent_core_repo():
    text = MAKEFILE.read_text(encoding="utf-8")
    assert "requirements.lock.txt" in text
    assert "tests quant_system/tests scripts/tests" in text
    assert "tests/test_api_contract.py" in text
    assert "quant_system/tests/test_api_contract.py" in text
    assert "ci: lint-config fast contract" in text
    assert "integration:" in text
    assert "slow:" in text
