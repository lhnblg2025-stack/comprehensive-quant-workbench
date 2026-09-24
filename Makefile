PYTHON ?= python3
PYTEST := $(PYTHON) -m pytest -q -p no:cacheprovider

.PHONY: lint-config fast api-contract contract integration slow ci help

# The lock file is the single source of truth for pinned runtime versions.
requirements.lock.txt:
	@true

help:
	@echo "lint-config    compile every Python package (syntax/import sanity)"
	@echo "fast           small public + release-safety contract set"
	@echo "api-contract   HTTP/API contract tests"
	@echo "contract       full non-integration suite"
	@echo "integration    tests marked 'integration'"
	@echo "slow           full suite including integration"
	@echo "ci             lint-config + fast + contract"

lint-config:
	$(PYTHON) -m compileall -q quant_system quant_platform quant_web scripts tests research

fast:
	$(PYTEST) tests/test_public.py tests/test_extended.py tests/test_cloud_release_safety.py tests/test_recovery_data_base.py

api-contract:
	$(PYTEST) quant_system/tests/test_api_contract.py

# Full non-integration suite. Tests that need the runtime data warehouse,
# generated artefacts or optional adapters are skipped by conftest.py with an
# explicit reason, so this target is safe on a clean checkout.
contract:
	$(PYTEST) tests quant_system/tests scripts/tests quant_web/tests -m 'not integration'

run_integration:
	$(PYTEST) tests quant_system/tests scripts/tests quant_web/tests -m integration

integration: run_integration
	@$(MAKE) run_integration

slow:
	$(PYTEST) tests quant_system/tests quant_web/tests scripts/tests

ci: lint-config fast contract
