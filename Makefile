PYTHON ?= python3
PYTEST := $(PYTHON) -m pytest -q -p no:cacheprovider
SUITES := tests quant_system/tests scripts/tests quant_web/tests

.PHONY: lint-config fast api-contract contract integration slow ci help

help:
	@echo "lint-config    compile every Python package (syntax/import sanity)"
	@echo "fast           small public + release-safety contract set"
	@echo "api-contract   FastAPI/http contract tests"
	@echo "ci             what CI runs: lint-config + fast + api-contract"
	@echo "contract       full non-integration suite (needs runtime data for some tests)"
	@echo "integration    tests marked 'integration'"
	@echo "slow           full suite including integration"

lint-config:
	$(PYTHON) -m compileall -q quant_system quant_platform quant_web scripts tests research

fast:
	$(PYTEST) tests/test_public.py tests/test_extended.py tests/test_cloud_release_safety.py tests/test_recovery_data_base.py

api-contract:
	$(PYTEST) quant_system/tests/test_api_contract.py

# Full non-integration suite. Some tests need the runtime data warehouse and
# generated artefacts; conftest.py skips them with an explicit reason when those
# assets are absent, so this target is safe to run on a clean clone.
contract:
	$(PYTEST) $(SUITES) -m 'not integration'

run_integration:
	$(PYTEST) $(SUITES) -m integration

integration: run_integration
	@$(MAKE) run_integration

slow:
	$(PYTEST) $(SUITES)

ci: lint-config fast api-contract
