PYTHON ?= python3

.PHONY: lint-config fast contract integration slow ci
requirements.lock.txt:
	@true
lint-config:
	$(PYTHON) -m compileall -q quant_system quant_platform quant_web scripts tests
fast:
	$(PYTHON) -m pytest -q tests/test_public.py tests/test_extended.py tests/test_cloud_release_safety.py tests/test_recovery_data_base.py
api-contract:
	$(PYTHON) -m pytest -q tests/test_api_contract.py quant_system/tests/test_api_contract.py
contract:
	$(PYTHON) -m pytest -q tests quant_system/tests scripts/tests quant_web/tests -m 'not integration'
run_integration:
	$(PYTHON) -m pytest -q tests quant_system/tests scripts/tests quant_web/tests -m integration
integration: run_integration
	@$(MAKE) run_integration
slow:
	$(PYTHON) -m pytest -q tests quant_system/tests quant_web/tests scripts/tests
ci: lint-config fast contract
