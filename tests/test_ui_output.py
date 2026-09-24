"""Live UI output regression: every page and public API must return output.

Skipped automatically when quant_web (127.0.0.1:8600) is not running, so this
does not make offline unit-test runs red.
"""
from __future__ import annotations

import socket

import pytest

from scripts.ui_output_random_test import (
    _classify,
    _extract_routes,
    _request,
    _static_pages,
)


def _server_up() -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect(("127.0.0.1", 8600))
        return True
    except OSError:
        return False
    finally:
        s.close()


pytestmark = pytest.mark.skipif(not _server_up(), reason="quant_web (127.0.0.1:8600) not running")


@pytest.mark.integration
def test_all_pages_return_200_with_output():
    for path in _static_pages():
        r = _request(path)
        assert r["status"] == 200, f"{path}: {r}"
        assert r["output_present"], f"{path}: empty body"


@pytest.mark.integration
def test_all_public_apis_return_output():
    classified = _classify(_extract_routes(), _static_pages())
    for path in classified["public_api"]:
        r = _request(path)
        assert r["status"] is not None and r["status"] < 500, f"{path}: {r}"
        assert r["output_present"], f"{path}: empty body"
