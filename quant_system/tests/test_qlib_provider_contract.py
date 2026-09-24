from __future__ import annotations

import pytest

from quant_system import qlib_handler


def test_qlib_handler_is_explicit_when_runtime_missing():
    if qlib_handler.DataHandlerLP is None:
        with pytest.raises(RuntimeError, match="pyqlib_runtime_missing"):
            qlib_handler.ASharePITHandler()
    else:
        assert issubclass(qlib_handler.ASharePITHandler, qlib_handler.DataHandlerLP)
