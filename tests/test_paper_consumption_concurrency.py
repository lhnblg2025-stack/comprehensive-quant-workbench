from __future__ import annotations

import json
import multiprocessing
import threading
from pathlib import Path


def test_paper_preview_claim_is_single_consumer(monkeypatch, tmp_path):
    import quant_web.server as server

    monkeypatch.setattr(server, "ROOT", tmp_path)
    server._PAPER_ORDER_IDS.clear()
    server._write_paper_order_records({
        "order-1": {
            "kind": "preview",
            "fingerprint": "fp",
            "expires_at": 4102444800,
            "preview": {"client_order_id": "order-1"},
        }
    })
    results = []
    barrier = threading.Barrier(2)

    def consume():
        barrier.wait()
        results.append(server._claim_paper_preview("order-1", "fp"))

    workers = [threading.Thread(target=consume) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    claimed = [record for record, duplicate in results if record is not None]
    duplicates = [duplicate for record, duplicate in results if duplicate is not None]
    assert len(claimed) == 1
    assert claimed[0]["kind"] == "processing"
    assert len(duplicates) == 1
    assert duplicates[0]["kind"] == "processing"


def _write_delivery(index_path: str, delivery_id: str) -> None:
    import scripts.execution_ledger as ledger

    ledger.DELIVERY_INDEX = Path(index_path)
    ledger.DELIVERY_DIR = Path(index_path).parent
    ledger.LEDGER_LOCK = Path(index_path).parent.parent / ".execution_ledger.lock"
    ledger.record_delivery(delivery_id=delivery_id, record={"status": "ok"})


def test_execution_ledger_preserves_cross_process_writes(tmp_path):
    import scripts.execution_ledger as ledger

    index = tmp_path / "delivery_receipts" / "index.json"
    ledger.DELIVERY_INDEX = index
    ledger.DELIVERY_DIR = index.parent
    ledger.LEDGER_LOCK = tmp_path / ".execution_ledger.lock"
    ctx = multiprocessing.get_context("fork")
    processes = [ctx.Process(target=_write_delivery, args=(str(index), f"p-{i}")) for i in range(8)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0

    payload = json.loads(index.read_text(encoding="utf-8"))
    assert {f"p-{i}" for i in range(8)}.issubset(payload["entries"])
    assert payload["count"] == len(payload["entries"])
