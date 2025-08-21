from __future__ import annotations

from pathlib import Path

from app.core.storage import LocalStorageBackend


def test_local_storage_store_and_retrieve(tmp_path: Path):
    backend = LocalStorageBackend(tmp_path)
    data = b"hello world"
    uri = backend.store_file(data, "test.txt")
    assert uri.startswith("file://")

    out = backend.retrieve_file(uri)
    assert out == data
