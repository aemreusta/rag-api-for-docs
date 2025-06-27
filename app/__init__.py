"""Application package initialisation.

This module applies a small *runtime* compatibility shim so that our internal
tests (and any legacy callers) can continue using the `httpx.AsyncClient(app=…)`
constructor signature that was removed in HTTPX ≥0.28.

The shim is intentionally lightweight and **only** activates when the outdated
`app` argument is supplied – all other usage patterns delegate to the original
implementation unchanged.
"""

from __future__ import annotations

from typing import Any

import httpx

_orig_async_client_init = httpx.AsyncClient.__init__  # type: ignore[attr-defined]


def _patched_async_client_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any):  # type: ignore[override]
    """Drop-in replacement that maps the deprecated `app` kwarg to ASGITransport."""

    app = kwargs.pop("app", None)
    if app is not None:
        # Lazily import to avoid unnecessary overhead when the kwarg is absent.
        from httpx import ASGITransport

        kwargs.setdefault("transport", ASGITransport(app=app))

    _orig_async_client_init(self, *args, **kwargs)  # type: ignore[misc]


# Monkey-patch once at import-time.
httpx.AsyncClient.__init__ = _patched_async_client_init  # type: ignore[assignment]
