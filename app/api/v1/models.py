import httpx
from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings

router = APIRouter()


@router.get("/providers/validate")
async def validate_provider_key(
    provider: str = Query(..., pattern="^(openrouter|groq|google)$"),
    api_key: str | None = Query(
        None, description="API key to validate; if omitted, server-configured key will be used"
    ),
):
    """Validate a provider API key by calling a lightweight endpoint.

    - openrouter: GET https://openrouter.ai/api/v1/models
    - groq: GET https://api.groq.com/openai/v1/models
    - google: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent
      with a trivial prompt
    """
    # Pick effective key (prefer explicit param, otherwise server config)
    if api_key is None:
        if provider == "openrouter":
            api_key = settings.OPENROUTER_API_KEY
        elif provider == "groq":
            api_key = settings.GROQ_API_KEY
        else:  # google
            api_key = settings.GOOGLE_AI_STUDIO_API_KEY

    if not api_key:
        return {"provider": provider, "valid": False, "status": 0, "detail": "key_not_configured"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            if provider == "openrouter":
                r = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                ok = r.status_code == 200
            elif provider == "groq":
                r = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                ok = r.status_code == 200
            else:  # google
                payload = {
                    "contents": [
                        {"parts": [{"text": "ping"}]}  # trivial request
                    ]
                }
                r = await client.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                    headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
                    json=payload,
                )
                ok = r.status_code == 200
    except httpx.HTTPError:
        ok = False
        r = None

    detail = None
    if r is not None and not ok:
        try:
            detail = r.json()
        except Exception:
            detail = r.text

    return {
        "provider": provider,
        "valid": ok,
        "status": r.status_code if r else 0,
        "detail": detail,
    }


@router.get("/models")
async def list_models(
    provider: str = Query("openrouter"),
    only_gemini: bool = Query(True),
    api_key: str | None = Query(None, description="Optional provider API key override"),
):
    """Return available models for the configured provider using server API keys.

    Currently supports provider=openrouter. When only_gemini=True, filters to Google Gemini models.
    """
    if provider != "openrouter":
        raise HTTPException(
            status_code=400, detail="Only 'openrouter' provider is supported currently"
        )

    effective_key = api_key or settings.OPENROUTER_API_KEY
    if not effective_key:
        raise HTTPException(
            status_code=503, detail="OpenRouter API key is not configured on the server"
        )

    url = "https://openrouter.ai/api/v1/models"
    headers = {"Authorization": f"Bearer {effective_key}"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch models: {e}") from e

    models = data.get("data", [])

    # Normalize and optionally filter
    normalized = []
    for m in models:
        model_id = m.get("id") or m.get("name")
        display = m.get("name") or model_id
        if not model_id:
            continue
        if only_gemini and not (
            str(model_id).startswith("google/") or "gemini" in str(model_id).lower()
        ):
            continue
        normalized.append({"id": model_id, "name": display})

    return {"models": normalized}
