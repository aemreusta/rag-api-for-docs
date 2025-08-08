import httpx
from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_api_key
from app.core.config import settings

router = APIRouter()


@router.get("/models", dependencies=[Depends(get_api_key)])
async def list_models(
    provider: str = Query("openrouter"),
    only_gemini: bool = Query(True),
):
    """Return available models for the configured provider using server API keys.

    Currently supports provider=openrouter. When only_gemini=True, filters to Google Gemini models.
    """
    if provider != "openrouter":
        raise HTTPException(
            status_code=400, detail="Only 'openrouter' provider is supported currently"
        )

    if not settings.OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=503, detail="OpenRouter API key is not configured on the server"
        )

    url = "https://openrouter.ai/api/v1/models"
    headers = {"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"}

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
