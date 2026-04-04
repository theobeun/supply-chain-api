"""
Supply Chain Dashboard — FastAPI Backend
=========================================
Three endpoints:
  POST /upload   → receive data, check cache, run ML if needed, return results
  GET  /results  → return latest results for the authenticated user
  GET  /health   → healthcheck

Auth: Supabase JWT token verified on every request via middleware.
Cache: MD5 hash of uploaded data → skip ML if same data already processed.
"""

import hashlib
import json
import os
import time
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# ── ML pipeline import ──
from ml_pipeline import forecast_single_kpi, compute_correlations, pearson

# ══════════════════════════════════════════════════════════════
# CONFIG — loaded from environment variables
# ══════════════════════════════════════════════════════════════

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://zzrbsmlxbpdpifzvotwz.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")  # Set via env var
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")  # Optional, for server-side ops

# ══════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Supply Chain Dashboard API",
    version="1.0.0",
    description="ML forecasting backend for the Supply Chain KPI Dashboard",
)

# CORS — allow the dashboard HTML to call us from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
# AUTH — verify Supabase JWT on every protected request
# ══════════════════════════════════════════════════════════════

async def get_current_user(request: Request) -> dict:
    """
    Extract and verify the Supabase JWT from the Authorization header.
    Returns the user object from Supabase.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header[7:]  # Strip "Bearer "

    # Verify token by calling Supabase auth endpoint
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": SUPABASE_ANON_KEY,
            },
        )

    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = response.json()
    user["access_token"] = token  # Keep token for Supabase data operations
    return user


# ══════════════════════════════════════════════════════════════
# SUPABASE DATA HELPERS
# ══════════════════════════════════════════════════════════════

def _supabase_headers(token: str) -> dict:
    """Headers for authenticated Supabase REST API calls."""
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


async def _supabase_get(table: str, token: str, params: dict) -> list:
    """GET rows from a Supabase table (filtered by RLS automatically)."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=_supabase_headers(token),
            params=params,
        )
    if response.status_code != 200:
        return []
    return response.json()


async def _supabase_upsert(table: str, token: str, data: dict) -> bool:
    """INSERT or UPDATE a row in a Supabase table."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers={
                **_supabase_headers(token),
                "Prefer": "resolution=merge-duplicates,return=representation",
            },
            json=data,
        )
    return response.status_code in (200, 201)


async def _supabase_delete(table: str, token: str, params: dict) -> bool:
    """DELETE rows from a Supabase table."""
    async with httpx.AsyncClient() as client:
        response = await client.delete(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=_supabase_headers(token),
            params=params,
        )
    return response.status_code in (200, 204)


# ══════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════

class KPISeriesInput(BaseModel):
    """A single KPI time series with its associated features."""
    kpi_name: str
    kpi_values: list[float]
    kpi_dates: list[str]
    feature_matrix: Optional[list[list[float]]] = None
    feature_names: list[str] = []
    horizons: list[int] = [1, 3]
    freq: str = "monthly"
    module: str = "economic"  # economic, cs, supplier, operational, sku


class UploadPayload(BaseModel):
    """Full upload payload: one or more KPI series to forecast."""
    series: list[KPISeriesInput]


class UploadResponse(BaseModel):
    cached: bool
    computation_time_s: float
    results: dict  # Full results JSON


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Simple healthcheck — used by the dashboard to show server status."""
    return {
        "status": "ok",
        "supabase_url": SUPABASE_URL,
        "ml_pipeline": "loaded",
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_data(payload: UploadPayload, user: dict = Depends(get_current_user)):
    """
    Receive KPI data, check cache, run ML pipeline if needed.

    Flow:
    1. Compute MD5 hash of the payload
    2. Check if this exact data was already processed (cache hit)
    3. If cached → return stored results immediately
    4. If not → run ML pipeline, store results, return them
    """
    user_id = user["id"]
    token = user["access_token"]

    # ── 1. Compute hash of the input data ──
    payload_json = payload.model_dump_json(exclude_none=True)
    data_hash = hashlib.md5(payload_json.encode()).hexdigest()

    # ── 2. Check cache ──
    cached_results = await _supabase_get("ml_results", token, {
        "user_id": f"eq.{user_id}",
        "data_hash": f"eq.{data_hash}",
        "select": "results_json,computed_at",
        "order": "computed_at.desc",
        "limit": "1",
    })

    if cached_results:
        return UploadResponse(
            cached=True,
            computation_time_s=0.0,
            results=cached_results[0]["results_json"],
        )

    # ── 3. Run ML pipeline ──
    t0 = time.time()
    all_results = {}

    for s in payload.series:
        try:
            kpi_results = forecast_single_kpi(
                kpi_values=s.kpi_values,
                kpi_dates=s.kpi_dates,
                feature_matrix=s.feature_matrix,
                feature_names=s.feature_names,
                horizons=s.horizons,
                freq=s.freq,
            )

            # Compute correlations for best models
            correlations = compute_correlations(
                results=[{**r, "kpi": s.kpi_name} for r in kpi_results],
                kpi_values=s.kpi_values,
                kpi_dates=s.kpi_dates,
                feature_matrix=s.feature_matrix,
                feature_names=s.feature_names,
            )

            key = f"{s.module}__{s.kpi_name}"
            all_results[key] = {
                "forecasts": kpi_results,
                "correlations": correlations,
            }
        except Exception as e:
            all_results[f"{s.module}__{s.kpi_name}"] = {
                "forecasts": [],
                "correlations": [],
                "error": str(e),
            }

    computation_time = time.time() - t0

    results_package = {
        "kpis": all_results,
        "computation_time_s": round(computation_time, 2),
        "data_hash": data_hash,
    }

    # ── 4. Store in Supabase (replace old results for this user) ──
    # Delete old results for this user
    await _supabase_delete("ml_results", token, {"user_id": f"eq.{user_id}"})
    await _supabase_delete("users_data", token, {"user_id": f"eq.{user_id}"})

    # Store new data reference
    await _supabase_upsert("users_data", token, {
        "user_id": user_id,
        "data_hash": data_hash,
        "raw_data": {"series_count": len(payload.series), "hash": data_hash},
    })

    # Store ML results
    await _supabase_upsert("ml_results", token, {
        "user_id": user_id,
        "data_hash": data_hash,
        "results_json": results_package,
    })

    return UploadResponse(
        cached=False,
        computation_time_s=round(computation_time, 2),
        results=results_package,
    )


@app.get("/results")
async def get_results(user: dict = Depends(get_current_user)):
    """
    Return the latest ML results for the authenticated user.
    Used when the dashboard loads — if results exist, display them immediately.
    """
    user_id = user["id"]
    token = user["access_token"]

    rows = await _supabase_get("ml_results", token, {
        "user_id": f"eq.{user_id}",
        "select": "results_json,data_hash,computed_at",
        "order": "computed_at.desc",
        "limit": "1",
    })

    if not rows:
        return {"has_results": False, "results": None}

    return {
        "has_results": True,
        "data_hash": rows[0].get("data_hash"),
        "computed_at": rows[0].get("computed_at"),
        "results": rows[0]["results_json"],
    }


# ══════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
