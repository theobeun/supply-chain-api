"""
Supply Chain Dashboard — FastAPI Backend (Async)
==================================================
POST /upload   → queue ML task, return task_id immediately
GET  /status/{task_id} → check task progress & get results when done
GET  /results  → return latest saved results
GET  /health   → healthcheck
DELETE /account → delete user account + data
"""

import hashlib
import os
import time
import uuid
import threading
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

from ml_pipeline import forecast_single_kpi, compute_correlations

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

app = FastAPI(title="Supply Chain Dashboard API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# In-memory task store
tasks: dict = {}

# ── Auth ──
async def get_current_user(request: Request) -> dict:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth_header[7:]
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{SUPABASE_URL}/auth/v1/user", headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_ANON_KEY})
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = r.json()
    user["access_token"] = token
    return user

# ── Supabase helpers ──
def _sb_headers(token: str) -> dict:
    return {"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {token}", "Content-Type": "application/json", "Prefer": "return=representation"}

def _sb_save_sync(user_id, token, data_hash, results):
    import requests as req
    h = _sb_headers(token)
    req.delete(f"{SUPABASE_URL}/rest/v1/ml_results?user_id=eq.{user_id}", headers=h)
    req.delete(f"{SUPABASE_URL}/rest/v1/users_data?user_id=eq.{user_id}", headers=h)
    req.post(f"{SUPABASE_URL}/rest/v1/users_data", headers=h, json={"user_id": user_id, "data_hash": data_hash, "raw_data": {"hash": data_hash}})
    req.post(f"{SUPABASE_URL}/rest/v1/ml_results", headers=h, json={"user_id": user_id, "data_hash": data_hash, "results_json": results})

# ── Models ──
class KPISeriesInput(BaseModel):
    kpi_name: str
    kpi_values: list[float]
    kpi_dates: list[str]
    feature_matrix: Optional[list[list[float]]] = None
    feature_names: list[str] = []
    horizons: list[int] = [1, 3]
    freq: str = "monthly"
    module: str = "economic"

class UploadPayload(BaseModel):
    series: list[KPISeriesInput]

# ── Background ML worker ──
def _run_ml_task(task_id, payload, user_id, token, data_hash):
    task = tasks[task_id]
    total = len(payload.series)
    task.update({"total": total, "status": "running"})
    all_results = {}
    t0 = time.time()

    for i, s in enumerate(payload.series):
        task.update({"progress": i, "current_kpi": s.kpi_name, "percent": int((i / max(total, 1)) * 100)})
        try:
            kpi_results = forecast_single_kpi(
                kpi_values=s.kpi_values, kpi_dates=s.kpi_dates,
                feature_matrix=s.feature_matrix, feature_names=s.feature_names,
                horizons=s.horizons, freq=s.freq,
            )
            correlations = compute_correlations(
                results=[{**r, "kpi": s.kpi_name} for r in kpi_results],
                kpi_values=s.kpi_values, kpi_dates=s.kpi_dates,
                feature_matrix=s.feature_matrix, feature_names=s.feature_names,
            )
            all_results[f"{s.module}__{s.kpi_name}"] = {"forecasts": kpi_results, "correlations": correlations}
        except Exception as e:
            all_results[f"{s.module}__{s.kpi_name}"] = {"forecasts": [], "correlations": [], "error": str(e)}

    comp_time = round(time.time() - t0, 2)
    results_package = {"kpis": all_results, "computation_time_s": comp_time, "data_hash": data_hash}

    try:
        _sb_save_sync(user_id, token, data_hash, results_package)
    except Exception as e:
        print(f"Warning: Supabase save failed: {e}")

    task.update({"status": "done", "progress": total, "percent": 100, "current_kpi": "", "results": results_package, "computation_time_s": comp_time})

# ── Endpoints ──
@app.get("/health")
async def health():
    active = sum(1 for t in tasks.values() if t["status"] == "running")
    return {"status": "ok", "supabase_url": SUPABASE_URL, "ml_pipeline": "loaded", "active_tasks": active}

@app.post("/upload")
async def upload_data(payload: UploadPayload, user: dict = Depends(get_current_user)):
    user_id, token = user["id"], user["access_token"]
    data_hash = hashlib.md5(payload.model_dump_json(exclude_none=True).encode()).hexdigest()

    # Check cache
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{SUPABASE_URL}/rest/v1/ml_results", headers=_sb_headers(token),
            params={"user_id": f"eq.{user_id}", "data_hash": f"eq.{data_hash}", "select": "results_json", "order": "computed_at.desc", "limit": "1"})
    if r.status_code == 200 and r.json():
        task_id = str(uuid.uuid4())
        tasks[task_id] = {"status": "done", "progress": len(payload.series), "total": len(payload.series),
            "percent": 100, "current_kpi": "", "results": r.json()[0]["results_json"], "cached": True, "computation_time_s": 0.0}
        return {"task_id": task_id, "cached": True, "total_kpis": len(payload.series)}

    # Launch background task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "queued", "progress": 0, "total": len(payload.series), "percent": 0, "current_kpi": "", "results": None, "cached": False, "error": None}
    threading.Thread(target=_run_ml_task, args=(task_id, payload, user_id, token, data_hash), daemon=True).start()
    return {"task_id": task_id, "cached": False, "total_kpis": len(payload.series)}

@app.get("/status/{task_id}")
async def get_task_status(task_id: str, user: dict = Depends(get_current_user)):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    resp = {"task_id": task_id, "status": task["status"], "progress": task.get("progress", 0),
        "total": task.get("total", 0), "percent": task.get("percent", 0),
        "current_kpi": task.get("current_kpi", ""), "cached": task.get("cached", False)}
    if task["status"] == "done":
        resp["results"] = task["results"]
        resp["computation_time_s"] = task.get("computation_time_s", 0)
    if task["status"] == "error":
        resp["error"] = task.get("error", "Unknown error")
    # Cleanup old tasks
    if len(tasks) > 30:
        old = [tid for tid, t in tasks.items() if t["status"] == "done" and tid != task_id]
        for tid in old[:len(old)-10]:
            tasks.pop(tid, None)
    return resp

@app.get("/results")
async def get_results(user: dict = Depends(get_current_user)):
    user_id, token = user["id"], user["access_token"]
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{SUPABASE_URL}/rest/v1/ml_results", headers=_sb_headers(token),
            params={"user_id": f"eq.{user_id}", "select": "results_json,data_hash,computed_at", "order": "computed_at.desc", "limit": "1"})
    if r.status_code != 200 or not r.json():
        return {"has_results": False, "results": None}
    row = r.json()[0]
    return {"has_results": True, "data_hash": row.get("data_hash"), "computed_at": row.get("computed_at"), "results": row["results_json"]}

@app.delete("/account")
async def delete_account(user: dict = Depends(get_current_user)):
    user_id, token = user["id"], user["access_token"]
    async with httpx.AsyncClient() as client:
        await client.delete(f"{SUPABASE_URL}/rest/v1/ml_results?user_id=eq.{user_id}", headers=_sb_headers(token))
        await client.delete(f"{SUPABASE_URL}/rest/v1/users_data?user_id=eq.{user_id}", headers=_sb_headers(token))
    if SUPABASE_SERVICE_KEY:
        async with httpx.AsyncClient() as client:
            await client.delete(f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}",
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"})
        return {"deleted": True, "message": "Account and all data permanently deleted"}
    return {"deleted": False, "message": "Data deleted but admin key missing for full account removal"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
