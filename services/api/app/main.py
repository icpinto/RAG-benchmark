from __future__ import annotations
from pathlib import Path
import os, shutil, uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from ragbench_core.engine.database import init_db, get_run, list_runs, list_results, upsert_run
from ragbench_core.utils.config import load_yaml, parse_yaml_text
from services.worker.app.tasks import run_benchmark_task

app=FastAPI(title="RAG Benchmark Platform API", version="0.1.0")
CONFIG_ROOT=Path(os.getenv("RAGBENCH_CONFIG_DIR", "/app/configs")).resolve()

class RunRequest(BaseModel):
    config_path: str = "/app/configs/default.yaml"

class ConfigSaveRequest(BaseModel):
    config_path: str = "/app/configs/default.yaml"
    content: str

def _resolve_config_path(config_path: str) -> Path:
    raw=Path(config_path)
    path=(raw if raw.is_absolute() else CONFIG_ROOT / raw).resolve()
    try:
        path.relative_to(CONFIG_ROOT)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Config path must be inside {CONFIG_ROOT}")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise HTTPException(status_code=400, detail="Config path must end with .yaml or .yml")
    return path

@app.on_event("startup")
def startup():
    init_db()

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/configs")
def list_configs():
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    files=sorted(
        str(path.relative_to(CONFIG_ROOT))
        for path in CONFIG_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )
    return {"config_root":str(CONFIG_ROOT),"files":files}

@app.get("/configs/content")
def get_config_content(config_path: str = Query("/app/configs/default.yaml")):
    path=_resolve_config_path(config_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Config not found: {path}")
    return {"config_path":str(path),"content":path.read_text(encoding="utf-8")}

@app.put("/configs/content")
def save_config_content(req: ConfigSaveRequest):
    path=_resolve_config_path(req.config_path)
    try:
        parsed=parse_yaml_text(req.content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(req.content, encoding="utf-8")
    return {"config_path":str(path),"experiment_name":parsed.get("experiment",{}).get("name","benchmark"),"status":"saved"}

@app.post("/benchmarks/run")
def run_benchmark(req: RunRequest):
    if not Path(req.config_path).exists():
        raise HTTPException(status_code=404, detail=f"Config not found: {req.config_path}")
    config=load_yaml(req.config_path)
    run_id=str(uuid.uuid4())
    upsert_run(run_id, config.get("experiment",{}).get("name","benchmark"), "queued", config)
    run_benchmark_task.delay(run_id, req.config_path)
    return {"run_id":run_id,"status":"queued"}

@app.get("/benchmarks")
def benchmarks():
    return list_runs()

@app.get("/benchmarks/{run_id}")
def benchmark(run_id: str):
    row=get_run(run_id)
    if not row: raise HTTPException(status_code=404, detail="Run not found")
    return row

@app.get("/benchmarks/{run_id}/results")
def benchmark_results(run_id: str):
    return list_results(run_id)

@app.post("/data/upload")
async def upload_data(files: list[UploadFile] = File(...)):
    raw_dir=Path("/app/data/raw"); raw_dir.mkdir(parents=True, exist_ok=True)
    saved=[]
    for f in files:
        dest=raw_dir/f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(str(dest))
    return {"saved":saved}

@app.post("/data/ground-truth")
async def upload_ground_truth(file: UploadFile = File(...)):
    eval_dir=Path("/app/data/eval"); eval_dir.mkdir(parents=True, exist_ok=True)
    dest=eval_dir/"qa.json"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"saved":str(dest)}
