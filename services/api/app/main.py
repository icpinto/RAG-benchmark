from __future__ import annotations
from pathlib import Path
import json, os, shutil, uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, Field
from ragbench_core.engine.database import init_db, get_run, list_runs, list_results, upsert_run
from ragbench_core.utils.config import load_yaml, parse_yaml_text
from services.worker.app.tasks import run_benchmark_task

app=FastAPI(title="RAG Benchmark Platform API", version="0.1.0")
CONFIG_ROOT=Path(os.getenv("RAGBENCH_CONFIG_DIR", "/app/configs")).resolve()
DATA_ROOT=Path(os.getenv("RAGBENCH_DATA_DIR", "/app/data")).resolve()
RAW_DATA_DIR=(DATA_ROOT / "raw").resolve()
EVAL_DATA_DIR=(DATA_ROOT / "eval").resolve()
GROUND_TRUTH_PATH=(EVAL_DATA_DIR / "qa.json").resolve()

class RunRequest(BaseModel):
    config_path: str = "/app/configs/default.yaml"

class ConfigSaveRequest(BaseModel):
    config_path: str = "/app/configs/default.yaml"
    content: str

class QuestionRequest(BaseModel):
    question: str
    expected_answer: str = ""
    ground_truth_doc_ids: list[str] = Field(default_factory=list)
    ground_truth_chunk_ids: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)



def _resolve_data_path(path: str, root: Path) -> Path:
    raw=Path(path)
    candidate=(raw if raw.is_absolute() else root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Path must be inside {root}")
    return candidate

def _read_ground_truth() -> list[dict]:
    if not GROUND_TRUTH_PATH.exists():
        return []
    try:
        payload=json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid ground-truth JSON: {exc}")
    if not isinstance(payload, list):
        raise HTTPException(status_code=500, detail="Ground-truth JSON must contain a list of questions")
    return payload

def _write_ground_truth(questions: list[dict]) -> None:
    EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_PATH.write_text(json.dumps(questions, indent=2) + "\n", encoding="utf-8")

def _question_payload(req: QuestionRequest) -> dict:
    if hasattr(req, "model_dump"):
        return req.model_dump()
    return req.dict()

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
    raw_dir=RAW_DATA_DIR; raw_dir.mkdir(parents=True, exist_ok=True)
    saved=[]
    for f in files:
        dest=raw_dir/f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(str(dest))
    return {"saved":saved}

@app.post("/data/ground-truth")
async def upload_ground_truth(file: UploadFile = File(...)):
    EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest=GROUND_TRUTH_PATH
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"saved":str(dest)}


@app.get("/data/files")
def list_data_files():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    files=[]
    for path in sorted(RAW_DATA_DIR.rglob("*")):
        if path.is_file():
            stat=path.stat()
            files.append({
                "name":path.name,
                "path":str(path.relative_to(RAW_DATA_DIR)),
                "size_bytes":stat.st_size,
                "modified_at":stat.st_mtime,
            })
    return {"data_root":str(RAW_DATA_DIR),"files":files}

@app.get("/data/files/content")
def get_data_file_content(path: str = Query(...)):
    data_path=_resolve_data_path(path, RAW_DATA_DIR)
    if not data_path.exists() or not data_path.is_file():
        raise HTTPException(status_code=404, detail=f"Data file not found: {data_path}")
    if data_path.stat().st_size > 1_000_000:
        raise HTTPException(status_code=413, detail="File is too large to preview in the UI")
    try:
        content=data_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=415, detail="Only UTF-8 text files can be previewed")
    return {"path":str(data_path.relative_to(RAW_DATA_DIR)),"content":content}

@app.get("/data/questions")
def list_questions():
    return {"path":str(GROUND_TRUTH_PATH),"questions":_read_ground_truth()}

@app.post("/data/questions")
def create_question(req: QuestionRequest):
    questions=_read_ground_truth()
    questions.append(_question_payload(req))
    _write_ground_truth(questions)
    return {"index":len(questions)-1,"question":questions[-1],"status":"created"}

@app.put("/data/questions/{index}")
def update_question(index: int, req: QuestionRequest):
    questions=_read_ground_truth()
    if index < 0 or index >= len(questions):
        raise HTTPException(status_code=404, detail="Question not found")
    questions[index]=_question_payload(req)
    _write_ground_truth(questions)
    return {"index":index,"question":questions[index],"status":"updated"}

@app.delete("/data/questions/{index}")
def delete_question(index: int):
    questions=_read_ground_truth()
    if index < 0 or index >= len(questions):
        raise HTTPException(status_code=404, detail="Question not found")
    removed=questions.pop(index)
    _write_ground_truth(questions)
    return {"index":index,"question":removed,"status":"deleted"}
