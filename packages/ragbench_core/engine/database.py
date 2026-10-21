from __future__ import annotations
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Text, text, inspect
from copy import deepcopy
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
from ragbench_core.settings import settings

Base=declarative_base()
engine=create_engine(settings.postgres_url, pool_pre_ping=True)
SessionLocal=sessionmaker(bind=engine)

class BenchmarkRun(Base):
    __tablename__="benchmark_runs"
    id=Column(String, primary_key=True)
    name=Column(String)
    status=Column(String)
    config=Column(JSON)
    created_at=Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at=Column(DateTime, default=lambda: datetime.now(timezone.utc))
    summary=Column(JSON, default=dict)
    dataset_hash=Column(String, index=True, nullable=True)
    config_hash=Column(String, index=True, nullable=True)
    component_versions=Column(JSON, default=dict)
    dependency_versions=Column(JSON, default=dict)
    model_identifiers=Column(JSON, default=dict)
    artifact_dir=Column(Text, nullable=True)
    exports=Column(JSON, default=dict)
    error=Column(Text, nullable=True)

class BenchmarkResult(Base):
    __tablename__="benchmark_results"
    id=Column(String, primary_key=True)
    run_id=Column(String, index=True)
    case_id=Column(String, index=True)
    question=Column(Text)
    question_id=Column(String, nullable=True)
    metrics=Column(JSON)
    answer=Column(Text)
    case_config=Column(JSON)
    artifact_paths=Column(JSON, default=dict)
    model_identifiers=Column(JSON, default=dict)
    created_at=Column(DateTime, default=lambda: datetime.now(timezone.utc))

def init_db():
    Base.metadata.create_all(engine)
    _ensure_metadata_columns()

def _ensure_metadata_columns():
    # Lightweight compatibility for existing development databases created before
    # persistence metadata columns were added. Fresh databases are handled by SQLAlchemy.
    statements=[
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS dataset_hash VARCHAR",
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS config_hash VARCHAR",
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS component_versions JSON",
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS dependency_versions JSON",
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS model_identifiers JSON",
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS artifact_dir TEXT",
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS exports JSON",
        "ALTER TABLE benchmark_results ADD COLUMN IF NOT EXISTS artifact_paths JSON",
        "ALTER TABLE benchmark_results ADD COLUMN IF NOT EXISTS model_identifiers JSON",
    ]
    statements.append("ALTER TABLE benchmark_results ADD COLUMN IF NOT EXISTS question_id VARCHAR")
    with engine.begin() as conn:
        inspector = inspect(conn)
        existing = {name: {c["name"] for c in inspector.get_columns(name)} for name in ("benchmark_runs", "benchmark_results")}
        for statement in statements:
            words = statement.split()
            table, column = words[2], words[8]
            if column not in existing[table]:
                conn.execute(text(statement if engine.dialect.name == "postgresql" else statement.replace(" IF NOT EXISTS", "")))
                existing[table].add(column)

def upsert_run(run_id: str, name: str, status: str, config: dict, summary: dict | None = None, error: str | None = None, **metadata):
    init_db()
    db=SessionLocal()
    try:
        row=db.get(BenchmarkRun, run_id)
        if not row:
            row=BenchmarkRun(id=run_id, name=name, status=status, config=config, summary=summary or {}, error=error, **metadata)
            db.add(row)
        else:
            row.status=status; row.updated_at=datetime.now(timezone.utc); row.summary=summary or row.summary; row.error=error
            for key,value in metadata.items():
                if hasattr(row, key):
                    setattr(row, key, value)
        db.commit()
    finally:
        db.close()

def insert_result(result_id: str, run_id: str, case_id: str, question: str, metrics: dict, answer: str, case_config: dict, artifact_paths: dict | None = None, model_identifiers: dict | None = None, question_id: str | None = None):
    db=SessionLocal()
    try:
        db.add(BenchmarkResult(id=result_id, run_id=run_id, case_id=case_id, question=question, question_id=question_id, metrics=metrics, answer=answer, case_config=case_config, artifact_paths=artifact_paths or {}, model_identifiers=model_identifiers or {}))
        db.commit()
    finally:
        db.close()

def get_run(run_id: str):
    init_db(); db=SessionLocal()
    try:
        r=db.get(BenchmarkRun, run_id)
        if not r: return None
        return {"id":r.id,"name":r.name,"status":r.status,"summary":r.summary,"error":r.error,"created_at":str(r.created_at),"updated_at":str(r.updated_at),"dataset_hash":r.dataset_hash,"config_hash":r.config_hash,"component_versions":r.component_versions,"dependency_versions":r.dependency_versions,"model_identifiers":r.model_identifiers,"artifact_dir":r.artifact_dir,"exports":r.exports,"question_ids":(r.config or {}).get("_snapshot", {}).get("question_ids", [])}
    finally: db.close()

def get_run_config(run_id: str) -> dict:
    with SessionLocal() as db:
        row = db.get(BenchmarkRun, run_id)
        if row is None:
            raise ValueError(f"Run not found: {run_id}")
        return deepcopy(row.config)


def list_runs(limit: int = 20):
    init_db(); db=SessionLocal()
    try:
        rows=db.query(BenchmarkRun).order_by(BenchmarkRun.created_at.desc()).limit(limit).all()
        return [{"id":r.id,"name":r.name,"status":r.status,"summary":r.summary,"error":r.error,"created_at":str(r.created_at),"dataset_hash":r.dataset_hash,"config_hash":r.config_hash,"model_identifiers":r.model_identifiers} for r in rows]
    finally: db.close()

def list_results(run_id: str):
    init_db(); db=SessionLocal()
    try:
        rows=db.query(BenchmarkResult).filter(BenchmarkResult.run_id==run_id).all()
        return [{"case_id":r.case_id,"question_id":r.question_id,"question":r.question,"metrics":r.metrics,"answer":r.answer,"case_config":r.case_config,"artifact_paths":r.artifact_paths,"model_identifiers":r.model_identifiers} for r in rows]
    finally: db.close()
