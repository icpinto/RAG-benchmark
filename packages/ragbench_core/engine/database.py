from __future__ import annotations
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Float, Text
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
    error=Column(Text, nullable=True)

class BenchmarkResult(Base):
    __tablename__="benchmark_results"
    id=Column(String, primary_key=True)
    run_id=Column(String, index=True)
    case_id=Column(String, index=True)
    question=Column(Text)
    metrics=Column(JSON)
    answer=Column(Text)
    case_config=Column(JSON)
    created_at=Column(DateTime, default=lambda: datetime.now(timezone.utc))

def init_db():
    Base.metadata.create_all(engine)

def upsert_run(run_id: str, name: str, status: str, config: dict, summary: dict | None = None, error: str | None = None):
    init_db()
    db=SessionLocal()
    try:
        row=db.get(BenchmarkRun, run_id)
        if not row:
            row=BenchmarkRun(id=run_id, name=name, status=status, config=config, summary=summary or {}, error=error)
            db.add(row)
        else:
            row.status=status; row.updated_at=datetime.now(timezone.utc); row.summary=summary or row.summary; row.error=error
        db.commit()
    finally:
        db.close()

def insert_result(result_id: str, run_id: str, case_id: str, question: str, metrics: dict, answer: str, case_config: dict):
    db=SessionLocal()
    try:
        db.add(BenchmarkResult(id=result_id, run_id=run_id, case_id=case_id, question=question, metrics=metrics, answer=answer, case_config=case_config))
        db.commit()
    finally:
        db.close()

def get_run(run_id: str):
    init_db(); db=SessionLocal()
    try:
        r=db.get(BenchmarkRun, run_id)
        if not r: return None
        return {"id":r.id,"name":r.name,"status":r.status,"summary":r.summary,"error":r.error,"created_at":str(r.created_at),"updated_at":str(r.updated_at)}
    finally: db.close()

def list_runs(limit: int = 20):
    init_db(); db=SessionLocal()
    try:
        rows=db.query(BenchmarkRun).order_by(BenchmarkRun.created_at.desc()).limit(limit).all()
        return [{"id":r.id,"name":r.name,"status":r.status,"summary":r.summary,"error":r.error,"created_at":str(r.created_at)} for r in rows]
    finally: db.close()

def list_results(run_id: str):
    init_db(); db=SessionLocal()
    try:
        rows=db.query(BenchmarkResult).filter(BenchmarkResult.run_id==run_id).all()
        return [{"case_id":r.case_id,"question":r.question,"metrics":r.metrics,"answer":r.answer,"case_config":r.case_config} for r in rows]
    finally: db.close()
