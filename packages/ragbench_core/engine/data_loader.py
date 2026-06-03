from __future__ import annotations
import json
from pathlib import Path
from ragbench_core.models.schema import QueryExample

def list_sources(input_dir: str, remote_urls_file: str | None = None) -> list[str]:
    sources=[]
    p=Path(input_dir)
    if p.exists():
        for f in p.rglob("*"):
            if f.is_file() and f.name != "urls.txt": sources.append(str(f))
    if remote_urls_file and Path(remote_urls_file).exists():
        urls=[u.strip() for u in Path(remote_urls_file).read_text().splitlines() if u.strip() and not u.strip().startswith("#")]
        sources.extend(urls)
    return sources

def load_ground_truth(path: str) -> list[QueryExample]:
    raw=json.loads(Path(path).read_text())
    return [QueryExample(**x) for x in raw]
