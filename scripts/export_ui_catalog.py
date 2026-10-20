"""Refresh the shared React test fixture after changing component metadata."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.api.app.config_catalog import get_config_schema

if __name__ == "__main__":
    target = ROOT / "services/ui/src/test/catalog.json"
    target.write_text(
        json.dumps(get_config_schema(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Updated {target.relative_to(ROOT)}")
