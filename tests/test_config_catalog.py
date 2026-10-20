import json
from pathlib import Path

from ragbench_core.registry import register_builtin_components
from services.api.app.config_catalog import get_config_schema


def test_catalog_covers_all_registered_builtin_pipeline_components():
    registry = register_builtin_components()
    catalog = get_config_schema()
    definitions = {
        stage["key"]: {item["name"] for item in stage["components"]}
        for stage in catalog["stages"]
    }
    for family in registry._REGISTER_METHOD_GROUPS.values():
        if family == "metrics":
            assert {item["name"] for item in catalog["metric_components"]} == set(
                registry.metrics
            )
            continue
        names = (
            definitions["dense_indexer"] | definitions["sparse_indexer"]
            if family == "indexers"
            else definitions[family]
        )
        assert names == set(getattr(registry, family)), family


def test_catalog_matches_frontend_contract_fixture():
    fixture = Path("services/ui/src/test/catalog.json")
    assert get_config_schema() == json.loads(fixture.read_text(encoding="utf-8"))


def test_catalog_fields_are_unique_and_only_engine_expanded_parameters_are_sweepable():
    catalog = get_config_schema()

    def check(fields):
        keys = [item["key"] for item in fields]
        assert len(keys) == len(set(keys)), keys
        for item in fields:
            if item.get("fields"):
                check(item["fields"])

    for section in catalog["sections"]:
        check(section["fields"])
    for stage in catalog["stages"]:
        for component in stage["components"]:
            check(component["fields"])
            if stage["key"] not in {"chunkers", "dense_indexer"}:
                assert not any(item.get("sweepable") for item in component["fields"])


def test_catalog_is_fresh_and_does_not_resolve_environment_secrets(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "never-return-this-secret")
    first = get_config_schema()
    assert "never-return-this-secret" not in json.dumps(first)
    first["stages"].clear()
    assert get_config_schema()["stages"]


def test_schema_endpoint_is_read_only_without_provider_or_database_calls(monkeypatch):
    from fastapi.testclient import TestClient
    from services.api.app import main

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Schema endpoint must not load configurations or initialize the database"
        )

    monkeypatch.setattr(main, "load_yaml", forbidden)
    monkeypatch.setattr(main, "init_db", forbidden)
    response = TestClient(main.app).get("/configs/schema")
    assert response.status_code == 200
    assert response.json() == get_config_schema()
