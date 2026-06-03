from ragbench_core.utils.config import load_yaml
from ragbench_core.engine.config_expander import expand_cases

def test_expand_cases():
    config=load_yaml("configs/default.yaml")
    cases=expand_cases(config)
    assert cases
    assert len(cases) <= config["runtime"]["max_combinations"]
