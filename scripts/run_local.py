from ragbench_core.utils.config import load_yaml
from ragbench_core.engine.runner import BenchmarkRunner
import uuid

if __name__ == "__main__":
    config=load_yaml("configs/default.yaml")
    print(BenchmarkRunner().run(str(uuid.uuid4()), config))
