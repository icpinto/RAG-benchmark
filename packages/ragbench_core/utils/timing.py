from contextlib import contextmanager
import time

@contextmanager
def timer():
    start = time.perf_counter()
    box = {"seconds": 0.0}
    try:
        yield box
    finally:
        box["seconds"] = time.perf_counter() - start
