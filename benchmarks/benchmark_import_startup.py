from __future__ import annotations

import subprocess
import sys
import time


def run_python(statement):
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-c", statement],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=20,
    )
    result.check_returncode()
    return time.perf_counter() - start


def bench(name, statement):
    elapsed = run_python(statement)
    print(f"{name}: {elapsed:.6f}s")


if __name__ == "__main__":
    bench("import pypropel", "import pypropel")
    bench("import pypropel then msa", "import pypropel; pypropel.msa")
