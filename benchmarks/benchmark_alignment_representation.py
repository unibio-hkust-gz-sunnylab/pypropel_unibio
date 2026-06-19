from __future__ import annotations

import timeit
import sys
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pypropel.prot.feature.alignment.representation.Binary import Binary
from pypropel.prot.feature.alignment.representation.Frequency import Frequency


def load_test_helpers():
    path = ROOT / "tests" / "test_alignment_representation_optimization.py"
    spec = importlib.util.spec_from_file_location("alignment_representation_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load_test_helpers()
legacy_matrix = helpers.legacy_matrix
legacy_onehot = helpers.legacy_onehot


def make_msa(rows=180, cols=80):
    rng = np.random.default_rng(7)
    alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY-"))
    return ["".join(row.tolist()) for row in rng.choice(alphabet, size=(rows, cols))]


def bench(name, func, number=3):
    elapsed = timeit.timeit(func, number=number) / number
    print(f"{name}: {elapsed:.6f}s")


if __name__ == "__main__":
    msa = make_msa()
    np.testing.assert_array_equal(Binary(msa).onehot(), legacy_onehot(msa))
    np.testing.assert_allclose(Frequency(msa).matrix(), legacy_matrix(msa))

    bench("legacy onehot", lambda: legacy_onehot(msa), number=1)
    bench("optimized onehot", lambda: Binary(msa).onehot(), number=5)
    bench("legacy frequency matrix", lambda: legacy_matrix(msa), number=1)
    bench("optimized frequency matrix", lambda: Frequency(msa).matrix(), number=3)
