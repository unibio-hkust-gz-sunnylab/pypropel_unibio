from __future__ import annotations

import timeit
import sys
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pypropel.prot.feature.alignment.frequency.Single import Single


def load_test_helpers():
    path = ROOT / "tests" / "test_alignment_frequency_optimization.py"
    spec = importlib.util.spec_from_file_location("alignment_frequency_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load_test_helpers()
legacy_alignment = helpers.legacy_alignment
legacy_columns = helpers.legacy_columns


def make_msa(rows=1500, cols=700):
    rng = np.random.default_rng(42)
    alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY-"))
    return ["".join(row.tolist()) for row in rng.choice(alphabet, size=(rows, cols))]


def bench(name, func, number=3):
    elapsed = timeit.timeit(func, number=number) / number
    print(f"{name}: {elapsed:.6f}s")


if __name__ == "__main__":
    msa = make_msa()
    freq, omit = Single(msa).columns()
    legacy_freq, legacy_omit = legacy_columns(msa)
    for aa in legacy_freq:
        np.testing.assert_allclose(freq[aa], legacy_freq[aa])
    np.testing.assert_allclose(omit, legacy_omit)
    assert Single(msa).alignment() == legacy_alignment(msa)

    bench("legacy columns", lambda: legacy_columns(msa), number=1)
    bench("optimized columns", lambda: Single(msa).columns(), number=5)
    bench("legacy alignment", lambda: legacy_alignment(msa), number=1)
    bench("optimized alignment", lambda: Single(msa).alignment(), number=5)
