from __future__ import annotations

import timeit
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_test_helpers():
    path = ROOT / "tests" / "test_distance_optimization.py"
    spec = importlib.util.spec_from_file_location("distance_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load_test_helpers()
ConcreteDistance = helpers.ConcreteDistance
chain = helpers.chain
legacy_one2one_all = helpers.legacy_one2one_all
residue = helpers.residue


def make_chain(chain_id, offset, residues=30, atoms=5):
    built = []
    for res_id in range(1, residues + 1):
        coords = [
            (f"C{atom_id}", (float(res_id + offset), float(atom_id), float(atom_id % 3)))
            for atom_id in range(atoms)
        ]
        built.append(residue("ALA", res_id, coords))
    return chain(chain_id, built)


def bench(name, func, number=3):
    elapsed = timeit.timeit(func, number=number) / number
    print(f"{name}: {elapsed:.6f}s")


if __name__ == "__main__":
    chain1 = make_chain("A", 0)
    chain2 = make_chain("B", 5)
    dist = ConcreteDistance()
    assert dist.one2one_all(chain1, chain2) == legacy_one2one_all(chain1, chain2)

    bench("legacy one2one_all", lambda: legacy_one2one_all(chain1, chain2), number=1)
    bench("optimized one2one_all", lambda: dist.one2one_all(chain1, chain2), number=3)
