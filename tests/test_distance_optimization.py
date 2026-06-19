from __future__ import annotations

import unittest

import numpy as np
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

from pypropel.prot.structure.distance.Distance import distance, three_to_one


class ConcreteDistance(distance):
    def calculate(self):
        return None


def atom(name, coord):
    return Atom(name, np.asarray(coord, dtype=float), 1.0, 1.0, " ", name.rjust(4), 1, element=name[0])


def residue(resname, seq_id, coords):
    res = Residue((" ", seq_id, " "), resname, "")
    for atom_name, coord in coords:
        res.add(atom(atom_name, coord))
    return res


def hetero_residue(resname, seq_id, coords):
    res = Residue(("H_TEST", seq_id, " "), resname, "")
    for atom_name, coord in coords:
        res.add(atom(atom_name, coord))
    return res


def chain(chain_id, residues):
    chn = Chain(chain_id)
    for res in residues:
        chn.add(res)
    return chn


def legacy_min_distance(residue_1, residue_2):
    tmp_atom_dist = []
    for atom_1 in residue_1:
        if atom_1.get_name() != "H":
            for atom_2 in residue_2:
                if atom_2.get_name() != "H":
                    tmp_atom_dist.append(residue_1[atom_1.get_name()] - residue_2[atom_2.get_name()])
    return min(tmp_atom_dist)


def legacy_one2one_all(chain1, chain2):
    dist_matrix = []
    count_hetamt_1 = 0
    count_hetamt_2 = 0
    for index_1, residue_1 in enumerate(chain1):
        if residue_1.get_id()[0] != " ":
            count_hetamt_1 = count_hetamt_1 + 1
            continue
        for index_2, residue_2 in enumerate(chain2):
            if residue_2.get_id()[0] != " ":
                count_hetamt_2 = count_hetamt_2 + 1
                continue
            min_dist = legacy_min_distance(residue_1, residue_2)
            dist_matrix.append([
                index_1 + 1 - count_hetamt_1,
                three_to_one[residue_1.get_resname()],
                residue_1.id[1],
                index_2 + 1 - count_hetamt_2,
                "U" if residue_2.get_resname() == "UNK" else three_to_one[residue_2.get_resname()],
                residue_2.id[1],
                min_dist,
            ])
    return dist_matrix


class DistanceOptimizationTest(unittest.TestCase):
    def setUp(self):
        self.chain1 = chain("A", [
            residue("ALA", 1, [("N", (0.0, 0.0, 0.0)), ("CA", (1.0, 0.0, 0.0)), ("H", (99.0, 0.0, 0.0))]),
            residue("GLY", 2, [("N", (5.0, 0.0, 0.0)), ("CA", (6.0, 0.0, 0.0))]),
        ])
        self.chain2 = chain("B", [
            residue("SER", 1, [("N", (0.0, 3.0, 0.0)), ("CA", (1.0, 3.0, 0.0))]),
            residue("VAL", 2, [("N", (8.0, 0.0, 0.0)), ("CA", (9.0, 0.0, 0.0))]),
        ])
        self.dist = ConcreteDistance()

    def test_one2one_all_matches_legacy(self):
        self.assertEqual(
            self.dist.one2one_all(self.chain1, self.chain2),
            legacy_one2one_all(self.chain1, self.chain2),
        )

    def test_one2one_all_preserves_legacy_hetero_indexing(self):
        chain2 = chain("B", [
            residue("SER", 1, [("N", (0.0, 3.0, 0.0))]),
            hetero_residue("SER", 2, [("N", (99.0, 99.0, 99.0))]),
            residue("VAL", 3, [("N", (8.0, 0.0, 0.0))]),
        ])

        self.assertEqual(
            self.dist.one2one_all(self.chain1, chain2),
            legacy_one2one_all(self.chain1, chain2),
        )

    def test_one2one_minimal_matches_legacy(self):
        all_distances = legacy_one2one_all(self.chain1, self.chain2)
        expected = [
            [1, "A", 1, min(row[6] for row in all_distances if row[0] == 1)],
            [2, "G", 2, min(row[6] for row in all_distances if row[0] == 2)],
        ]
        self.assertEqual(self.dist.one2one_minimal(self.chain1, self.chain2), expected)

    def test_check_matches_legacy_threshold_behavior(self):
        self.assertTrue(self.dist.check(self.chain1, self.chain2, thres=3.1))
        self.assertFalse(self.dist.check(self.chain1, self.chain2, thres=1.0))


if __name__ == "__main__":
    unittest.main()
