from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from pypropel.prot.feature.alignment.representation.Binary import Binary
from pypropel.prot.feature.alignment.representation.Frequency import Frequency


AA_TO_INDEX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY-")}


def legacy_onehot(msa):
    msa_row = len(msa)
    msa_col = len(msa[0])
    binary_matrix = [[0 for _ in range(msa_col * 21)] for _ in range(msa_row)]
    for i in range(msa_row):
        for j in range(msa_col):
            if msa[i][j] in AA_TO_INDEX:
                binary_matrix[i][j * 21 + AA_TO_INDEX[msa[i][j]]] = 1
    return np.array(binary_matrix)


def legacy_calc_sgl_col(msa, x):
    bases = [row[x] for row in msa]
    num_total = len(bases)
    freq_all_bases = Counter(bases).most_common()
    freq_single = [0 for _ in range(num_total)]
    for i in range(len(freq_all_bases)):
        for j in range(num_total):
            if np.array_equal(freq_all_bases[i][0], bases[j]):
                freq_single[j] = round(freq_all_bases[i][1] / num_total, 4)
                if bases[j] == '-':
                    freq_single[j] = 0
    return freq_single


def legacy_matrix(msa):
    return np.transpose(np.array([
        legacy_calc_sgl_col(msa, i)
        for i in range(len(msa[0]))
    ]))


class AlignmentRepresentationOptimizationTest(unittest.TestCase):
    def test_onehot_matches_legacy(self):
        msa = [
            "ACD-EFX",
            "A-DGEFY",
            "WCD-EF-",
        ]

        np.testing.assert_array_equal(Binary(msa).onehot(), legacy_onehot(msa))

    def test_frequency_column_matches_legacy(self):
        msa = [
            "ACD-EFX",
            "A-DGEFY",
            "WCD-EF-",
        ]

        self.assertEqual(Frequency(msa).calc_sgl_col(2), legacy_calc_sgl_col(msa, 2))

    def test_frequency_matrix_matches_legacy(self):
        msa = [
            "ACD-EFX",
            "A-DGEFY",
            "WCD-EF-",
        ]

        np.testing.assert_allclose(Frequency(msa).matrix(), legacy_matrix(msa))


if __name__ == "__main__":
    unittest.main()
