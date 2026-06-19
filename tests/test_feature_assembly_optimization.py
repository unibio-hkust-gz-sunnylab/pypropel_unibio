from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pypropel.prot.feature.PSSM import PSSM
from pypropel.prot.feature.rsa.Assemble import Assemble as RSAAssemble
from pypropel.prot.feature.ss.Assemble import Assemble as SSAssemble


class FeatureAssemblyOptimizationTest(unittest.TestCase):
    def test_sspro8_matches_legacy_encoding(self):
        df = pd.DataFrame([["H"], ["E"], ["S"]])
        base = [[0], [1]]
        window_ids = [([None, 1], [2]), ([3], [None])]

        actual = SSAssemble().sspro8(df, [row[:] for row in base], window_ids)

        expected = [
            [0] + [0.0] * 8 + [0, 0, 0, 0, 0, 0, 0, 1] + [0, 0, 0, 0, 1, 0, 0, 0],
            [1] + [0, 1, 0, 0, 0, 0, 0, 0] + [0.0] * 8,
        ]
        self.assertEqual(actual, expected)

    def test_psipred_matches_legacy_window_append(self):
        df = pd.DataFrame([
            [1, "A", "H", 0.1, 0.2, 0.7],
            [2, "C", "E", 0.3, 0.4, 0.3],
        ])
        actual = SSAssemble().psipred(df, [[0], [1]], [([1], [None]), ([2], [1])])
        expected = [
            [0, 0.1, 0.2, 0.7, 0.0, 0.0, 0.0],
            [1, 0.3, 0.4, 0.3, 0.1, 0.2, 0.7],
        ]
        self.assertEqual(actual, expected)

    def test_rsa_accpro20_preserves_current_mapping(self):
        df = pd.DataFrame([[0.0], [0.2], [0.205], [0.9], [1.0]])
        actual = RSAAssemble().accpro20(df, [[0], [1]], [[1, 2, None], [3, 4, 5]], mode="single")

        def onehot(active_idx):
            return [1 if idx == active_idx else 0 for idx in range(20)]

        expected = [
            [0] + onehot(19) + onehot(15) + [0.0] * 20,
            [1] + onehot(2) + onehot(1) + onehot(0),
        ]
        self.assertEqual(actual, expected)

    def test_pssm_blast_and_hhm_match_legacy_window_append(self):
        pssm = {1: [1] * 20, 2: [2] * 20}
        hhm = {1: np.ones(30), 2: np.ones(30) * 2}
        window_ids = [[[None, 1], [2]], [[1]]]

        actual_blast = PSSM().blast_(pssm, [[0], [1]], window_ids)
        expected_blast = [
            [0] + [0] * 20 + [1] * 20 + [2] * 20,
            [1] + [1] * 20,
        ]
        self.assertEqual(actual_blast, expected_blast)

        actual_hhm = PSSM().hhm_(hhm, [[0], [1]], window_ids)
        expected_hhm = [
            [0] + [0] * 30 + [1.0] * 30 + [2.0] * 30,
            [1] + [1.0] * 30,
        ]
        self.assertEqual(actual_hhm, expected_hhm)


if __name__ == "__main__":
    unittest.main()
