from __future__ import annotations

import unittest

import numpy as np

from pypropel.prot.feature.alignment.frequency.Single import Single
from pypropel.prot.sequence.Symbol import Symbol


def legacy_columns(msa):
    aa_alphabet = Symbol().single(gap=True, universal=False)
    msa_row = len(msa)
    msa_col = len(msa[0])
    counts = {aa: [0] * msa_col for aa in aa_alphabet}
    for homolog in msa:
        for alignment_pos, base in enumerate(homolog):
            if base in counts:
                counts[base][alignment_pos] += 1
    freq = {aa: np.array(counts[aa]) / msa_row for aa in aa_alphabet}
    return freq, np.array(counts['-']) / msa_row


def legacy_alignment(msa):
    aa_alphabet = Symbol().single(gap=True, universal=False)
    total_num_msa = len(msa) * len(msa[0])
    counts = {aa: 0 for aa in aa_alphabet}
    for row in msa:
        for base in row:
            if base in counts:
                counts[base] += 1
    freq_array = np.array([counts[aa] for aa in aa_alphabet]) / total_num_msa
    return {aa: freq_array[i] for i, aa in enumerate(aa_alphabet)}


class AlignmentFrequencyOptimizationTest(unittest.TestCase):
    def test_columns_matches_legacy_counts(self):
        msa = [
            "ACD-EFX",
            "A-DGEFY",
            "WCD-EF-",
            "------Z",
        ]

        expected_freq, expected_omit = legacy_columns(msa)
        actual_freq, actual_omit = Single(msa).columns()

        self.assertEqual(actual_freq.keys(), expected_freq.keys())
        for aa in expected_freq:
            np.testing.assert_allclose(actual_freq[aa], expected_freq[aa])
        np.testing.assert_allclose(actual_omit, expected_omit)

    def test_alignment_matches_legacy_counts(self):
        msa = [
            "ACD-EFX",
            "A-DGEFY",
            "WCD-EF-",
            "------Z",
        ]

        expected = legacy_alignment(msa)
        actual = Single(msa).alignment()

        self.assertEqual(actual.keys(), expected.keys())
        for aa in expected:
            self.assertEqual(actual[aa], expected[aa])


if __name__ == "__main__":
    unittest.main()
