__author__ = "Jianfeng Sun"
__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"
__email__ = "jianfeng.sunmt@gmail.com"
__maintainer__ = "Jianfeng Sun"

import numpy as np
import pandas as pd
from pypropel.prot.sequence.Symbol import Symbol


class Single:

    def __init__(
            self,
            msa,
    ):
        self.msa = msa
        self.aa_alphabet = Symbol().single(gap=True, universal=False)
        self.msa_row = len(self.msa)
        self.msa_col = len(self.msa[0])

    def _encoded(self, ):
        arr = np.frombuffer(''.join(self.msa).encode('ascii'), dtype=np.uint8)
        return arr.reshape(self.msa_row, self.msa_col)

    def _alphabet_codes(self, ):
        lut = np.full(256, -1, dtype=np.int16)
        for i, aa in enumerate(self.aa_alphabet):
            lut[ord(aa)] = i
        return lut[self._encoded()]

    def columns(self, ):
        """
        Frequency of 20 amino acids and 1 gap in each column of msa.

        qi represents one of 20 amino acids and 1 gap in a column of msa.
           sum_t(qi) is total amount of qi in column t of msa. The total
           amount of rows in the msa is sum(rows). Frequencies matrix is
           calculated by sum(qi)/sum(rows).

        Examples
        --------
           1atzA: QPLDVILLLDGSSSFPASYFDEMKSFAKAFISKANIGPRLTQVSVLQYGSITTIDVPWNVVPEKAHLLSLVDVMQ
           return 75x21 matrix.

        Returns
        -------
        2d array : numpy.ndarray
            row: 75; col: 21

        """
        codes = self._alphabet_codes()
        valid = codes >= 0
        col_offsets = np.arange(self.msa_col, dtype=np.int64) * len(self.aa_alphabet)
        flat_ids = (codes + col_offsets).ravel()
        counts = np.bincount(
            flat_ids[valid.ravel()],
            minlength=self.msa_col * len(self.aa_alphabet),
        )
        count_array = counts.reshape(self.msa_col, len(self.aa_alphabet)).T
        freq_array = count_array / self.msa_row
        freq = {}
        for i, aa in enumerate(self.aa_alphabet):
            freq[aa] = freq_array[i]
        return freq, freq['-']

    def columnsByPandas(self, ):
        msa_sp = []
        for homolog in self.msa:
            msa_sp.append(list(homolog))
        msa_sp_df = pd.DataFrame(msa_sp)
        freq = {}
        for aa in self.aa_alphabet:
            freq[aa] = []
        for alignment_pos in msa_sp_df.columns:
            base_count = msa_sp_df[alignment_pos].value_counts().to_dict()
            for i, aa in enumerate(self.aa_alphabet):
                if aa in base_count.keys():
                    # freq[aa].append(base_count[aa] / self.msa_row)
                    freq[aa].append(base_count[aa])
                else:
                    freq[aa].append(0)
        return freq

    def alignment(self, ):
        """
        Frequency of 20 amino acids and 1 gap in a whole msa.

        qi represents one of 21 amino acids. sum(qi) is total amount of qi in a whole msa.
           The total amount of all amino acids in a whole msa is sum(qi). Frequencies are
           calculated by sum(qi)/sum(all).

        Examples
        --------
        1atzA: QPLDVILLLDGSSSFPASYFDEMKSFAKAFISKANIGPRLTQVSVLQYGSITTIDVPWNVVPEKAHLLSLVDVMQ
           return 1x21 matrix.

        Returns
        -------
            1d array: numpy.ndarray
             row: 1; col: 21
        """
        total_num_MSA = self.msa_row * self.msa_col
        codes = self._alphabet_codes()
        counts = np.bincount(
            codes[codes >= 0],
            minlength=len(self.aa_alphabet),
        )
        freq_array = counts / total_num_MSA
        freq = {}
        for i, aa in enumerate(self.aa_alphabet):
            freq[aa] = freq_array[i]
        return freq
