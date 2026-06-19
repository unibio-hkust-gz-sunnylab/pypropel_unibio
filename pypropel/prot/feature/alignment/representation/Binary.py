__author__ = "Jianfeng Sun"
__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"
__email__ = "jianfeng.sunmt@gmail.com"
__maintainer__ = "Jianfeng Sun"

import numpy as np


class Binary:

    def __init__(self, msa):
        self.msa = msa
        self.msa_row = len(self.msa)
        self.msa_col = len(self.msa[0])

    def onehot(self):
        encoded = np.frombuffer(''.join(self.msa).encode('ascii'), dtype=np.uint8)
        encoded = encoded.reshape(self.msa_row, self.msa_col)
        lut = np.full(256, -1, dtype=np.int16)
        for i, aa in enumerate(b'ACDEFGHIKLMNPQRSTVWY-'):
            lut[aa] = i
        codes = lut[encoded]
        binary_matrix = np.zeros((self.msa_row, self.msa_col, 21), dtype=int)
        valid = codes >= 0
        row_ids, col_ids = np.nonzero(valid)
        binary_matrix[row_ids, col_ids, codes[valid]] = 1
        return binary_matrix.reshape(self.msa_row, self.msa_col * 21)
