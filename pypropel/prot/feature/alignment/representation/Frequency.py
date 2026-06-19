__author__ = "Jianfeng Sun"
__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"
__email__ = "jianfeng.sunmt@gmail.com"
__maintainer__ = "Jianfeng Sun"

import numpy as np
from pypropel.prot.feature.alignment.symbol.Single import Single as sglalignsymbol


class Frequency:

    def __init__(self, msa):
        self.msa = msa
        self.msa_col = len(self.msa[0])
        self.passingle = sglalignsymbol(self.msa)

    def calc_sgl_col(self, x):
        bases = np.array(self.passingle.extract(x))
        num_total = bases.shape[0]
        values, inverse, counts = np.unique(
            bases,
            return_inverse=True,
            return_counts=True,
        )
        freq_single = np.round(counts[inverse] / num_total, 4)
        freq_single[values[inverse] == '-'] = 0
        return freq_single.tolist()

    def matrix(self ):
        msa_matrix = np.array([list(row) for row in self.msa])
        base_freq_matrix = np.zeros(msa_matrix.shape, dtype=float)
        for i in range(self.msa_col):
            values, inverse, counts = np.unique(
                msa_matrix[:, i],
                return_inverse=True,
                return_counts=True,
            )
            col_freq = np.round(counts[inverse] / msa_matrix.shape[0], 4)
            col_freq[values[inverse] == '-'] = 0
            base_freq_matrix[:, i] = col_freq
        return base_freq_matrix


if __name__ == "__main__":
    p = Frequency(
        msa=msa
    )
    # print(p.calc_sgl_col(1))
    print(p.matrix())
