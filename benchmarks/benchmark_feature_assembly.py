from __future__ import annotations

import timeit

import numpy as np
import pandas as pd

from pypropel.prot.feature.PSSM import PSSM
from pypropel.prot.feature.rsa.Assemble import Assemble as RSAAssemble
from pypropel.prot.feature.ss.Assemble import Assemble as SSAssemble


def make_pair_windows(rows=1000, width=5):
    windows = []
    for i in range(rows):
        left = [None if j == 0 else ((i + j) % rows) + 1 for j in range(width)]
        right = [((i + j + width) % rows) + 1 for j in range(width)]
        windows.append((left, right))
    return windows


def legacy_psipred(df_psipred_ss2, list_2d, window_aa_ids):
    psipred = df_psipred_ss2.values.tolist()
    window_aa_ids_ = [i[0] + i[1] for i in window_aa_ids]
    list_2d_ = list_2d
    for i, aa_win_ids in enumerate(window_aa_ids_):
        for j in aa_win_ids:
            if j is None:
                list_2d_[i] = list_2d_[i] + np.zeros(3).tolist()
            else:
                list_2d_[i] = list_2d_[i] + psipred[j-1][3:]
    return list_2d_


def legacy_solvpred(df_solvpred, list_2d, window_aa_ids):
    solvpred = df_solvpred.values.tolist()
    window_aa_ids_ = [i[0] + i[1] for i in window_aa_ids]
    list_2d_ = list_2d
    for i, aa_win_ids in enumerate(window_aa_ids_):
        for j in aa_win_ids:
            if j is None:
                list_2d_[i].append(0)
            else:
                list_2d_[i].append(solvpred[j-1][2])
    return list_2d_


def legacy_blast(pssm, list_2d, window_aa_ids):
    list_2d_ = list_2d
    for i, aa_win_ids in enumerate(window_aa_ids):
        for j in aa_win_ids:
            for k in j:
                if k is None:
                    list_2d_[i] = list_2d_[i] + [0 for i in range(20)]
                else:
                    list_2d_[i] = list_2d_[i] + pssm[k]
    return list_2d_


def bench(name, func, number=3):
    elapsed = timeit.timeit(func, number=number) / number
    print(f"{name}: {elapsed:.6f}s")


if __name__ == "__main__":
    windows = make_pair_windows()
    base = [[i] for i in range(len(windows))]
    psipred = pd.DataFrame([[i, "A", "H", 0.1, 0.2, 0.7] for i in range(len(windows))])
    solvpred = pd.DataFrame([[i, "A", 0.5] for i in range(len(windows))])
    pssm = {i + 1: [float(i % 20)] * 20 for i in range(len(windows))}
    nested_windows = [[left, right] for left, right in windows]

    assert SSAssemble().psipred(psipred, [row[:] for row in base], windows) == legacy_psipred(psipred, [row[:] for row in base], windows)
    assert RSAAssemble().solvpred(solvpred, [row[:] for row in base], windows, mode="pair") == legacy_solvpred(solvpred, [row[:] for row in base], windows)
    assert PSSM().blast_(pssm, [row[:] for row in base], nested_windows) == legacy_blast(pssm, [row[:] for row in base], nested_windows)

    bench("legacy psipred", lambda: legacy_psipred(psipred, [row[:] for row in base], windows), number=2)
    bench("optimized psipred", lambda: SSAssemble().psipred(psipred, [row[:] for row in base], windows), number=5)
    bench("legacy solvpred", lambda: legacy_solvpred(solvpred, [row[:] for row in base], windows), number=2)
    bench("optimized solvpred", lambda: RSAAssemble().solvpred(solvpred, [row[:] for row in base], windows, mode="pair"), number=5)
    bench("legacy blast", lambda: legacy_blast(pssm, [row[:] for row in base], nested_windows), number=2)
    bench("optimized blast", lambda: PSSM().blast_(pssm, [row[:] for row in base], nested_windows), number=5)
