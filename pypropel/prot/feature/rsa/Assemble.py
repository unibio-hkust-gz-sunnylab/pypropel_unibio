__author__ = "Jianfeng Sun"
__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"
__email__ = "jianfeng.sunmt@gmail.com"
__maintainer__ = "Jianfeng Sun"

import time
import numpy as np
from pypropel.util.Console import Console


class Assemble:

    def __init__(
            self,
            verbose: bool = True,

    ):
        self.console = Console()
        self.console.verbose = verbose

    def accpro(
            self,
            df_accpro,
            list_2d,
            window_aa_ids,
            mode='single',
    ):
        start_time = time.time()
        accpro = df_accpro.values.tolist()
        print(df_accpro)
        window_aa_ids_ = window_aa_ids if mode == 'single' else [i[0] + i[1] for i in window_aa_ids]
        list_2d_ = list_2d
        zero = np.zeros(2).tolist()
        for i, aa_win_ids in enumerate(window_aa_ids_):
            # print(i)
            # print(aa_win_ids)
            row_features = []
            for j in aa_win_ids:
                # print(aa_win_ids)
                if j is None:
                    row_features.extend(zero)
                else:
                    if accpro[j - 1][0] == 'e':
                        row_features.extend([0, 1])
                    else:
                        row_features.extend([1, 0])
            list_2d_[i].extend(row_features)
        end_time = time.time()
        self.console.print('=========>ACCpro solvent: {time}s.'.format(time=end_time - start_time))
        return list_2d_

    def accpro20(
            self,
            df_accpro20,
            list_2d,
            window_aa_ids,
            mode='single',
    ):
        start_time = time.time()
        accpro20 = df_accpro20.values.tolist()
        # print(accpro20)
        window_aa_ids_ = window_aa_ids if mode == 'single' else [i[0] + i[1] for i in window_aa_ids]
        list_2d_ = list_2d
        zero = np.zeros(20).tolist()
        value_to_index = {
            0.0: 19,
            0.05: 18,
            0.1: 17,
            0.15: 16,
            0.2: 15,
            0.25: 14,
            0.3: 13,
            0.35: 12,
            0.4: 11,
            0.45: 10,
            0.5: 9,
            0.55: 8,
            0.6: 7,
            0.65: 6,
            0.7: 5,
            0.75: 4,
            0.205: 2,
            0.9: 1,
        }
        onehot = {
            value: [1 if idx == active_idx else 0 for idx in range(20)]
            for value, active_idx in value_to_index.items()
        }
        default = [1 if idx == 0 else 0 for idx in range(20)]
        for i, aa_win_ids in enumerate(window_aa_ids_):
            # print(i)
            row_features = []
            for j in aa_win_ids:
                # print(aa_win_ids)
                if j is None:
                    row_features.extend(zero)
                else:
                    row_features.extend(onehot.get(accpro20[j - 1][0], default))
            list_2d_[i].extend(row_features)
        end_time = time.time()
        self.console.print('=========>ACCpro20 solvent: {time}s.'.format(time=end_time - start_time))
        return list_2d_

    def solvpred(
            self,
            df_solvpred,
            list_2d,
            window_aa_ids,
            mode='single',
    ):
        start_time = time.time()
        solvpred = df_solvpred.values.tolist()
        # print(solvpred)
        window_aa_ids_ = window_aa_ids if mode == 'single' else [i[0] + i[1] for i in window_aa_ids]
        list_2d_ = list_2d
        for i, aa_win_ids in enumerate(window_aa_ids_):
            # print(i)
            row_features = []
            for j in aa_win_ids:
                # print(aa_win_ids)
                if j is None:
                    row_features.append(0)
                else:
                    row_features.append(solvpred[j-1][2])
            list_2d_[i].extend(row_features)
        end_time = time.time()
        print('=========>solvpred solvent: {time}s.'.format(time=end_time - start_time))
        return list_2d_


if __name__ == "__main__":
    from pypropel.prot.sequence.Fasta import Fasta as sfasta
    from pypropel.path import to
    import tmkit as tmk

    sequence = sfasta().get(
        fasta_fpn=to("data/fasta/1aigL.fasta")
    )
    print(sequence)

    pos_list = tmk.seq.pos_list_pair(len_seq=len(sequence), seq_sep_superior=None, seq_sep_inferior=0)
    # print(pos_list)

    positions = tmk.seq.pos_pair(sequence=sequence, pos_list=pos_list)
    # print(positions)

    window_size = 0
    win_aa_ids = tmk.seq.win_id_pair(
        sequence=sequence,
        position=positions,
        window_size=window_size,
    )
    print(win_aa_ids)

    features_1d_in = [[] for i in range(len(sequence))]
    features_2d_in = positions

    from pypropel.prot.feature.rsa.Reader import Reader as a11yreader

    p = Assemble()
    # df_accpro = a11yreader().accpro(
    #     accpro_path=to('data/accessibility/accpro/'),
    #     prot_name='1aig',
    #     file_chain='L'
    # )
    # print(p.accpro(
    #     df_accpro=df_accpro,
    #     list_2d=positions,
    #     window_aa_ids=win_aa_ids,
    #     mode='pair'
    # ))

    # df_accpro20 = a11yreader().accpro20(
    #     accpro20_path=to('data/accessibility/accpro20/'),
    #     prot_name='1aig',
    #     file_chain='L'
    # )
    # print(p.accpro20(
    #     df_accpro20,
    #     list_2d=positions,
    #     window_aa_ids=win_aa_ids,
    #     mode='pair'
    # ))

    df_solvpred = a11yreader().solvpred(
        solvpred_fp=to('data/accessibility/solvpred/'),
        prot_name='1aig',
        file_chain='L',
    )
    print(p.solvpred(
        df_solvpred=df_solvpred,
        list_2d=positions,
        window_aa_ids=win_aa_ids,
        mode='pair'
    ))
