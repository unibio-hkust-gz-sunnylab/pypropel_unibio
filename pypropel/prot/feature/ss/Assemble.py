__author__ = "Jianfeng Sun"
__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"
__email__ = "jianfeng.sunmt@gmail.com"
__maintainer__ = "Jianfeng Sun"

import time
import numpy as np
from pypropel.util.normalization.Standardize import Standardize
from pypropel.util.Console import Console


class Assemble:

    def __init__(
            self,
            verbose: bool = True,
    ):
        self.console = Console()
        self.console.verbose = verbose

    def spider3(
            self,
            df_spider3,
            list_2d,
            window_aa_ids,
            mark=-3,
            std=False,
    ):
        start_time = time.time()
        spider3 = df_spider3.values.tolist()
        # print(spider3)
        window_aa_ids_ = [i[0] + i[1] for i in window_aa_ids]
        list_2d_ = list_2d
        standardize = Standardize()
        zero = np.zeros(13).tolist()
        for i, aa_win_ids in enumerate(window_aa_ids_):
            # print(i)
            row_features = []
            for j in aa_win_ids:
                # print(aa_win_ids)
                if j is None:
                    row_features.extend(zero)
                else:
                    trans = standardize.minmax1(spider3[j-1][mark:]) if std else spider3[j-1][mark:]
                    row_features.extend(trans)
            list_2d_[i].extend(row_features)
        end_time = time.time()
        print('------> spider3 ss: {time}s.'.format(time=end_time - start_time))
        return list_2d_

    def sspro(
            self,
            df_sspro,
            list_2d,
            window_aa_ids,
    ):
        start_time = time.time()
        sspro = df_sspro.values.tolist()
        print(sspro)
        window_aa_ids_ = [i[0] + i[1] for i in window_aa_ids]
        list_2d_ = list_2d
        zero = np.zeros(3).tolist()
        for i, aa_win_ids in enumerate(window_aa_ids_):
            row_features = []
            for j in aa_win_ids:
                # print(aa_win_ids)
                if j is None:
                    row_features.extend(zero)
                else:

                    if sspro[j - 1][0] == 'H':
                        row_features.extend([0, 0, 1])
                    elif sspro[j - 1][0] == 'E':
                        row_features.extend([0, 1, 0])
                    else:
                        print(j - 1)
                        row_features.extend([1, 0, 0])
            list_2d_[i].extend(row_features)
        end_time = time.time()
        print('------> sspro ss: {time}s.'.format(time=end_time - start_time))
        return list_2d_

    def sspro8(
            self,
            df_sspro8,
            list_2d,
            window_aa_ids,
    ):
        start_time = time.time()
        sspro8 = df_sspro8.values.tolist()
        # print(sspro8)
        window_aa_ids_ = [i[0] + i[1] for i in window_aa_ids]
        list_2d_ = list_2d
        zero = np.zeros(8).tolist()
        sspro8_map = {
            'H': [1 if idx == 7 else 0 for idx in range(8)],
            'G': [1 if idx == 6 else 0 for idx in range(8)],
            'I': [1 if idx == 5 else 0 for idx in range(8)],
            'E': [1 if idx == 4 else 0 for idx in range(8)],
            'B': [1 if idx == 3 else 0 for idx in range(8)],
            'T': [1 if idx == 2 else 0 for idx in range(8)],
            'S': [1 if idx == 1 else 0 for idx in range(8)],
        }
        default = [1 if idx == 0 else 0 for idx in range(8)]
        for i, aa_win_ids in enumerate(window_aa_ids_):
            # print(i)
            row_features = []
            for j in aa_win_ids:
                # print(aa_win_ids)
                if j is None:
                    row_features.extend(zero)
                else:
                    row_features.extend(sspro8_map.get(sspro8[j - 1][0], default))
            list_2d_[i].extend(row_features)
        end_time = time.time()
        print('------> sspro8 ss: {time}s.'.format(time=end_time - start_time))
        return list_2d_
    
    def psipred(
            self,
            df_psipred_ss2,
            list_2d,
            window_aa_ids,
    ):
        start_time = time.time()
        psipred = df_psipred_ss2.values.tolist()
        # print(psipred)
        window_aa_ids_ = [i[0] + i[1] for i in window_aa_ids]
        list_2d_ = list_2d
        zero = np.zeros(3).tolist()
        for i, aa_win_ids in enumerate(window_aa_ids_):
            # print(i)
            row_features = []
            for j in aa_win_ids:
                # print(aa_win_ids)
                if j is None:
                    row_features.extend(zero)
                else:
                    row_features.extend(psipred[j-1][3:])
            list_2d_[i].extend(row_features)
        end_time = time.time()
        print('------> psipred ss: {time}s.'.format(time=end_time - start_time))
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

    from pypropel.prot.feature.ss.Reader import Reader as ssreader

    p = Assemble()
    df_spider3 = ssreader().spider3(
        spider3_path=to('data/ss/spider3/'),
        prot_name='E',
        file_chain=''
    )
    print(p.spider3(
        df_spider3=df_spider3,
        list_2d=positions,
        window_aa_ids=win_aa_ids,
        # std=True
    ))

    # df_sspro = ssreader().sspro(
    #     sspro_path=to('data/ss/sspro/'),
    #     prot_name='1aig',
    #     file_chain='L'
    # )
    # print(p.sspro(
    #     df_sspro=df_sspro,
    #     list_2d=positions,
    #     window_aa_ids=win_aa_ids
    # ))

    # df_sspro8 = ssreader().sspro8(
    #     sspro8_path=to('data/ss/sspro8/'),
    #     prot_name='1aig',
    #     file_chain='L'
    # )
    # print(p.sspro8(
    #     df_sspro8=df_sspro8,
    #     list_2d=features_2d_in,
    #     window_aa_ids=win_aa_ids
    # ))

    # df_psipred_ss2 = ssreader().psipred(
    #     psipred_ss2_path=to('data/ss/psipred/'),
    #     prot_name='1aig', # 1aig L 1bcc D
    #     file_chain='L',
    # )
    # print(p.psipred(
    #     df_psipred_ss2=df_psipred_ss2,
    #     list_2d=positions,
    #     window_aa_ids=win_aa_ids
    # ))
