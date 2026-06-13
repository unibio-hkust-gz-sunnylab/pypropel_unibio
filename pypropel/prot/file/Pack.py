__author__ = "Jianfeng Sun"
__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"
__email__ = "jianfeng.sunmt@gmail.com"
__maintainer__ = "Jianfeng Sun"

from pypropel.prot.structure.chain.Splitter import Splitter
from pypropel.prot.structure.chain.Format import Format
from pypropel.prot.structure.convert.ToFasta import ToFasta
from pypropel.prot.structure.hetatm.Remove import Remove as hetatmremover
from pypropel.prot.sequence.IsEmpty import IsEmpty
from pypropel.prot.sequence.IsMatch import IsMatch
from pypropel.prot.sequence.Name import Name as chainname
from pypropel.util.FileIO import FileIO
from pypropel.util.Console import Console
import os
import pandas as pd


class Pack:

    def __init__(
            self,
            prot_df,
            verbose: bool = True,
    ):
        self.prot_df = prot_df
        self.console = Console()
        self.console.verbose = verbose

    def execute(
            self,
            pdb_cplx_fp,
            pdb_fp,
            xml_fp,
            fasta_fp,
            kind='pdb<->xml',
    ):
        # ### /* block 1. split into chains */ ###
        self.console.print('=========>++++++++++++++++++++split into chains...\n+++++++++++++++++++++++++++')
        pdb_cplx_split_fp = self._sanitize_complex_pdbs(
            pdb_cplx_fp=pdb_cplx_fp,
            pdb_fp=pdb_fp,
        )
        Splitter(
            prot_df=self.prot_df,
            pdb_path=pdb_cplx_split_fp,
            sv_fp=pdb_fp,
        ).pdb_per_chain()
        prot_df = self._filter_existing_split_chains(
            pdb_fp=pdb_fp,
        )
        if prot_df.empty:
            self.console.print('============>No split-chain PDB files were created; stopping pack.')
            return 'Finished'
        # ### /* block 2. delete END from PDB files */ ###
        self.console.print('=========>++++++++++++++++++++delete END from PDB files...\n+++++++++++++++++++++++++++')
        FileIO().makedir(pdb_fp + '/delend/')
        Format(
            prot_df=prot_df,
            sv_fp=pdb_fp + '/delend/',
        ).del_END_frompdb(
            pdb_path=pdb_fp,
        )
        # ### /* block 3. remove hetatm from PDB files */ ###
        self.console.print('=========>++++++++++++++++++++remove hetatm from PDB files...\n+++++++++++++++++++++++++++')
        hetatmremover(prot_df=prot_df).biopython(
            pdb_path=pdb_fp + '/delend/',
            sv_fp=pdb_fp,
        )
        # ### /* block 4. isMatch */ ###
        self.console.print('=========>++++++++++++++++++++is match...\n+++++++++++++++++++++++++++')
        IsMatch(
            prot_df=prot_df,
            pdb_path=pdb_fp + '/delend/',
            xml_path=xml_fp,
            sv_mismatch_fp=fasta_fp,
            kind=kind,
        ).execute()
        # ### /* block 5. ToFasta */ ###
        self.console.print('=========>++++++++++++++++++++to Fasta...\n+++++++++++++++++++++++++++')
        ToFasta(
            prot_df=prot_df,
            sv_fp=fasta_fp
        ).frompdb(
            pdb_path=pdb_fp + '/delend/',
        )
        # ### /* block 6. isEmpty */ ###
        self.console.print('=========>++++++++++++++++++++is empty...\n+++++++++++++++++++++++++++')
        IsEmpty(
            prot_df,
            sv_empty_fp=fasta_fp,
            fasta_fp=fasta_fp,
        ).fasta()
        return 'Finished'

    def _sanitize_complex_pdbs(
            self,
            pdb_cplx_fp,
            pdb_fp,
    ):
        """
        Write ligand-stripped complex PDB copies for Biopython chain splitting.

        PyPropel removes HETATM records later in the pack flow, but Biopython
        parses the whole complex before that stage. Some PDBTM transformed
        entries contain malformed ligand HETATM records whose fixed-width
        columns make Biopython reject the entire structure. Removing HETATM
        records before splitting preserves protein ATOM records and matches the
        eventual packed-chain content.
        """
        sanitized_fp = os.path.join(pdb_fp, 'complex_no_hetatm') + '/'
        FileIO().makedir(sanitized_fp)
        rows = []
        for prot_name in pd.unique(self.prot_df['prot']):
            src = os.path.join(pdb_cplx_fp, str(prot_name) + '.pdb')
            dst = os.path.join(sanitized_fp, str(prot_name) + '.pdb')
            row = {
                'prot': prot_name,
                'source': src,
                'sanitized': dst,
                'atom_records': 0,
                'hetatm_removed': 0,
                'malformed_records_removed': 0,
                'status': 'ok',
            }
            try:
                with open(src) as fin, open(dst, 'w') as fout:
                    for line in fin:
                        if line.startswith('ATOM  '):
                            row['atom_records'] += 1
                            fout.write(line)
                        elif line.startswith('HETATM'):
                            row['hetatm_removed'] += 1
                            if self._has_invalid_pdb_resseq(line):
                                row['malformed_records_removed'] += 1
                        else:
                            fout.write(line)
            except FileNotFoundError:
                row['status'] = 'missing_source'
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(pdb_fp, 'complex_sanitize_manifest.txt'),
            sep='\t',
            index=False,
        )
        return sanitized_fp

    def _filter_existing_split_chains(
            self,
            pdb_fp,
    ):
        ok_rows = []
        missing_rows = []
        for i in self.prot_df.index:
            prot_name = self.prot_df.loc[i, 'prot']
            prot_chain = self.prot_df.loc[i, 'chain']
            file_chain = chainname().chain(prot_chain)
            chain_fpn = os.path.join(pdb_fp, str(prot_name) + file_chain + '.pdb')
            row = {
                'prot': prot_name,
                'chain': prot_chain,
                'path': chain_fpn,
            }
            if os.path.isfile(chain_fpn) and os.path.getsize(chain_fpn) > 0:
                ok_rows.append({'prot': prot_name, 'chain': prot_chain})
            else:
                missing_rows.append(row)
        pd.DataFrame(missing_rows).to_csv(
            os.path.join(pdb_fp, 'missing_split_records.txt'),
            sep='\t',
            index=False,
        )
        pd.DataFrame(ok_rows).to_csv(
            os.path.join(pdb_fp, 'successful_split_records.txt'),
            sep='\t',
            index=False,
        )
        return pd.DataFrame(ok_rows, columns=['prot', 'chain']).reset_index(drop=True)

    def _has_invalid_pdb_resseq(self, line):
        if not line.startswith(('ATOM  ', 'HETATM')):
            return False
        try:
            int(line[22:26].split()[0])
        except (IndexError, ValueError):
            return True
        return False


if __name__ == "__main__":
    from pypropel.path import to

    import pandas as pd

    prot_df = pd.DataFrame({
        'prot': ['1aig', '1aij', '1xqf'],
        'chain': ['L', 'L', 'A'],
    })

    p = Pack(prot_df)

    print(p.execute(
        pdb_cplx_fp=to('data/pdb/complex/pdbtm/'),
        pdb_fp=to('data/'),
        xml_fp=to('data/xml/'),
        fasta_fp=to('data/'),
    ))
