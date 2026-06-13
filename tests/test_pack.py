from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pypropel.prot.file.Pack import Pack
from pypropel.prot.structure.chain.Splitter import Splitter


class PackTest(unittest.TestCase):
    def test_sanitized_complex_pdb_splits_when_malformed_hetatm_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complex_dir = root / "complex"
            packed_dir = root / "packed"
            complex_dir.mkdir()
            packed_dir.mkdir()
            (complex_dir / "2acz.pdb").write_text(
                "\n".join(
                    [
                        "HEADER    TEST PDB",
                        "ATOM      1  N   ALA C   1      11.104  13.207   2.100  1.00 20.00           N",
                        "ATOM      2  CA  ALA C   1      12.104  13.207   2.100  1.00 20.00           C",
                        "HETATM    3  CL12 AT5 C 131      21.657 -24.957  15.072  1.00 85.07           C",
                        "ATOM      4  N   GLY D   1      10.104  12.207   2.100  1.00 20.00           N",
                        "ATOM      5  CA  GLY D   1      10.704  12.807   2.800  1.00 20.00           C",
                        "END",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            prot_df = pd.DataFrame({"prot": ["2acz", "2acz"], "chain": ["C", "D"]})
            pack = Pack(prot_df=prot_df, verbose=False)

            sanitized_dir = pack._sanitize_complex_pdbs(
                pdb_cplx_fp=str(complex_dir) + "/",
                pdb_fp=str(packed_dir) + "/",
            )
            Splitter(
                prot_df=prot_df,
                pdb_path=sanitized_dir,
                sv_fp=str(packed_dir) + "/",
                verbose=False,
            ).pdb_per_chain()
            split_df = pack._filter_existing_split_chains(str(packed_dir) + "/")

            self.assertEqual(
                split_df.to_dict(orient="records"),
                [{"prot": "2acz", "chain": "C"}, {"prot": "2acz", "chain": "D"}],
            )
            self.assertTrue((packed_dir / "2aczC.pdb").exists())
            self.assertTrue((packed_dir / "2aczD.pdb").exists())
            manifest = pd.read_csv(packed_dir / "complex_sanitize_manifest.txt", sep="\t")
            self.assertEqual(int(manifest.loc[0, "hetatm_removed"]), 1)
            self.assertEqual(int(manifest.loc[0, "malformed_records_removed"]), 1)


if __name__ == "__main__":
    unittest.main()
