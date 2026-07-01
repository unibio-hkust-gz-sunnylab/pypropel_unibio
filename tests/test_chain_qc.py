from pathlib import Path
import tempfile
import unittest

import pypropel as pp
from pypropel.prot.file.ChainQC import ChainQC
from pypropel.qc import inspect_chain


class ChainQCTest(unittest.TestCase):
    def test_qc_is_available_from_lazy_package_facade(self):
        self.assertEqual(pp.qc.inspect_chain.__name__, "inspect_chain")

    def test_class_api_accepts_pypropel_prot_chain_keys(self):
        rows = ChainQC(
            qc_mode="first_failure",
            fasta_reader=lambda _path: self.fail("fasta_reader should not run"),
        ).inspect_many([
            {
                "prot": "4abc",
                "chain": "B",
                "fasta_path": "/tmp/does-not-exist.fasta",
            }
        ])

        self.assertEqual(rows[0]["pdb_id"], "4abc")
        self.assertEqual(rows[0]["chain_id"], "B")
        self.assertEqual(rows[0]["status"], "missing_required_file")

    def test_class_api_requires_chain_identifiers(self):
        with self.assertRaisesRegex(KeyError, "pdb_id"):
            ChainQC().inspect_many([{"chain": "A"}])

    def test_all_problems_reports_empty_sequence_and_monomer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complex_path = root / "1abc.pdb"
            packed_path = root / "1abcA.pdb"
            fasta_path = root / "1abcA.fasta"
            xml_path = root / "1abc.xml"
            complex_path.write_text("HEADER test\n")
            packed_path.write_text("HEADER test\n")
            fasta_path.write_text(">1abc_A\n")
            xml_path.write_text("<pdbtm />\n")

            row = inspect_chain(
                "1abc",
                "A",
                complex_pdb_path=complex_path,
                packed_pdb_path=packed_path,
                fasta_path=fasta_path,
                xml_path=xml_path,
                qc_mode="all_problems",
                fasta_reader=lambda _path: "",
                complex_chain_counter=lambda _path: 1,
            )

        self.assertEqual(row["status"], "empty_sequence")
        self.assertEqual(row["qc_problem_count"], 2)
        self.assertEqual(row["qc_problem_codes"], "empty_sequence|monomer_complex")

    def test_all_problems_reports_missing_and_available_parse_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complex_path = root / "2abc.pdb"
            fasta_path = root / "2abcA.fasta"
            xml_path = root / "2abc.xml"
            complex_path.write_text("HEADER test\n")
            fasta_path.write_text(">2abc_A\nACD\n")
            xml_path.write_text("<pdbtm />\n")

            def fail_topology(_path, _pdb_id, _chain_id):
                raise ValueError("bad topology")

            row = inspect_chain(
                "2abc",
                "A",
                complex_pdb_path=complex_path,
                packed_pdb_path=root / "missing.pdb",
                fasta_path=fasta_path,
                xml_path=xml_path,
                qc_mode="all_problems",
                fasta_reader=lambda _path: "ACD",
                complex_chain_counter=lambda _path: 2,
                topology_reader=fail_topology,
            )

        self.assertEqual(row["status"], "missing_required_file")
        self.assertEqual(row["detail"], "packed_pdb")
        self.assertEqual(row["sequence_length"], 3)
        self.assertEqual(row["complex_chain_count"], 2)
        self.assertEqual(row["qc_problem_codes"], "missing_required_file|library_qc_failed")
        self.assertIn("topology:ValueError", row["qc_problem_details"])

    def test_first_failure_keeps_legacy_short_circuit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = inspect_chain(
                "3abc",
                "A",
                complex_pdb_path=root / "missing_complex.pdb",
                fasta_path=root / "missing.fasta",
                qc_mode="first_failure",
                fasta_reader=lambda _path: self.fail("fasta_reader should not run"),
                complex_chain_counter=lambda _path: self.fail("complex_chain_counter should not run"),
            )

        self.assertEqual(row["status"], "missing_required_file")
        self.assertEqual(row["detail"], "complex_pdb,fasta")
        self.assertNotIn("qc_problem_codes", row)


if __name__ == "__main__":
    unittest.main()
