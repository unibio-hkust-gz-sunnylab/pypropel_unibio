__author__ = "Jianfeng Sun"
__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"
__email__ = "jianfeng.sunmt@gmail.com"
__maintainer__ = "Jianfeng Sun"

from pathlib import Path
from typing import Callable, Iterable

from Bio.PDB import PDBParser

from pypropel.prot.sequence.Fasta import Fasta


Problem = tuple[str, str]


class ChainQC:

    def __init__(
            self,
            qc_mode: str = "first_failure",
            fasta_reader: Callable[[Path], str] | None = None,
            complex_chain_counter: Callable[[Path], int] | None = None,
            topology_reader: Callable[[Path, str, str], object] | None = None,
    ):
        if qc_mode not in {"first_failure", "all_problems"}:
            raise ValueError(f"Unsupported chain QC mode: {qc_mode}")
        self.qc_mode = qc_mode
        self.fasta_reader = fasta_reader
        self.complex_chain_counter = complex_chain_counter
        self.topology_reader = topology_reader

    def inspect(
            self,
            pdb_id: str,
            chain_id: str,
            complex_pdb_path: str | Path | None = None,
            packed_pdb_path: str | Path | None = None,
            fasta_path: str | Path | None = None,
            xml_path: str | Path | None = None,
    ) -> dict:
        paths = {
            "complex_pdb": _as_path(complex_pdb_path),
            "packed_pdb": _as_path(packed_pdb_path),
            "fasta": _as_path(fasta_path),
            "xml": _as_path(xml_path),
        }
        row = {
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "full_id": f"{pdb_id}_{chain_id}",
            "complex_pdb_path": str(paths["complex_pdb"] or ""),
            "packed_pdb_path": str(paths["packed_pdb"] or ""),
            "fasta_path": str(paths["fasta"] or ""),
            "xml_path": str(paths["xml"] or ""),
        }

        missing = _missing_paths(paths.items())
        if self.qc_mode == "first_failure":
            return _inspect_chain_first_failure(
                row=row,
                pdb_id=pdb_id,
                chain_id=chain_id,
                paths=paths,
                missing=missing,
                fasta_reader=self.fasta_reader,
                complex_chain_counter=self.complex_chain_counter,
                topology_reader=self.topology_reader,
            )
        return _inspect_chain_all_problems(
            row=row,
            pdb_id=pdb_id,
            chain_id=chain_id,
            paths=paths,
            missing=missing,
            fasta_reader=self.fasta_reader,
            complex_chain_counter=self.complex_chain_counter,
            topology_reader=self.topology_reader,
        )

    def inspect_many(self, records: Iterable[dict]) -> list[dict]:
        return [
            self.inspect(
                pdb_id=_record_value(record, "pdb_id", "prot"),
                chain_id=_record_value(record, "chain_id", "chain"),
                complex_pdb_path=record.get("complex_pdb_path"),
                packed_pdb_path=record.get("packed_pdb_path"),
                fasta_path=record.get("fasta_path"),
                xml_path=record.get("xml_path"),
            )
            for record in records
        ]


def inspect_chain(
        pdb_id: str,
        chain_id: str,
        complex_pdb_path: str | Path | None = None,
        packed_pdb_path: str | Path | None = None,
        fasta_path: str | Path | None = None,
        xml_path: str | Path | None = None,
        qc_mode: str = "first_failure",
        fasta_reader: Callable[[Path], str] | None = None,
        complex_chain_counter: Callable[[Path], int] | None = None,
        topology_reader: Callable[[Path, str, str], object] | None = None,
) -> dict:
    """
    Inspect reusable per-chain preprocessing QC conditions.

    ``qc_mode="first_failure"`` preserves the legacy style where the first
    detected problem becomes the row status. ``qc_mode="all_problems"`` tests
    each available condition independently and records all detected problem
    codes in ``qc_problem_codes``.
    """
    return ChainQC(
        qc_mode=qc_mode,
        fasta_reader=fasta_reader,
        complex_chain_counter=complex_chain_counter,
        topology_reader=topology_reader,
    ).inspect(
        pdb_id=pdb_id,
        chain_id=chain_id,
        complex_pdb_path=complex_pdb_path,
        packed_pdb_path=packed_pdb_path,
        fasta_path=fasta_path,
        xml_path=xml_path,
    )


def inspect_chains(
        records: Iterable[dict],
        qc_mode: str = "first_failure",
        fasta_reader: Callable[[Path], str] | None = None,
        complex_chain_counter: Callable[[Path], int] | None = None,
        topology_reader: Callable[[Path, str, str], object] | None = None,
) -> list[dict]:
    return ChainQC(
        qc_mode=qc_mode,
        fasta_reader=fasta_reader,
        complex_chain_counter=complex_chain_counter,
        topology_reader=topology_reader,
    ).inspect_many(records)


def _inspect_chain_first_failure(
        row: dict,
        pdb_id: str,
        chain_id: str,
        paths: dict[str, Path | None],
        missing: list[str],
        fasta_reader: Callable[[Path], str] | None,
        complex_chain_counter: Callable[[Path], int] | None,
        topology_reader: Callable[[Path, str, str], object] | None,
) -> dict:
    if missing:
        return _finish_problem_row(row, "missing_required_file", ",".join(missing))

    try:
        sequence = _read_fasta(paths["fasta"], fasta_reader)
        chain_count = _count_complex_chains(paths["complex_pdb"], complex_chain_counter)
        tmh_segment_count = _read_topology_segment_count(paths["xml"], pdb_id, chain_id, topology_reader)
    except Exception as exc:  # noqa: BLE001
        return _finish_problem_row(row, "library_qc_failed", repr(exc))

    if not sequence:
        return _finish_problem_row(
            row,
            "empty_sequence",
            "",
            sequence_length=0,
            complex_chain_count=chain_count,
            tmh_segment_count=tmh_segment_count,
        )
    if chain_count < 2:
        return _finish_problem_row(
            row,
            "monomer_complex",
            "",
            sequence_length=len(sequence),
            complex_chain_count=chain_count,
            tmh_segment_count=tmh_segment_count,
        )
    row.update({
        "status": "eligible",
        "detail": "",
        "sequence_length": len(sequence),
        "complex_chain_count": chain_count,
        "tmh_segment_count": tmh_segment_count,
    })
    return row


def _inspect_chain_all_problems(
        row: dict,
        pdb_id: str,
        chain_id: str,
        paths: dict[str, Path | None],
        missing: list[str],
        fasta_reader: Callable[[Path], str] | None,
        complex_chain_counter: Callable[[Path], int] | None,
        topology_reader: Callable[[Path, str, str], object] | None,
) -> dict:
    problems: list[Problem] = []
    if missing:
        problems.append(("missing_required_file", ",".join(missing)))

    sequence = ""
    sequence_available = False
    if _path_exists(paths["fasta"]):
        try:
            sequence = _read_fasta(paths["fasta"], fasta_reader)
            sequence_available = True
            if not sequence:
                problems.append(("empty_sequence", ""))
        except Exception as exc:  # noqa: BLE001
            problems.append(("library_qc_failed", f"fasta:{repr(exc)}"))

    chain_count = 0
    if _path_exists(paths["complex_pdb"]):
        try:
            chain_count = _count_complex_chains(paths["complex_pdb"], complex_chain_counter)
            if chain_count < 2:
                problems.append(("monomer_complex", ""))
        except Exception as exc:  # noqa: BLE001
            problems.append(("library_qc_failed", f"complex_pdb:{repr(exc)}"))

    tmh_segment_count = 0
    if _path_exists(paths["xml"]) and topology_reader is not None:
        try:
            tmh_segment_count = _read_topology_segment_count(paths["xml"], pdb_id, chain_id, topology_reader)
        except Exception as exc:  # noqa: BLE001
            problems.append(("library_qc_failed", f"topology:{repr(exc)}"))

    if problems:
        status, detail = primary_chain_qc_problem(problems)
        row.update({
            "status": status,
            "detail": detail,
            "sequence_length": len(sequence) if sequence_available else 0,
            "complex_chain_count": chain_count,
            "tmh_segment_count": tmh_segment_count,
            "qc_problem_count": len(problems),
            "qc_problem_codes": "|".join(code for code, _detail in problems),
            "qc_problem_details": "; ".join(_format_problem(code, detail) for code, detail in problems),
        })
        return row

    row.update({
        "status": "eligible",
        "detail": "",
        "sequence_length": len(sequence),
        "complex_chain_count": chain_count,
        "tmh_segment_count": tmh_segment_count,
        "qc_problem_count": 0,
        "qc_problem_codes": "",
        "qc_problem_details": "",
    })
    return row


def primary_chain_qc_problem(problems: list[Problem]) -> Problem:
    priority = [
        "missing_required_file",
        "library_qc_failed",
        "empty_sequence",
        "monomer_complex",
    ]
    for status in priority:
        for code, detail in problems:
            if code == status:
                return code, detail
    return problems[0]


def count_chains(path: str | Path) -> int:
    path = Path(path)
    structure = PDBParser(QUIET=True).get_structure(path.stem, str(path))
    return len(list(structure[0].get_chains()))


def _as_path(path: str | Path | None) -> Path | None:
    return Path(path) if path else None


def _missing_paths(paths: Iterable[tuple[str, Path | None]]) -> list[str]:
    return [name for name, path in paths if path is not None and not path.exists()]


def _path_exists(path: Path | None) -> bool:
    return path is not None and path.exists()


def _record_value(record: dict, primary_key: str, fallback_key: str) -> str:
    if primary_key in record:
        return record[primary_key]
    if fallback_key in record:
        return record[fallback_key]
    raise KeyError(f"record must contain {primary_key!r} or {fallback_key!r}")


def _read_fasta(path: Path | None, fasta_reader: Callable[[Path], str] | None) -> str:
    if path is None:
        return ""
    if fasta_reader is not None:
        return fasta_reader(path)
    return Fasta().get(str(path))


def _count_complex_chains(path: Path | None, complex_chain_counter: Callable[[Path], int] | None) -> int:
    if path is None:
        return 0
    if complex_chain_counter is not None:
        return complex_chain_counter(path)
    return count_chains(path)


def _read_topology_segment_count(
        path: Path | None,
        pdb_id: str,
        chain_id: str,
        topology_reader: Callable[[Path, str, str], object] | None,
) -> int:
    if path is None or topology_reader is None:
        return 0
    topology = topology_reader(path, pdb_id, chain_id)
    if topology is None:
        return 0
    if isinstance(topology, int):
        return topology
    if isinstance(topology, tuple) and topology:
        return len(topology[0])
    return len(topology)


def _finish_problem_row(
        row: dict,
        status: str,
        detail: str,
        sequence_length: int = 0,
        complex_chain_count: int = 0,
        tmh_segment_count: int = 0,
) -> dict:
    row.update({
        "status": status,
        "detail": detail,
        "sequence_length": sequence_length,
        "complex_chain_count": complex_chain_count,
        "tmh_segment_count": tmh_segment_count,
    })
    return row


def _format_problem(code: str, detail: str) -> str:
    if detail:
        return f"{code}:{detail}"
    return code
