__author__ = "Jianfeng Sun"
__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"
__email__ = "jianfeng.sunmt@gmail.com"
__maintainer__ = "Jianfeng Sun"

import os
import sys
sys.path.append(os.path.dirname(os.getcwd()) + '/')
from abc import ABCMeta, abstractmethod
import numpy as np
# from Bio.PDB.Polypeptide import three_to_one
from pypropel.util.Console import Console
console = Console()

three_to_one = {
    'CYS': 'C',
    'ASP': 'D',
    'SER': 'S',
    'GLN': 'Q',
    'LYS': 'K',
    'ILE': 'I',
    'PRO': 'P',
    'THR': 'T',
    'PHE': 'F',
    'ASN': 'N',
    'GLY': 'G',
    'HIS': 'H',
    'LEU': 'L',
    'ARG': 'R',
    'TRP': 'W',
    'ALA': 'A',
    'VAL':'V',
    'GLU': 'E',
    'TYR': 'Y',
    'MET': 'M',
}


class distance(metaclass=ABCMeta):

    @abstractmethod
    def calculate(self):
        pass

    def _standard_residue_records(self, chain):
        records = []
        count_hetatm = 0
        for index, residue in enumerate(chain):
            if residue.get_id()[0] != ' ':
                count_hetatm += 1
                continue
            records.append((index, index + 1 - count_hetatm, residue))
        return records

    def _heavy_atom_coords(self, residue):
        return np.asarray([
            atom.get_coord()
            for atom in residue
            if atom.get_name() != 'H'
        ], dtype=float)

    def _min_atom_distance(self, coords1, coords2):
        diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
        return float(np.sqrt(np.sum(diff * diff, axis=2)).min())

    def one2one_minimal(
            self,
            chain1,
            chain2,
            verbose: bool = False,
    ):
        """
        Notes
        -----
        It outputs minimal distances of each residue in chain 1 to a residue in chain 2.


        Parameters
        ----------
        chain1
            a biopython-typed chain ob 1
        chain2
            a biopython-typed chain ob 2

        Returns
        -------
        2d array

        """
        console.verbose = verbose
        dist_matrix = []
        chain2_records = [
            (index_2, fasta_id_2, residue_2, self._heavy_atom_coords(residue_2))
            for index_2, fasta_id_2, residue_2 in self._standard_residue_records(chain2)
        ]
        for index_1, fasta_id_1, residue_1 in self._standard_residue_records(chain1):
            console.print("==================>residue 1 ID: {}".format(index_1))
            coords1 = self._heavy_atom_coords(residue_1)
            residue_dist = [
                self._min_atom_distance(coords1, coords2)
                for _, _, _, coords2 in chain2_records
            ]
            min_residue_dist = min(residue_dist)
            dist_matrix.append([
                fasta_id_1,
                three_to_one[residue_1.get_resname()],
                residue_1.id[1],
                min_residue_dist
            ])
        return dist_matrix

    def one2one_all(
            self,
            chain1,
            chain2,
            verbose: bool = False,
    ):
        """

        Notes
        -----
        It outputs the minimum distance of each residue in chain 1 to
        each residue in the chain 2.

        Parameters
        ----------
        chain1
            a biopython-typed chain ob 1
        chain2
            a biopython-typed chain ob 2

        Returns
        -------
        2d array
        """
        console.verbose = verbose
        dist_matrix = []
        chain2_records = [
            (
                index_2,
                residue_2,
                None if residue_2.get_id()[0] != ' ' else self._heavy_atom_coords(residue_2),
            )
            for index_2, residue_2 in enumerate(chain2)
        ]
        count_hetamt_2 = 0
        for index_1, fasta_id_1, residue_1 in self._standard_residue_records(chain1):
            console.print("==================>residue 1 ID: {}".format(index_1))
            coords1 = self._heavy_atom_coords(residue_1)
            for index_2, residue_2, coords2 in chain2_records:
                if residue_2.get_id()[0] != ' ':
                    count_hetamt_2 += 1
                    continue
                min_dist = self._min_atom_distance(coords1, coords2)
                dist_matrix.append([
                    fasta_id_1,
                    three_to_one[residue_1.get_resname()],
                    residue_1.id[1],
                    index_2 + 1 - count_hetamt_2,
                    'U' if residue_2.get_resname() == 'UNK' else three_to_one[residue_2.get_resname()],
                    residue_2.id[1],
                    min_dist,
                ])
        return dist_matrix

    def check(
            self,
            chain1,
            chain2,
            thres=6,
            verbose: bool = False,
    ):
        """
        Each residue in the chain 1 has a minimum distance against all of
        residues in the chain 2.
        It stops the calculations of the minimum distance of each residue
        to each residue in the chain 2 when it detects a minimum distance
        of less than thres, 6 by default.

        Parameters
        ----------
        chain1
            a biopython-typed chain ob 1
        chain2
            a biopython-typed chain ob 2
        thres

        Returns
        -------
        2d array

        """
        console.verbose = verbose
        chain2_records = [
            (index_2, residue_2, self._heavy_atom_coords(residue_2))
            for index_2, _, residue_2 in self._standard_residue_records(chain2)
        ]
        for index_1, _, residue_1 in self._standard_residue_records(chain1):
            console.print("==================>residue 1 ID: {}".format(index_1))
            coords1 = self._heavy_atom_coords(residue_1)
            for index_2, residue_2, coords2 in chain2_records:
                min_dist = self._min_atom_distance(coords1, coords2)
                if min_dist < thres:
                    console.print("==================>residue {} and residue {} in interaction".format(index_1, index_2))
                    return True
        return False
