# Copyright 2025 Entalpic
from shutil import which

import moyopy
from monty.tempfile import ScratchDir
from moyopy.interface import MoyoAdapter
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure


class MoyoSymmetry:
    """
    Moyo symmetry using Moyo library

    Parameters
    ----------
    symprec : float, optional
        Symmetry precision tollerance. Defaults to 1e-4.
    angle_tolerance : float, optional
        Angle tolerance. Defaults to None.
    setting : str, optional
        Setting. Defaults to None.
    """

    def __init__(
        self, symprec: float = 1e-4, angle_tolerance: float = None, setting: str = None
    ):
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self.setting = setting

    def get_symmetry_label(self, structure: Structure) -> int:
        """Get symmetry space group number from structure

        Parameters
        ----------
        structure : Structure
            Input structure

        Returns
        -------
        int: space group number
        """
        cell = MoyoAdapter.from_structure(structure)
        dataset = moyopy.MoyoDataset(
            cell=cell,
            symprec=self.symprec,
            angle_tolerance=self.angle_tolerance,
            setting=self.setting,
        )
        return dataset.number


class SPGLibSymmetry:
    """
    Object used to compute symmetry based on SPGLib
    """

    def __init__(self, symprec: float = 0.01):
        """Set settings for Pymatgen's symmetry detection

        Args:
            symprec (float, optional): Symmetry precision tollerance.
              Defaults to 0.01.
        """
        self.symprec = symprec

    def get_symmetry_label(self, structure: Structure) -> int:
        """Get symmetry space group number from structure

        Args:
            structure (Structure): input structure

        Returns:
            int: space group number
        """
        sga = SpacegroupAnalyzer(structure, self.symprec)
        return sga.get_symmetry_dataset().number


class AFLOWSymmetry:
    """
    AFLOW prototype label using AFLOW libary
    """

    def __init__(self, aflow_executable: str = None):
        """AFLOW Symmetry

        Args:
            aflow_executable (str, optional): AFLOW executable.
                If none listed tries to find aflow in system path

        Raises:
            RuntimeError: If AFLOW is not found
        """

        self.aflow_executable = aflow_executable or which("aflow")

        print("aflow found in {}".format(self.aflow_executable))

        if self.aflow_executable is None:
            raise RuntimeError(
                "Requires aflow to be in the PATH or the absolute path to "
                f"the binary to be specified via {self.aflow_executable=}.\n"
            )

    def get_symmetry_label(self, structure: Structure, tolerance: float = 0.1) -> str:
        """
        Returns AFLOW label for a given structure
        Args:
            structure (Structure): structure to run AFLOW on
            tolerance (float, optional): AFLOW symmetry tolerance. Defaults to 0.1.

        Returns:
            str: AFLOW label
        """

        # fmt: off
        from aflow_xtal_finder import XtalFinder
        # fmt: on

        xtf = XtalFinder(self.aflow_executable)
        with ScratchDir("."):
            structure.to_file("POSCAR")
            data = xtf.get_prototype_label(
                [
                    "POSCAR",
                ],
                options="--tolerance={}".format(tolerance),
            )
        return data
