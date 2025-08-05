# Copyright 2025 Entalpic
import logging
import math
from shutil import which

import moyopy
from monty.tempfile import ScratchDir
from moyopy.interface import MoyoAdapter
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure


logger = logging.getLogger(__name__)


class MoyoSymmetry:
    """
    This is a wrapper around the functions of the Moyo library.
    It is used to get the symmetry label of a structure, while handling the parameters
    that can be used for the symmetry detection.
    By default, we use the same parameters in the original library, and let the user
    pass custom parameters if needed.

    Parameters
    ----------
    symprec (float, optional): Distance tolerance in Angstroms to find 
        crystal symmetry. Defaults to 0.01, default `symprec` for 
        pytmatgen's `SpacegroupAnalyzer`.
    rad_angle_tolerance (float, optional): Tolerance of angle between 
        basis vectors in radians to be tolerated in the symmetry 
        finding. Value in radians. Defaults to None, since the internet 
        suggests not to use this variable: https://github.com/spglib/spglib/issues/567
    setting : str, optional
        Setting. Defaults to None.
    """

    def __init__(
        self,
        symprec: float = 0.01,
        rad_angle_tolerance: float | None = None,
        setting: str | None = None
    ):
        self.symprec = symprec
        self.rad_angle_tolerance = rad_angle_tolerance
        self.setting = setting

    def get_symmetry_label(self, structure: Structure) -> int | None:
        """Get symmetry space group number from structure

        Parameters
        ----------
        structure : Structure
            Input structure

        Returns
        -------
        int: space group number
        """
        try:
            cell = MoyoAdapter.from_structure(structure)
            dataset = moyopy.MoyoDataset(
                cell=cell,
                symprec=self.symprec,
                angle_tolerance=self.rad_angle_tolerance,
                setting=self.setting,
            )
        except Exception as e:
            logger.warning(
                f"Error getting symmetry label for structure: {e}, will return None"
            )
            return None
        return dataset.number


class SPGLibSymmetry:
    """
    Object used to compute symmetry based on SPGLib
    """

    def __init__(
        self,
        symprec: float = 0.01,
        angle_tolerance: float = -1,
    ) -> None:
        """Set settings for Pymatgen's symmetry detection

        Args:
            symprec (float, optional): Distance tolerance in Angstroms to 
                find crystal symmetry. Defaults to 0.01, default `symprec`
                for pytmatgen's `SpacegroupAnalyzer`.
            angle_tolerance (float, optional): Tolerance of angle between 
                basis vectors in degrees to be tolerated in the symmetry 
                finding. Value in degrees. Defaults to -1 degrees, an 
                automatic algorithm in SPGLib. The internet suggests not 
                to use this variable: https://github.com/spglib/spglib/issues/567
        """
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

    def get_symmetry_label(self, structure: Structure) -> int | None:
        """Get symmetry space group number from structure

        Args:
            structure (Structure): input structure

        Returns:
            int: space group number
        """
        try:
            sga = SpacegroupAnalyzer(structure, self.symprec, self.angle_tolerance)
            return sga.get_symmetry_dataset().number
        except Exception as e:
            logger.warning(
                f"Error getting symmetry label for structure: {e}, will return None"
            )
            return None


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

    def get_symmetry_label(self, structure: Structure, tolerance: float = 0.1) -> str | None:
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

        try:
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
        except Exception as e:
            logger.warning(
                f"Error getting symmetry label for structure: {e}, will return None"
            )
            return None
