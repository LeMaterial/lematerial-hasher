# Copyright 2025 Entalpic
import logging
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
    symprec : float, optional
        Symmetry precision tolerance. Defaults to 1e-4.
    angle_tolerance : float, optional
        Angle tolerance. Defaults to None.
    setting : str, optional
        Setting. Defaults to None.
    """

    def __init__(
        self, symprec: float | None = None, angle_tolerance: float | None = None, setting: str | None = None
    ):
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
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
            # If any of the parameters are provided, we use the custom parameters
            # Otherwise, we use the default parameters (default behavior)
            if any([self.symprec, self.angle_tolerance, self.setting]):
                dataset = moyopy.MoyoDataset(
                    cell=cell,
                    symprec=self.symprec,
                    angle_tolerance=self.angle_tolerance,
                    setting=self.setting,
                )
            else:
                dataset = moyopy.MoyoDataset(cell=cell)
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

    def __init__(self, symprec: float = 0.01):
        """Set settings for Pymatgen's symmetry detection

        Args:
            symprec (float, optional): Symmetry precision tollerance.
              Defaults to 0.01.
        """
        self.symprec = symprec

    def get_symmetry_label(self, structure: Structure) -> int | None:
        """Get symmetry space group number from structure

        Args:
            structure (Structure): input structure

        Returns:
            int: space group number
        """
        try:
            sga = SpacegroupAnalyzer(structure, self.symprec)
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
