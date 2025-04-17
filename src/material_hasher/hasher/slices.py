# Copyright 2025 Entalpic
# This script requires specific dependencies for proper execution.
# Install them using:
# uv pip install -r requirements_slices.txt


from pymatgen.core.structure import Structure
from slices.core import SLICES

from material_hasher.hasher.base import HasherBase


class SLICESHasher(HasherBase):
    def __init__(self, primitive_reduction: bool = False):
        """
        Initializes the SLICESHasher with the SLICES backend.
        """
        super().__init__(primitive_reduction=primitive_reduction)
        self.backend = SLICES()

    def _get_material_hash(self, structure: Structure) -> str:
        """
        Converts a pymatgen Structure to a SLICES string.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object representing the crystal structure.

        Returns
        -------
        str
            The SLICES string representation of the structure.
        """
        return self.backend.structure2SLICES(structure)
