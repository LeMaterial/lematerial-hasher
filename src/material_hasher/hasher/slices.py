# Copyright 2025 Entalpic
# This script requires specific dependencies for proper execution.
# Install them using:
# uv pip install -r requirements_slices.txt


from pymatgen.core.structure import Structure

try:
    from slices.core import SLICES
except ImportError:
    import warnings

    warnings.warn(
        "Failed to import SLICES. If you would like to use this module, please consider running uv pip install -r requirements_slices.txt",
        ImportWarning,
    )


from material_hasher.hasher.base import HasherBase


class SLICESHasher(HasherBase):
    def __init__(self):
        """
        Initializes the SLICESHasher with the SLICES backend.
        """
        self.backend = SLICES()

    def get_material_hash(self, structure: Structure) -> str:
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
