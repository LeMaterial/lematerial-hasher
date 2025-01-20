# This script requires specific dependencies for proper execution.
# Install them using:
# uv pip install -r requirements_slices.txt


from typing import Optional

from pymatgen.core.structure import Lattice, Structure
from slices.core import SLICES

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

    def is_equivalent(
        self,
        structure1: Structure,
        structure2: Structure,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Check if two structures are similar based on the StructureMatcher of
        pymatgen. The StructureMatcher uses a similarity algorithm based on the
        maximum common subgraph isomorphism and the Jaccard index of the sites.

        Parameters
        ----------
        structure1 : Structure
            First structure to compare.
        structure2 : Structure
            Second structure to compare.

        Returns
        -------
        bool
            True if the two structures are similar, False otherwise.
        """

        hash_structure1 = self.get_material_hash(structure1)
        hash_structure2 = self.get_material_hash(structure2)

        return hash_structure1 == hash_structure2