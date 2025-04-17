# Copyright 2025 Entalpic
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from pymatgen.core import Structure

from material_hasher.types import StructureEquivalenceChecker
from material_hasher.utils import reduce_structure


class HasherBase(ABC, StructureEquivalenceChecker):
    """Abstract class for matching of the hashes between structures.

    Parameters
    ----------
    primitive_reduction : bool, optional
        Whether to reduce the structures to their primitive cells.
        Defaults to False.
    """

    def __init__(self, primitive_reduction: bool = False):
        self.primitive_reduction = primitive_reduction

    def get_material_hash(
        self,
        structure: Structure,
    ) -> str:
        """Returns a hash of the structure.

        Parameters
        ----------
        structure : Structure
            Structure to hash.

        Returns
        -------
        str
            Hash of the structure.
        """
        if self.primitive_reduction:
            structure = reduce_structure(structure)
        return self._get_material_hash(structure)

    @abstractmethod
    def _get_material_hash(
        self,
        structure: Structure,
    ) -> str:
        """Get the material hash of the structure.

        Should be implemented by the subclass.

        Parameters
        ----------
        structure : Structure
            Structure to hash.

        Returns
        -------
        str
            Hash of the structure.
        """
        pass

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

    def get_materials_hashes(
        self,
        structures: list[Structure],
    ) -> list[str]:
        """Returns a list of hashes of the structures.

        Parameters
        ----------
        structures : list[Structure]
            List of structures to hash.

        Returns
        -------
        list[str]
            List of hashes of the structures.
        """
        return [self.get_material_hash(structure) for structure in structures]

    def get_pairwise_equivalence(
        self, structures: list[Structure], threshold: Optional[float] = None
    ) -> np.ndarray:
        """Returns a matrix of equivalence between structures.

        Parameters
        ----------
        structures : list[Structure]
            List of structures to compare.
        threshold : float, optional
            Threshold to determine similarity, by default None and the
            algorithm's default threshold is used if it exists.

        Returns
        -------
        np.ndarray
            Matrix of equivalence between structures.
        """

        n = len(structures)
        equivalence_matrix = np.zeros((n, n), dtype=bool)

        # Fill triu + diag
        for i, structure1 in enumerate(structures):
            for j, structure2 in enumerate(structures):
                if i <= j:
                    equivalence_matrix[i, j] = self.is_equivalent(
                        structure1, structure2, threshold
                    )

        # Fill tril
        equivalence_matrix = equivalence_matrix | equivalence_matrix.T

        return equivalence_matrix
