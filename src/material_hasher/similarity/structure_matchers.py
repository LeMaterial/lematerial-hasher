# Copyright 2025 Entalpic
from typing import List, Optional

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure

from material_hasher.similarity.base import SimilarityMatcherBase


class PymatgenStructureSimilarity(SimilarityMatcherBase):
    """Implementation of the StructureMatcherBase using pymatgen's StructureMatcher."""

    def __init__(self, tolerance=0.01):
        self.tolerance = tolerance
        self.matcher = StructureMatcher(ltol=tolerance)
        self.structures: List[Structure] = []

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
        return self.matcher.fit(structure1, structure2)

    def get_similarity_score(
        self, structure1: Structure, structure2: Structure
    ) -> float:
        """
        Calculate a similarity score based on RMSD. Lower RMSD values indicate
        higher similarity. A perfect match gives a score of 1.0.

        Parameters
        ----------
        structure1 : Structure
            First structure to compare.
        structure2 : Structure
            Second structure to compare.

        Returns
        -------
        float
            Similarity score between the two structures.
        """
        dist = self.matcher.get_rms_dist(structure1, structure2)

        if dist is None:
            return 0.0

        return 1 - dist[0]

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
        matrix = np.zeros((n, n), dtype=bool)

        # Fill triu + diag
        for i in range(n):
            for j in range(i, n):
                matrix[i, j] = self.is_equivalent(
                    structures[i], structures[j], threshold
                )

        # Fill tril
        matrix = matrix.astype(int)
        matrix = matrix + matrix.T - np.diag(np.diag(matrix))  # probably just an or
        matrix = matrix.astype(bool)

        return matrix
