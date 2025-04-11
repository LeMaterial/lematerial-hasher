# Copyright 2025 Entalpic
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from pymatgen.core import Structure

from material_hasher.types import StructureEquivalenceChecker


class SimilarityMatcherBase(ABC, StructureEquivalenceChecker):
    """Abstract class for similarity matching between structures."""

    @abstractmethod
    def get_similarity_score(
        self, structure1: Structure, structure2: Structure
    ) -> float:
        """Returns a similarity score between two structures.

        Parameters
        ----------
        structure1 : Structure
            First structure to compare.
        structure2 : Structure

        Returns
        -------
        float
            Similarity score between the two structures.
        """
        pass

    @abstractmethod
    def is_equivalent(
        self,
        structure1: Structure,
        structure2: Structure,
        threshold: Optional[float] = None,
    ) -> bool:
        """Returns True if the two structures are equivalent according to the
        implemented algorithm.
        Uses a threshold to determine equivalence if provided and the algorithm
        does not have a built-in threshold.

        Parameters
        ----------
        structure1 : Structure
            First structure to compare.
        structure2 : Structure
            Second structure to compare.
        threshold : float, optional
            Threshold to determine similarity, by default None and the
            algorithm's default threshold is used if it exists.

        Returns
        -------
        bool
            True if the two structures are similar, False otherwise.
        """
        pass

    def get_pairwise_similarity_scores(
        self,
        structures: list[Structure],
    ) -> np.ndarray:
        """Returns a matrix $M$ of equivalence between structures.
        $M$ is a boolean symmetric matrix where entry $M[i, j]$ is ``True``
        if the hash of entries $i$ and $j$ are equivalent (up to ``threshold``)
        and ``False`` otherwise.

        Parameters
        ----------
        structures : list[Structure]
            List of structures to compare.

        Returns
        -------
        np.ndarray
            Matrix of similarity scores between structures.
        """

        n = len(structures)
        scores = np.zeros((n, n))

        # Fill triu + diag
        for i, structure1 in enumerate(structures):
            for j, structure2 in enumerate(structures):
                if i <= j:
                    scores[i, j] = self.get_similarity_score(structure1, structure2)

        # Fill tril
        scores = scores + scores.T - np.diag(np.diag(scores))

        return scores

    @abstractmethod
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
        pass
