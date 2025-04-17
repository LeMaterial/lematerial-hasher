# Copyright 2025 Entalpic
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from pymatgen.core import Structure

from material_hasher.types import StructureEquivalenceChecker
from material_hasher.utils import reduce_structure


class SimilarityMatcherBase(ABC, StructureEquivalenceChecker):
    """Abstract class for similarity matching between structures.

    Parameters
    ----------
    primitive_reduction : bool, optional
        Whether to reduce the structures to their primitive cells.
        Defaults to False.
    """

    def __init__(self, primitive_reduction: bool = False):
        self.primitive_reduction = primitive_reduction

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
        if self.primitive_reduction:
            structure1 = reduce_structure(structure1)
            structure2 = reduce_structure(structure2)
        return self._get_similarity_score(structure1, structure2)

    @abstractmethod
    def _get_similarity_score(
        self, structure1: Structure, structure2: Structure
    ) -> float:
        """Returns a similarity score between two structures.

        Should be implemented by the subclass.

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
        if self.primitive_reduction:
            structure1 = reduce_structure(structure1)
            structure2 = reduce_structure(structure2)
        return self._is_equivalent(structure1, structure2, threshold)

    @abstractmethod
    def _is_equivalent(
        self,
        structure1: Structure,
        structure2: Structure,
        threshold: Optional[float] = None,
    ) -> bool:
        """Returns True if the two structures are equivalent according to the
        implemented algorithm.
        Uses a threshold to determine equivalence if provided and the algorithm
        does not have a built-in threshold.

        Should be implemented by the subclass.

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
        """Returns a matrix of similarity scores between structures.

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
