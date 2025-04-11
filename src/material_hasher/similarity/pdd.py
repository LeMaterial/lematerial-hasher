# Copyright 2025 Entalpic
from typing import Optional

import numpy as np
from amd import PDD, PeriodicSet
from pymatgen.core import Structure
from scipy.stats import wasserstein_distance


from material_hasher.similarity.base import SimilarityMatcherBase


class PointwiseDistanceDistributionSimilarity(SimilarityMatcherBase):
    def __init__(self, cutoff: float = 100.0, threshold: float = 1e-3):
        """
        Initialize the PDD Generator.

        Parameters:
        cutoff (float): Cutoff distance for PDD calculation. Default is 100.
        threshold (float): Threshold for PDD comparison of two crystals. Default is 1e-3.
        """
        self.cutoff = int(cutoff)  # Ensure cutoff is an integer
        self.threshold = threshold

    def periodicset_from_structure(self, structure: Structure) -> PeriodicSet:
        """Convert a pymatgen Structure object to a PeriodicSet.

        Parameters
        ----------
        structure : pymatgen.Structure
            A pymatgen Structure object representing a crystal.

        Returns
        -------
        :class:`amd.PeriodicSet`
            Represents the crystal as a periodic set, consisting of a finite
            set of points (motif) and lattice (unit cell).

        Raises
        ------
        ValueError
            Raised if the structure has no valid sites.
        """

        # Unit cell
        cell = np.array(structure.lattice.matrix)

        # Coordinates and atomic numbers
        coords = np.array(structure.cart_coords)
        atomic_numbers = np.array([site.specie.number for site in structure.sites])

        # Check if the resulting motif is valid
        if len(coords) == 0:
            raise ValueError("The structure has no valid sites after filtering.")

        # Map coordinates to the unit cell (fractional positions mod 1)
        frac_coords = np.mod(structure.lattice.get_fractional_coords(coords), 1)

        motif = frac_coords

        return PeriodicSet(
            motif=motif,
            cell=cell,
            types=atomic_numbers,
        )

    def get_material_hash(self, structure: Structure) -> str:
        """
        Generate a hashed string for a single pymatgen structure based on its
        Point-wise Distance Distribution (PDD).

        Parameters
        ----------
        structure : pymatgen.Structure
            A pymatgen Structure object representing the crystal structure.

        Returns
        -------
        str
            A SHA256 hash string generated from the calculated PDD.
        """
        periodic_set = self.periodicset_from_structure(structure)

        pdd = PDD(
            periodic_set, int(self.cutoff), collapse=False
        )  # Ensure cutoff is an integer, without collapsing similar rows

        return pdd
    
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

        hash1 = self.get_material_hash(structure1)
        hash2 = self.get_material_hash(structure2)

        if hash1.shape != hash2.shape:
            return np.inf  # Not comparable

        return wasserstein_distance(hash1, hash2)
    
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
        PDD hashes are numpy arrays and are considered equivalent if the
        Wasserstein distance between them is less than a given threshold.

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

        if threshold is None:
            threshold = self.threshold

        distance = self.get_similarity_score(structure1, structure2)
        return distance <= threshold

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

        all_hashes = self.get_materials_hashes(structures)
        equivalence_matrix = np.zeros((len(all_hashes), len(all_hashes)), dtype=bool)

        # Fill triu + diag
        for i, hash1 in enumerate(all_hashes):
            for j, hash2 in enumerate(all_hashes):
                if i <= j:
                    equivalence_matrix[i, j] = self.is_hash_equivalent(
                        hash1, hash2, threshold
                    )

        # Fill tril
        equivalence_matrix = equivalence_matrix | equivalence_matrix.T

        return equivalence_matrix
