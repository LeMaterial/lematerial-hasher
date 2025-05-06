# Copyright 2025 Entalpic
from typing import Any, Optional, TypeVar

import numpy as np
from pymatgen.core import Structure

from material_hasher.hasher.base import HasherBase
from material_hasher.similarity.base import SimilarityMatcherBase
from material_hasher.types import StructureEquivalenceChecker

EquivalenceCheckerType = TypeVar(
    "EquivalenceCheckerType", bound=HasherBase | SimilarityMatcherBase
)


class DatasetStore:
    """Stores the hashes or embedding vectors of a dataset.

    This is used for comparing structures against a reference dataset.

    Parameters
    ----------
    equivalence_checker_class : type[EquivalenceCheckerType]
        The class of the equivalence checker to use (either a hasher or similarity matcher).
    equivalence_checker_kwargs : dict[str, Any]
        Keyword arguments to pass to the equivalence checker constructor.
    """

    def __init__(
        self,
        equivalence_checker_class: type[EquivalenceCheckerType],
        equivalence_checker_kwargs: dict[str, Any] = {},
    ):
        self.equivalence_checker_class = equivalence_checker_class
        self.equivalence_checker_kwargs = equivalence_checker_kwargs
        self.equivalence_checker = self.equivalence_checker_class(
            **self.equivalence_checker_kwargs
        )
        self.embeddings: list[np.ndarray | str] = []

    @staticmethod
    def _get_structure_embedding(
        structure: Structure, equivalence_checker: StructureEquivalenceChecker
    ) -> np.ndarray | str:
        """Get the embedding or hash of a structure.

        Parameters
        ----------
        structure : Structure
            The structure to get the embedding/hash for.
        equivalence_checker : StructureEquivalenceChecker
            The equivalence checker to use.

        Returns
        -------
        np.ndarray | str
            The embedding vector for similarity matchers or hash string for hashers.
        """
        if isinstance(equivalence_checker, HasherBase):
            return equivalence_checker.get_material_hash(structure)
        elif isinstance(equivalence_checker, SimilarityMatcherBase):
            return equivalence_checker.get_structure_embeddings(structure)
        else:
            raise ValueError(
                f"Unsupported equivalence checker: {type(equivalence_checker)}"
            )

    def _get_structures_embeddings(
        self, structures: list[Structure]
    ) -> list[np.ndarray | str]:
        """Get the embeddings of a list of structures.

        Parameters
        ----------
        structures : list[Structure]
            The structures to get embeddings for.

        Returns
        -------
        list[np.ndarray | str]
            List of embeddings or hashes for each structure.
        """
        return [
            self._get_structure_embedding(structure, self.equivalence_checker)
            for structure in structures
        ]

    def compute_and_store_embeddings(self, structures: list[Structure]) -> None:
        """Compute the embeddings/hashes of the given structures and store them.

        Parameters
        ----------
        structures : list[Structure]
            The structures to store embeddings/hashes for.
        """
        self.embeddings.extend(self._get_structures_embeddings(structures))

    def store_embeddings(self, embeddings: list[np.ndarray | str]) -> None:
        """Store the embeddings/hashes of the given structures.

        Parameters
        ----------
        embeddings : list[np.ndarray | str]
            The embeddings/hashes to store.
        """
        self.embeddings.extend(embeddings)

    def is_equivalent(
        self, structure: Structure, threshold: Optional[float] = None
    ) -> list[bool]:
        """Check if a structure is equivalent to any of the stored structures.

        Parameters
        ----------
        structure : Structure
            The structure to check.
        threshold : float, optional
            Threshold for similarity matchers, by default None.

        Returns
        -------
        list[bool]
            List of boolean values indicating equivalence with each stored structure.
        """
        query_embedding = self._get_structure_embedding(
            structure, self.equivalence_checker
        )

        if isinstance(self.equivalence_checker, HasherBase):
            return [
                query_embedding == stored_embedding
                for stored_embedding in self.embeddings
            ]
        elif isinstance(self.equivalence_checker, SimilarityMatcherBase):
            return [
                self.equivalence_checker.get_similarity_embeddings(
                    query_embedding, stored_embedding
                )
                >= (
                    threshold
                    if threshold is not None
                    else self.equivalence_checker.threshold
                )
                for stored_embedding in self.embeddings
            ]
        else:
            raise ValueError(
                f"Unsupported equivalence checker: {type(self.equivalence_checker)}"
            )

    def reset(self) -> None:
        """Reset the dataset store."""
        self.embeddings = []

    def save(self, path: str) -> None:
        """Save the dataset store to a file.

        Parameters
        ----------
        path : str
            Path to save the dataset store to.
        """
        save_data = {
            "equivalence_checker_class": self.equivalence_checker_class.__name__,
            "equivalence_checker_kwargs": self.equivalence_checker_kwargs,
            "embeddings": self.embeddings,
        }
        np.save(path, save_data, allow_pickle=True)

    @classmethod
    def load(
        cls,
        path: str,
        equivalence_checker_class: type[EquivalenceCheckerType],
        equivalence_checker_kwargs: dict[str, Any] = {},
    ) -> "DatasetStore":
        """Load the dataset store from a file.

        Parameters
        ----------
        path : str
            Path to load the dataset store from.
        equivalence_checker_class : type[EquivalenceCheckerType]
            The class of the equivalence checker to use.
        equivalence_checker_kwargs : dict[str, Any]
            Keyword arguments to pass to the equivalence checker constructor.

        Returns
        -------
        DatasetStore
            The loaded dataset store.
        """
        save_data = np.load(path, allow_pickle=True).item()

        # Verify the equivalence checker class matches
        if save_data["equivalence_checker_class"] != equivalence_checker_class.__name__:
            raise ValueError(
                f"Loaded equivalence checker class {save_data['equivalence_checker_class']} "
                f"does not match provided class {equivalence_checker_class.__name__}"
            )

        store = cls(equivalence_checker_class, equivalence_checker_kwargs)
        store.embeddings = save_data["embeddings"]
        return store
