# Copyright 2025 Entalpic
import os
from typing import Optional

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from material_hasher.dataset_store import DatasetStore
from material_hasher.hasher.base import HasherBase
from material_hasher.similarity.base import SimilarityMatcherBase


class DummyHasher(HasherBase):
    """A dummy hasher that just uses the number of sites as a hash."""

    def get_material_hash(self, structure: Structure) -> str:
        return str(len(structure))


class DummySimilarityMatcher(SimilarityMatcherBase):
    """A dummy similarity matcher that compares number of sites."""

    def __init__(self, threshold: float = 0.49):
        self.threshold = threshold

    def get_similarity_score(
        self, structure1: Structure, structure2: Structure
    ) -> float:
        # Return the absolute difference between the number of sites
        return 1 / (np.abs(len(structure1) - len(structure2)) + 1)

    def get_similarity_embeddings(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> float:
        return 1 / (np.abs(embeddings1 - embeddings2) + 1)

    def is_equivalent(
        self,
        structure1: Structure,
        structure2: Structure,
        threshold: Optional[float] = None,
    ) -> bool:
        score = self.get_similarity_score(structure1, structure2)
        return score >= (threshold if threshold is not None else self.threshold)

    def get_pairwise_equivalence(
        self, structures: list[Structure], threshold: Optional[float] = None
    ) -> np.ndarray:
        n = len(structures)
        result = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                result[i, j] = self.is_equivalent(
                    structures[i], structures[j], threshold
                )
        return result

    def get_structure_embeddings(self, structure: Structure) -> np.ndarray:
        # Return a 1D array with the number of sites
        return np.array([len(structure)], dtype=float)


@pytest.fixture
def simple_structures():
    """Create a few simple test structures."""
    lattice = Lattice.cubic(1.0)
    structures = [
        Structure(lattice, ["H"], [[0, 0, 0]]),  # 1 site
        Structure(lattice, ["H", "He"], [[0, 0, 0], [0.5, 0.5, 0.5]]),  # 2 sites
        Structure(lattice, ["H"], [[0, 0, 0]]),  # 1 site
    ]
    return structures


def test_hasher_store(simple_structures):
    """Test storing and comparing structures using the dummy hasher."""
    store = DatasetStore(DummyHasher)

    # Store first two structures
    store.store_embeddings(simple_structures[:2])

    # Compare third structure (should match first but not second)
    results = store.is_equivalent(simple_structures[2])
    assert len(results) == 2
    assert results[0]
    assert not results[1]


def test_similarity_store(simple_structures):
    """Test storing and comparing structures using the dummy similarity matcher."""
    store = DatasetStore(DummySimilarityMatcher, {"threshold": 0.95})

    store.store_embeddings(simple_structures[:2])

    results = store.is_equivalent(simple_structures[2])
    assert len(results) == 2
    assert results[0]
    assert not results[1]


def test_save_load_hasher(simple_structures, tmp_path):
    """Test saving and loading a hasher store."""
    save_path = os.path.join(tmp_path, "store.npy")

    # Create and save store
    store = DatasetStore(DummyHasher)
    store.store_embeddings(simple_structures[:2])
    store.save(save_path)

    # Load store and verify
    loaded_store = DatasetStore.load(save_path, DummyHasher)
    results = loaded_store.is_equivalent(simple_structures[2])
    assert len(results) == 2
    assert results[0]  # Should match first structure (1 site)
    assert not results[1]  # Should not match second structure (2 sites)


def test_save_load_similarity(simple_structures, tmp_path):
    """Test saving and loading a similarity matcher store."""
    save_path = os.path.join(tmp_path, "store.npy")

    store = DatasetStore(DummySimilarityMatcher, {"threshold": 0.95})
    store.store_embeddings(simple_structures[:2])
    store.save(save_path)

    loaded_store = DatasetStore.load(
        save_path, DummySimilarityMatcher, {"threshold": 0.95}
    )
    results = loaded_store.is_equivalent(simple_structures[2])
    assert len(results) == 2
    assert results[0]
    assert not results[1]


def test_reset(simple_structures):
    """Test resetting the store."""
    store = DatasetStore(DummyHasher)

    # Store structures and verify
    store.store_embeddings(simple_structures[:2])
    assert len(store.embeddings) == 2

    # Reset and verify
    store.reset()
    assert len(store.embeddings) == 0


def test_incompatible_checker_load(simple_structures, tmp_path):
    """Test that loading with incompatible checker raises error."""
    save_path = os.path.join(tmp_path, "store.npy")

    # Save with hasher
    store = DatasetStore(DummyHasher)
    store.store_embeddings(simple_structures[:2])
    store.save(save_path)

    # Try to load with similarity matcher
    with pytest.raises(ValueError, match="does not match provided class"):
        DatasetStore.load(save_path, DummySimilarityMatcher)
