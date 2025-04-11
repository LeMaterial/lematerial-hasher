import numpy as np
import pytest
from material_hasher.similarity.structure_matchers import PymatgenStructureSimilarity
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure


@pytest.fixture
def simple_cubic_structure():
    """Create a simple cubic structure."""
    lattice = Lattice.cubic(4.0)
    species = ["Fe"]
    coords = [[0, 0, 0]]
    return Structure(lattice, species, coords)


@pytest.fixture
def slightly_distorted_cubic():
    """Create a slightly distorted cubic structure."""
    lattice = Lattice.cubic(4.0 * 1.02)
    species = ["Fe"]
    coords = [[0.015, 0.015, 0.015]]
    return Structure(lattice, species, coords)


@pytest.fixture
def different_structure():
    """Create a different structure (BCC instead of simple cubic)."""
    lattice = Lattice.cubic(4.0)
    species = ["Fe", "Fe"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]  # BCC structure
    return Structure(lattice, species, coords)


def test_structure_equivalence(
    simple_cubic_structure, slightly_distorted_cubic, different_structure
):
    matcher = PymatgenStructureSimilarity(tolerance=0.02)

    # Test similar structures
    assert matcher.is_equivalent(simple_cubic_structure, slightly_distorted_cubic)

    # Test different structures
    assert not matcher.is_equivalent(simple_cubic_structure, different_structure)

    # Test self-equivalence
    assert matcher.is_equivalent(simple_cubic_structure, simple_cubic_structure)


def test_similarity_score(
    simple_cubic_structure, slightly_distorted_cubic, different_structure
):
    matcher = PymatgenStructureSimilarity(tolerance=0.02)

    # Test similar structures - should have a small RMSD
    score_similar = matcher.get_similarity_score(
        simple_cubic_structure, slightly_distorted_cubic
    )
    assert score_similar > 0.9  # Small RMSD for similar structures

    # Test different structures - should have a larger RMSD
    score_different = matcher.get_similarity_score(
        simple_cubic_structure, different_structure
    )
    assert (
        score_different < score_similar
    )  # Different structures should have larger RMSD


def test_pairwise_equivalence(
    simple_cubic_structure, slightly_distorted_cubic, different_structure
):
    matcher = PymatgenStructureSimilarity(tolerance=0.02)
    structures = [simple_cubic_structure, slightly_distorted_cubic, different_structure]

    matrix = matcher.get_pairwise_equivalence(structures)

    # Expected matrix shape
    assert matrix.shape == (3, 3)

    # Matrix should be symmetric
    assert np.array_equal(matrix, matrix.T)

    # Diagonal should be True (self-equivalence)
    assert np.all(np.diag(matrix))

    # First two structures should be equivalent
    assert matrix[0, 1] and matrix[1, 0]

    # Different structure should not be equivalent to others
    assert not matrix[0, 2] and not matrix[1, 2]
