import numpy as np
import pytest
from material_hasher.similarity.base import SimilarityMatcherBase
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure


class PerfectMatcher(SimilarityMatcherBase):
    """A dummy matcher that only considers structures equivalent if they're identical."""

    def get_similarity_score(
        self, structure1: Structure, structure2: Structure
    ) -> float:
        """Return 0.0 for identical structures, 1.0 otherwise."""
        return 0.0 if structure1 == structure2 else 1.0

    def is_equivalent(
        self, structure1: Structure, structure2: Structure, threshold=None
    ) -> bool:
        """Return True only for identical structures."""
        return structure1 == structure2

    def get_pairwise_equivalence(
        self, structures: list[Structure], threshold=None
    ) -> np.ndarray:
        """Get pairwise equivalence matrix."""
        n = len(structures)
        matrix = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i, n):
                matrix[i, j] = self.is_equivalent(structures[i], structures[j])

        matrix = matrix | matrix.T
        return matrix


class DistanceBasedMatcher(SimilarityMatcherBase):
    """A dummy matcher that uses lattice parameter differences for similarity."""

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def get_similarity_score(
        self, structure1: Structure, structure2: Structure
    ) -> float:
        """Return the relative difference in lattice parameters."""
        a1 = structure1.lattice.a
        a2 = structure2.lattice.a
        return abs(a1 - a2) / max(a1, a2)

    def is_equivalent(
        self, structure1: Structure, structure2: Structure, threshold=None
    ) -> bool:
        """Return True if lattice parameters are within threshold."""
        threshold = threshold if threshold is not None else self.threshold
        return self.get_similarity_score(structure1, structure2) <= threshold

    def get_pairwise_equivalence(
        self, structures: list[Structure], threshold=None
    ) -> np.ndarray:
        """Get pairwise equivalence matrix."""
        n = len(structures)
        matrix = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i, n):
                matrix[i, j] = self.is_equivalent(
                    structures[i], structures[j], threshold=threshold
                )

        matrix = matrix | matrix.T
        return matrix


@pytest.fixture
def structures():
    """Create a list of test structures with different lattice parameters."""
    return [
        Structure(Lattice.cubic(4.0), ["Fe"], [[0, 0, 0]]),
        Structure(Lattice.cubic(4.0), ["Fe"], [[0, 0, 0]]),  # Identical to first
        Structure(Lattice.cubic(4.1), ["Fe"], [[0, 0, 0]]),  # Similar
        Structure(Lattice.cubic(5.0), ["Fe"], [[0, 0, 0]]),  # Different
    ]


def test_perfect_matcher(structures):
    matcher = PerfectMatcher()

    # Test similarity scores
    assert matcher.get_similarity_score(structures[0], structures[1]) == 0.0
    assert matcher.get_similarity_score(structures[0], structures[2]) == 1.0

    # Test equivalence
    assert matcher.is_equivalent(structures[0], structures[1])
    assert not matcher.is_equivalent(structures[0], structures[2])

    # Test pairwise similarity scores
    scores = matcher.get_pairwise_similarity_scores(structures[:3])
    assert scores.shape == (3, 3)
    assert scores[0, 1] == 0.0  # Identical structures
    assert scores[0, 2] == 1.0  # Different structures

    # Test pairwise equivalence
    equiv = matcher.get_pairwise_equivalence(structures[:3])
    assert equiv.shape == (3, 3)
    assert equiv[0, 1] and equiv[1, 0]  # Symmetric
    assert not equiv[0, 2] and not equiv[2, 0]  # Not equivalent


def test_distance_based_matcher(structures):
    matcher = DistanceBasedMatcher(threshold=0.05)  # 5% threshold

    # Test similarity scores
    score01 = matcher.get_similarity_score(structures[0], structures[1])
    score02 = matcher.get_similarity_score(structures[0], structures[2])
    assert score01 == 0.0  # Identical
    assert 0.0 < score02 < 0.1  # Similar but different

    # Test equivalence with different thresholds
    assert matcher.is_equivalent(
        structures[0], structures[2], threshold=0.1
    )  # Should be equivalent with 10% threshold
    assert not matcher.is_equivalent(
        structures[0], structures[2], threshold=0.01
    )  # Should not be equivalent with 1% threshold

    # Test pairwise similarity scores
    scores = matcher.get_pairwise_similarity_scores(structures)
    assert scores.shape == (4, 4)
    assert np.allclose(scores, scores.T)  # Symmetric
    assert np.all(np.diag(scores) == 0)  # Zero on diagonal

    # Test pairwise equivalence
    equiv = matcher.get_pairwise_equivalence(structures, threshold=0.05)
    assert equiv.shape == (4, 4)
    assert np.all(np.diag(equiv))  # True on diagonal
    assert equiv[0, 1] and equiv[1, 0]  # Symmetric
    assert not equiv[0, 3]  # Far structures not equivalent


def test_interface_properties():
    """Test that the similarity interface maintains expected properties."""
    matcher = DistanceBasedMatcher()
    structures = [
        Structure(Lattice.cubic(4.0), ["Fe"], [[0, 0, 0]]),
        Structure(Lattice.cubic(4.1), ["Fe"], [[0, 0, 0]]),
    ]

    # Test symmetry of similarity scores
    score_ab = matcher.get_similarity_score(structures[0], structures[1])
    score_ba = matcher.get_similarity_score(structures[1], structures[0])
    assert np.isclose(score_ab, score_ba)

    # Test symmetry of equivalence
    equiv_ab = matcher.is_equivalent(structures[0], structures[1])
    equiv_ba = matcher.is_equivalent(structures[1], structures[0])
    assert equiv_ab == equiv_ba

    # Test pairwise matrices are symmetric
    scores = matcher.get_pairwise_similarity_scores(structures)
    equiv = matcher.get_pairwise_equivalence(structures)
    assert np.allclose(scores, scores.T)
    assert np.array_equal(equiv, equiv.T)


def test_threshold_behavior():
    """Test that threshold parameter works correctly."""
    matcher = DistanceBasedMatcher(threshold=0.05)
    struct1 = Structure(Lattice.cubic(4.0), ["Fe"], [[0, 0, 0]])
    struct2 = Structure(Lattice.cubic(4.04), ["Fe"], [[0, 0, 0]])  # 1% difference

    # Default threshold (5%) should make these equivalent
    assert matcher.is_equivalent(struct1, struct2)

    # Stricter threshold should make them different
    assert not matcher.is_equivalent(struct1, struct2, threshold=0.005)

    # More lenient threshold should keep them equivalent
    assert matcher.is_equivalent(struct1, struct2, threshold=0.1)
