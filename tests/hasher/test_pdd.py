import numpy as np
import pytest
from amd import PeriodicSet
from material_hasher.hasher.pdd import PointwiseDistanceDistributionHasher
from pymatgen.core import Lattice, Structure


@pytest.fixture
def pdd_hasher():
    return PointwiseDistanceDistributionHasher(cutoff=10.0)


@pytest.fixture
def si_structure():
    lattice = Lattice.cubic(5.43)
    species = ["Si"] * 8
    coords = [
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
        [0.0, 0.5, 0.5],
        [0.25, 0.75, 0.75],
        [0.5, 0.0, 0.5],
        [0.75, 0.25, 0.75],
        [0.5, 0.5, 0.0],
        [0.75, 0.75, 0.25],
    ]
    return Structure(lattice, species, coords)


def test_periodicset_conversion(pdd_hasher, si_structure):
    """Test conversion from pymatgen Structure to PeriodicSet."""
    periodic_set = pdd_hasher.periodicset_from_structure(si_structure)

    assert isinstance(periodic_set, PeriodicSet)
    assert len(periodic_set.motif) == len(si_structure)
    assert all(n == 14 for n in periodic_set.types)  # Si atomic number is 14


def test_hash_is_string(pdd_hasher, si_structure):
    """Test that the hash is a string with expected format."""
    hash_value = pdd_hasher.get_material_hash(si_structure)

    assert isinstance(hash_value, str)
    assert len(hash_value) == 64  # SHA256 hash length


def test_hash_consistency(pdd_hasher, si_structure):
    """Test that the same structure always gets the same hash."""
    hash1 = pdd_hasher.get_material_hash(si_structure)
    hash2 = pdd_hasher.get_material_hash(si_structure)

    assert hash1 == hash2


def test_equivalent_structures(pdd_hasher, si_structure):
    """Test that equivalent structures are identified as such."""
    # Create a shifted version of the same structure
    shifted_structure = si_structure.copy()
    shifted_structure.translate_sites(range(len(shifted_structure)), [0.5, 0.5, 0.5])

    assert pdd_hasher.is_equivalent(si_structure, shifted_structure)


def test_different_cutoffs(si_structure):
    """Test that significantly different cutoffs give different hashes."""
    hasher1 = PointwiseDistanceDistributionHasher(cutoff=5.0)
    hasher2 = PointwiseDistanceDistributionHasher(cutoff=10.0)

    hash1 = hasher1.get_material_hash(si_structure)
    hash2 = hasher2.get_material_hash(si_structure)

    assert hash1 != hash2


def test_empty_structure():
    """Test that empty structure raises ValueError."""
    hasher = PointwiseDistanceDistributionHasher()
    empty_structure = Structure(Lattice.cubic(1.0), [], [])

    with pytest.raises(ValueError):
        hasher.periodicset_from_structure(empty_structure)


def test_pairwise_equivalence(pdd_hasher, si_structure):
    """Test pairwise equivalence matrix generation."""
    # Create a different structure
    different_structure = Structure(
        Lattice.cubic(5.43),
        ["Si"] * 4,
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )

    structures = [si_structure, different_structure]
    equivalence_matrix = pdd_hasher.get_pairwise_equivalence(structures)

    expected = np.array([[True, False], [False, True]])

    assert isinstance(equivalence_matrix, np.ndarray)
    assert equivalence_matrix.shape == (2, 2)
    assert np.array_equal(equivalence_matrix, expected)
