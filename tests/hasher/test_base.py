import numpy as np
import pytest
from material_hasher.hasher.base import HasherBase
from pymatgen.core import Lattice, Structure


class SimpleHasher(HasherBase):
    """A simple concrete implementation of HasherBase for testing."""

    def get_material_hash(self, structure: Structure) -> str:
        """Simple hash based on number of sites and species."""
        species = sorted([str(site.specie) for site in structure])
        return f"{len(structure)}_{'-'.join(species)}"


@pytest.fixture
def simple_hasher():
    return SimpleHasher()


@pytest.fixture
def simple_structure():
    lattice = Lattice.cubic(4.0)
    species = ["Si", "Si"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    return Structure(lattice, species, coords)


@pytest.fixture
def different_structure():
    lattice = Lattice.cubic(4.0)
    species = ["Si", "Ge"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    return Structure(lattice, species, coords)


def test_get_material_hash(simple_hasher, simple_structure):
    """Test that get_material_hash returns expected format."""
    hash_value = simple_hasher.get_material_hash(simple_structure)
    assert isinstance(hash_value, str)
    assert hash_value == "2_Si-Si"


def test_is_equivalent_same_structure(simple_hasher, simple_structure):
    """Test that identical structures are equivalent."""
    assert simple_hasher.is_equivalent(simple_structure, simple_structure)


def test_is_equivalent_different_structures(
    simple_hasher, simple_structure, different_structure
):
    """Test that different structures are not equivalent."""
    assert not simple_hasher.is_equivalent(simple_structure, different_structure)


def test_get_materials_hashes(simple_hasher, simple_structure, different_structure):
    """Test getting hashes for multiple structures."""
    structures = [simple_structure, different_structure]
    hashes = simple_hasher.get_materials_hashes(structures)

    assert len(hashes) == 2
    assert hashes[0] == "2_Si-Si"
    assert hashes[1] == "2_Ge-Si"


def test_get_pairwise_equivalence(simple_hasher, simple_structure, different_structure):
    """Test pairwise equivalence matrix generation."""
    structures = [simple_structure, different_structure]
    equivalence_matrix = simple_hasher.get_pairwise_equivalence(structures)

    expected = np.array([[True, False], [False, True]])

    assert isinstance(equivalence_matrix, np.ndarray)
    assert equivalence_matrix.shape == (2, 2)
    assert np.array_equal(equivalence_matrix, expected)


def test_get_pairwise_equivalence_single_structure(simple_hasher, simple_structure):
    """Test pairwise equivalence matrix for a single structure."""
    structures = [simple_structure]
    equivalence_matrix = simple_hasher.get_pairwise_equivalence(structures)

    expected = np.array([[True]])

    assert isinstance(equivalence_matrix, np.ndarray)
    assert equivalence_matrix.shape == (1, 1)
    assert np.array_equal(equivalence_matrix, expected)


def test_get_pairwise_equivalence_symmetry(
    simple_hasher, simple_structure, different_structure
):
    """Test that the equivalence matrix is symmetric."""
    structures = [simple_structure, different_structure]
    equivalence_matrix = simple_hasher.get_pairwise_equivalence(structures)

    assert np.array_equal(equivalence_matrix, equivalence_matrix.T)
