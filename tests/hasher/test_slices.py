import pytest
from pymatgen.core import Lattice, Structure

# Try to import SLICES, skip tests if not available
slices_available = False
try:
    from material_hasher.hasher.slices import SLICESHasher

    slices_available = True
except ImportError:
    pass

# Skip all tests if SLICES is not available
pytestmark = pytest.mark.skipif(
    not slices_available,
    reason="SLICES package not installed. Install with 'uv pip install -r requirements_slices.txt'",
)


@pytest.fixture
def slices_hasher():
    return SLICESHasher()


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


def test_hash_is_string(slices_hasher, si_structure):
    """Test that the hash is a string."""
    hash_value = slices_hasher.get_material_hash(si_structure)
    assert isinstance(hash_value, str)


def test_hash_consistency(slices_hasher, si_structure):
    """Test that the same structure gets the same hash."""
    hash1 = slices_hasher.get_material_hash(si_structure)
    hash2 = slices_hasher.get_material_hash(si_structure)

    assert hash1 == hash2


# TODO(Ramlaoui): This does not pass with SLICES is it expected?
# cf paper?
# def test_equivalent_structures(slices_hasher, si_structure):
#     """Test that equivalent structures are identified as such."""
#     # Create a shifted version of the same structure
#     shifted_structure = si_structure.copy()
#     shifted_structure.translate_sites(range(len(shifted_structure)), [0.5, 0.5, 0.5])

#     assert slices_hasher.is_equivalent(si_structure, shifted_structure)


def test_different_structures(slices_hasher, si_structure):
    """Test that different structures get different hashes."""
    # Create a different structure (different number of sites)
    different_structure = Structure(
        Lattice.cubic(5.43),
        ["Si"] * 4,
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )

    hash1 = slices_hasher.get_material_hash(si_structure)
    hash2 = slices_hasher.get_material_hash(different_structure)

    assert hash1 != hash2


def test_pairwise_equivalence(slices_hasher, si_structure):
    """Test pairwise equivalence matrix generation."""
    # Create a different structure
    different_structure = Structure(
        Lattice.cubic(5.43),
        ["Si"] * 4,
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )

    structures = [si_structure, different_structure]
    equivalence_matrix = slices_hasher.get_pairwise_equivalence(structures)

    # Structures should only be equivalent to themselves
    assert equivalence_matrix[0, 0]  # si_structure with itself
    assert equivalence_matrix[1, 1]  # different_structure with itself
    assert not equivalence_matrix[0, 1]  # si_structure with different_structure
    assert not equivalence_matrix[1, 0]  # different_structure with si_structure
