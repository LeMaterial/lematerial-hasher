import pytest
from material_hasher.hasher.entalpic import (
    EntalpicMaterialsHasher,
    ShortenedEntalpicMaterialsHasher,
)
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Lattice, Structure


@pytest.fixture
def entalpic_hasher():
    return EntalpicMaterialsHasher()


@pytest.fixture
def shortened_hasher():
    return ShortenedEntalpicMaterialsHasher()


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


@pytest.fixture
def ge_structure():
    lattice = Lattice.cubic(5.43)
    species = ["Ge"] * 8
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


def test_entalpic_hasher_initialization():
    """Test different initialization parameters."""
    # Test with different bonding algorithm
    hasher1 = EntalpicMaterialsHasher(
        bonding_algorithm=VoronoiNN, bonding_kwargs={"tol": 0.1}
    )
    assert isinstance(hasher1.bonding_algorithm, type)
    assert hasher1.bonding_algorithm == VoronoiNN

    # Test with different bonding kwargs
    hasher2 = EntalpicMaterialsHasher(bonding_kwargs={"tol": 0.1})
    assert hasher2.bonding_kwargs == {"tol": 0.1}

    # Test with composition disabled
    hasher3 = EntalpicMaterialsHasher(include_composition=False)
    assert not hasher3.include_composition


def test_entalpic_hash_components(entalpic_hasher, si_structure):
    """Test that all hash components are present."""
    data = entalpic_hasher.get_entalpic_materials_data(si_structure)

    assert "bonding_graph_hash" in data
    assert "symmetry_label" in data
    assert "composition" in data

    # Test types
    assert isinstance(data["bonding_graph_hash"], str)
    assert isinstance(data["symmetry_label"], int)
    assert isinstance(data["composition"], str)


def test_shortened_vs_full_hash(shortened_hasher, entalpic_hasher, si_structure):
    """Test that shortened hash is different from full hash."""
    short_hash = shortened_hasher.get_material_hash(si_structure)
    full_hash = entalpic_hasher.get_material_hash(si_structure)

    assert short_hash != full_hash
    assert len(short_hash.split("_")) < len(full_hash.split("_"))


def test_hash_consistency(entalpic_hasher, si_structure):
    """Test that the same structure always gets the same hash."""
    hash1 = entalpic_hasher.get_material_hash(si_structure)
    hash2 = entalpic_hasher.get_material_hash(si_structure)

    assert hash1 == hash2


def test_equivalent_structures(entalpic_hasher, si_structure):
    """Test that equivalent structures are identified as such."""
    # Create a shifted version of the same structure
    shifted_structure = si_structure.copy()
    shifted_structure.translate_sites(range(len(shifted_structure)), [0.5, 0.5, 0.5])

    assert entalpic_hasher.is_equivalent(si_structure, shifted_structure)


def test_symmetry_detection(entalpic_hasher, si_structure):
    """Test that symmetry is correctly detected."""
    data = entalpic_hasher.get_entalpic_materials_data(si_structure)

    # Silicon has space group 227 (Fd-3m)
    assert data["symmetry_label"] == 227


def test_different_materials_have_different_hashes(
    entalpic_hasher, si_structure, ge_structure
):
    """Test that different materials have different hashes."""
    assert entalpic_hasher.get_material_hash(
        si_structure
    ) != entalpic_hasher.get_material_hash(ge_structure)

    noised_ge_structure = ge_structure.copy()

    # perturb the structure by 0.5 Angstrom (a lot!)
    noised_ge_structure.perturb(0.5)
    assert entalpic_hasher.get_material_hash(
        si_structure
    ) != entalpic_hasher.get_material_hash(noised_ge_structure)
