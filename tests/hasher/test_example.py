import pytest
from material_hasher.hasher.example import SimpleCompositionHasher
from pymatgen.core import Lattice, Structure


@pytest.fixture
def composition_hasher():
    return SimpleCompositionHasher()


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
def sio2_structure():
    lattice = Lattice.hexagonal(4.91, 5.40)
    species = ["Si"] * 3 + ["O"] * 6
    coords = [
        [0.4697, 0.0000, 0.0000],
        [0.0000, 0.4697, 0.6667],
        [0.5303, 0.5303, 0.3333],
        [0.4135, 0.2669, 0.1191],
        [0.2669, 0.4135, 0.5476],
        [0.7331, 0.1466, 0.7857],
        [0.5865, 0.8534, 0.8810],
        [0.8534, 0.5865, 0.4524],
        [0.1466, 0.7331, 0.2143],
    ]
    return Structure(lattice, species, coords)


def test_get_material_hash_si(composition_hasher, si_structure):
    """Test that silicon structure returns Si as hash."""
    hash_value = composition_hasher.get_material_hash(si_structure)
    assert hash_value == "Si"


def test_get_material_hash_sio2(composition_hasher, sio2_structure):
    """Test that silicon dioxide structure returns SiO2 as hash."""
    hash_value = composition_hasher.get_material_hash(sio2_structure)
    assert hash_value == "SiO2"


def test_is_equivalent_same_composition(composition_hasher, sio2_structure):
    """Test that structures with same composition are equivalent."""
    # Create a different SiO2 structure with same composition
    lattice = Lattice.cubic(4.91)
    species = ["Si"] * 3 + ["O"] * 6
    coords = [
        [x / 8, y / 8, z / 8] for x in range(3) for y in range(3) for z in range(3)
    ][:9]
    different_sio2 = Structure(lattice, species, coords)

    assert composition_hasher.is_equivalent(sio2_structure, different_sio2)


def test_is_equivalent_different_composition(
    composition_hasher, si_structure, sio2_structure
):
    """Test that structures with different compositions are not equivalent."""
    assert not composition_hasher.is_equivalent(si_structure, sio2_structure)


def test_get_materials_hashes_multiple(
    composition_hasher, si_structure, sio2_structure
):
    """Test getting hashes for multiple structures."""
    structures = [si_structure, sio2_structure]
    hashes = composition_hasher.get_materials_hashes(structures)

    assert len(hashes) == 2
    assert hashes[0] == "Si"
    assert hashes[1] == "SiO2"
