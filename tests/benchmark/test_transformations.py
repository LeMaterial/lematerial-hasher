import numpy as np
import pytest
from material_hasher.benchmark.transformations import (
    get_new_structure_with_gaussian_noise,
    get_new_structure_with_isometric_strain,
    get_new_structure_with_strain,
    get_new_structure_with_symm_ops,
    get_new_structure_with_translation,
)
from pymatgen.core import Lattice, Structure


@pytest.fixture
def simple_cubic_structure():
    """Create a simple cubic structure for testing."""
    lattice = Lattice.cubic(1.0)
    coords = [[0.0, 0.0, 0.0]]
    return Structure(lattice, ["Si"], coords)


@pytest.fixture
def complex_structure():
    """Create a more complex structure with multiple atoms for testing."""
    lattice = Lattice.cubic(5.0)
    coords = [
        [0.0, 0.0, 0.0],  # Si at origin
        [0.5, 0.2, 0.3],  # O offset from center
        [0.25, 0.4, 0.1],  # H creating asymmetry
    ]
    species = ["Si", "O", "H"]
    return Structure(lattice, species, coords)


def test_gaussian_noise_transformation(simple_cubic_structure):
    """Test gaussian noise transformation."""
    # Test with small noise
    sigma = 0.0001
    transformed = get_new_structure_with_gaussian_noise(simple_cubic_structure, sigma)

    # Check that structure is modified but not too much
    assert transformed != simple_cubic_structure
    assert np.allclose(
        transformed.cart_coords, simple_cubic_structure.cart_coords, atol=sigma * 10
    )

    # Test with larger noise
    sigma = 0.1
    transformed = get_new_structure_with_gaussian_noise(simple_cubic_structure, sigma)
    assert not np.allclose(
        transformed.cart_coords, simple_cubic_structure.cart_coords, atol=sigma / 10
    )


def test_isometric_strain_transformation(simple_cubic_structure):
    """Test isometric strain transformation."""
    # Test expansion
    pct = 1.1
    transformed = get_new_structure_with_isometric_strain(simple_cubic_structure, pct)
    assert transformed.volume > simple_cubic_structure.volume
    assert np.isclose(transformed.volume, simple_cubic_structure.volume * pct)

    # Test compression
    pct = 0.9
    transformed = get_new_structure_with_isometric_strain(simple_cubic_structure, pct)
    assert transformed.volume < simple_cubic_structure.volume
    assert np.isclose(transformed.volume, simple_cubic_structure.volume * pct)


def test_strain_transformation(simple_cubic_structure):
    """Test strain transformation."""
    sigma = 0.01
    transformed = get_new_structure_with_strain(simple_cubic_structure, sigma)

    # Check that structure is modified
    assert transformed != simple_cubic_structure

    # Check that volume is changed (strain should affect volume)
    assert transformed.volume != simple_cubic_structure.volume

    # Check that atomic positions are still valid
    assert all(
        coord >= 0 and coord <= 1 for site in transformed.frac_coords for coord in site
    )


def test_translation_transformation(complex_structure):
    """Test translation transformation."""
    sigma = 0.1
    transformed = get_new_structure_with_translation(complex_structure, sigma)

    # Check that structure is modified
    assert transformed != complex_structure

    # Check that relative distances between atoms are preserved
    original_distances = complex_structure.distance_matrix
    transformed_distances = transformed.distance_matrix
    assert np.allclose(original_distances, transformed_distances)


def test_symm_ops_transformation(complex_structure):
    """Test symmetry operations transformation."""
    transformed_structures = get_new_structure_with_symm_ops(
        complex_structure, "all_symmetries_found"
    )

    # Check that we get multiple structures
    assert len(transformed_structures) > 0

    # Check that all transformed structures have the same number of sites
    assert all(len(s) == len(complex_structure) for s in transformed_structures)

    # Check that all transformed structures have the same volume
    assert all(
        np.isclose(s.volume, complex_structure.volume) for s in transformed_structures
    )

    # Check that the structures are actually different
    if len(transformed_structures) > 1:
        assert not np.allclose(
            transformed_structures[0].cart_coords, transformed_structures[1].cart_coords
        )


def test_edge_cases():
    """Test edge cases for transformations."""

    # Test with zero parameters
    structure = Structure(Lattice.cubic(1.0), ["Si"], [[0, 0, 0]])

    # Zero noise should return effectively the same structure
    transformed = get_new_structure_with_gaussian_noise(structure, 0.0)
    assert np.allclose(transformed.cart_coords, structure.cart_coords)

    # Zero strain should return the same volume
    transformed = get_new_structure_with_isometric_strain(structure, 1.0)
    assert np.isclose(transformed.volume, structure.volume)
