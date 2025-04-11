import numpy as np
import pytest
from material_hasher.benchmark.run_transformations import (
    hasher_sensitivity,
    mean_sensitivity,
    sensitivity_over_parameter_range,
)
from material_hasher.hasher.base import HasherBase
from pymatgen.core import Lattice, Structure


class DummyConstantHasher(HasherBase):
    """A dummy hasher that always returns the same hash, simulating complete insensitivity to transformations."""

    def get_material_hash(self, structure: Structure) -> str:
        return "constant_hash"


class DummyRandomHasher(HasherBase):
    """A dummy hasher that always returns different hashes, simulating complete sensitivity to transformations."""

    def get_material_hash(self, structure: Structure) -> str:
        return f"random_hash_{np.random.rand()}"


@pytest.fixture
def dummy_structure():
    """Create a simple cubic structure for testing."""
    lattice = Lattice.cubic(1.0)
    coords = [[0.0, 0.0, 0.0]]
    return Structure(lattice, ["Si"], coords)


@pytest.fixture
def constant_hasher():
    """Create a constant hasher instance."""
    return DummyConstantHasher()


@pytest.fixture
def random_hasher():
    """Create a random hasher instance."""
    return DummyRandomHasher()


def test_hasher_sensitivity_constant(dummy_structure, constant_hasher):
    """Test that constant hasher always returns 1.0 sensitivity."""
    transformed_structures = [dummy_structure.copy(), dummy_structure.copy()]
    sensitivity = hasher_sensitivity(
        dummy_structure, transformed_structures, constant_hasher
    )
    assert sensitivity == 1.0


def test_hasher_sensitivity_random(dummy_structure, random_hasher):
    """Test that random hasher returns very low sensitivity."""
    transformed_structures = [dummy_structure.copy(), dummy_structure.copy()]
    sensitivity = hasher_sensitivity(
        dummy_structure, transformed_structures, random_hasher
    )
    assert sensitivity == 0.0


def test_mean_sensitivity(dummy_structure, constant_hasher, random_hasher):
    """Test mean sensitivity calculation with both hashers."""
    structures = [dummy_structure.copy() for _ in range(3)]
    test_case = "gaussian_noise"
    parameter = ("sigma", 0.0001)

    # Test constant hasher
    mean_sens_constant = mean_sensitivity(
        structures, test_case, parameter, constant_hasher
    )
    assert mean_sens_constant == 1.0

    # Test random hasher
    mean_sens_random = mean_sensitivity(structures, test_case, parameter, random_hasher)
    assert mean_sens_random == 0.0


def test_sensitivity_over_parameter_range(
    dummy_structure, constant_hasher, random_hasher
):
    """Test sensitivity over parameter range with both hashers."""
    structures = [dummy_structure.copy() for _ in range(3)]
    test_case = "gaussian_noise"

    # Test constant hasher
    results_constant = sensitivity_over_parameter_range(
        structures, test_case, constant_hasher
    )
    assert all(value == 1.0 for value in results_constant.values())

    # Test random hasher
    results_random = sensitivity_over_parameter_range(
        structures, test_case, random_hasher
    )
    assert all(value == 0.0 for value in results_random.values())
