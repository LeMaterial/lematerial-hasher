import numpy as np
import pandas as pd
import pytest
from material_hasher.benchmark.run_disordered import (
    benchmark_disordered_structures,
    run_group_structures_benchmark,
)
from material_hasher.hasher.base import HasherBase
from pymatgen.core import Lattice, Structure


class DummyConstantHasher(HasherBase):
    """A dummy hasher that always returns the same hash, simulating complete insensitivity."""

    def get_material_hash(self, structure: Structure) -> str:
        return "constant_hash"

    def is_equivalent(self, structure1: Structure, structure2: Structure) -> bool:
        return True


class DummyRandomHasher(HasherBase):
    """A dummy hasher that always returns different hashes, simulating complete sensitivity."""

    def get_material_hash(self, structure: Structure) -> str:
        return f"random_hash_{np.random.rand()}"

    def is_equivalent(self, structure1: Structure, structure2: Structure) -> bool:
        return False


@pytest.fixture
def dummy_structures():
    """Create a list of simple cubic structures for testing."""
    lattice = Lattice.cubic(1.0)
    structures = []
    for i in range(5):
        coords = [[0.0, 0.0, float(i) / 10]]  # Slightly different z coordinates
        structures.append(Structure(lattice, ["Si"], coords))
    return structures


@pytest.fixture
def constant_hasher():
    """Create a constant hasher instance."""
    return DummyConstantHasher()


@pytest.fixture
def random_hasher():
    """Create a random hasher instance."""
    return DummyRandomHasher()


def test_run_group_structures_benchmark_constant(dummy_structures, constant_hasher):
    """Test group structures benchmark with constant hasher."""
    metrics = run_group_structures_benchmark(
        constant_hasher,
        "test_group",
        dummy_structures,
        n_pick_random=3,
        n_random_structures=2,
        seeds=[0],
    )

    # Constant hasher should always consider structures equivalent
    assert len(metrics["success_rate"]) == 1  # One seed
    assert metrics["success_rate"][0] == 1.0


def test_run_group_structures_benchmark_random(dummy_structures, random_hasher):
    """Test group structures benchmark with random hasher."""
    metrics = run_group_structures_benchmark(
        random_hasher,
        "test_group",
        dummy_structures,
        n_pick_random=3,
        n_random_structures=2,
        seeds=[0],
    )

    # Random hasher should never consider structures equivalent
    assert len(metrics["success_rate"]) == 1  # One seed
    assert metrics["success_rate"][0] == 0.0


def test_run_group_structures_benchmark_small_group(dummy_structures, constant_hasher):
    """Test group structures benchmark with a small group (less than n_pick_random)."""
    small_group = dummy_structures[:2]  # Only use 2 structures
    metrics = run_group_structures_benchmark(
        constant_hasher,
        "small_group",
        small_group,
        n_pick_random=3,  # Larger than group size
        n_random_structures=2,
        seeds=[0],
    )

    assert len(metrics["success_rate"]) == 1
    assert metrics["success_rate"][0] == 1.0


@pytest.fixture
def mock_disordered_structures(dummy_structures):
    """Create a mock dictionary of disordered structures for testing."""
    return {
        "group1": dummy_structures[:3],
        "group2": dummy_structures[3:],
    }


# Mock the download_disordered_structures function
@pytest.fixture
def mock_download(monkeypatch, mock_disordered_structures):
    """Mock the download_disordered_structures function."""

    def mock_download_fn():
        return mock_disordered_structures

    monkeypatch.setattr(
        "material_hasher.benchmark.run_disordered.download_disordered_structures",
        mock_download_fn,
    )


def test_benchmark_disordered_structures(mock_download, constant_hasher):
    """Test the full benchmark with mocked data."""
    df_results, total_time = benchmark_disordered_structures(
        constant_hasher,
        seeds=[0, 1],  # Use two seeds for testing
    )

    # Check that we have results for all groups
    assert isinstance(df_results, pd.DataFrame)
    assert "group1" in df_results.index
    assert "group2" in df_results.index
    assert "dissimilar_case" in df_results.index

    # Check that total time is recorded
    assert "total_time (s)" in df_results.index
    assert isinstance(total_time, float)
    assert total_time > 0

    # For constant hasher, all success rates should be 1.0, except for dissimilar_case
    for idx in ["group1", "group2", "dissimilar_case"]:
        if idx == "dissimilar_case":
            assert all(rate == 0.0 for rate in df_results.loc[idx, "success_rate"])
        else:
            assert all(rate == 1.0 for rate in df_results.loc[idx, "success_rate"])


def test_benchmark_disordered_structures_random(mock_download, random_hasher):
    """Test the full benchmark with random hasher."""
    df_results, total_time = benchmark_disordered_structures(
        random_hasher,
        seeds=[0],  # Use only one seed for testing
    )

    # For random hasher, all success rates should be 0.0
    for idx in ["group1", "group2", "dissimilar_case"]:
        if idx == "dissimilar_case":  # We have a 1 chance to generate different hashes
            assert all(rate == 1.0 for rate in df_results.loc[idx, "success_rate"])
        else:
            assert all(rate == 0.0 for rate in df_results.loc[idx, "success_rate"])
