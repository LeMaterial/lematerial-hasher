import pytest
from datasets import Dataset
from material_hasher.benchmark.disordered import download_disordered_structures
from material_hasher.benchmark.run_transformations import get_data_from_hugging_face


@pytest.fixture
def small_test_dataset():
    """Create a small synthetic dataset for testing benchmark functions"""
    data = {
        "lattice_vectors": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
        "species_at_sites": [["Si"]],
        "cartesian_site_positions": [[[0, 0, 0]]],
    }
    return Dataset.from_dict(data)


def test_hugging_face_data_loading(small_test_dataset, monkeypatch):
    """Test that data loading works with a small test dataset"""

    def mock_load(*args, **kwargs):
        return small_test_dataset

    # Mock the dataset loading
    monkeypatch.setattr(
        "material_hasher.benchmark.run_transformations.load_dataset", mock_load
    )

    structures = get_data_from_hugging_face(n_test_elements=1)
    assert len(structures) == 1
    assert structures[0].formula == "Si1"


@pytest.mark.integration
def test_download_transformations_dataset():
    """
    Download the HF dataset
    """
    data = get_data_from_hugging_face(
        n_test_elements=2, n_rows=2
    )  # Use small number for test
    assert len(data) == 2


@pytest.mark.integration
def test_download_disordered_structures():
    """
    Download the HF dataset
    """
    structures = download_disordered_structures()
    assert len(structures) > 0
