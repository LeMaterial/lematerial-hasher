# Copyright 2025 Entalpic
import warnings

from material_hasher.hasher.bawl import BAWLHasher, ShortBAWLHasher

warnings.filterwarnings("always")

__all__ = ["BAWLHasher"]

HASHERS = {
    "BAWL": BAWLHasher,
    "Short-BAWL": ShortBAWLHasher,
}


try:
    from material_hasher.hasher.slices import SLICESHasher

    HASHERS.update({"SLICES": SLICESHasher})
except ImportError:
    warnings.warn(
        "Failed to import SLICES. If you would like to use this module, please consider running uv pip install -r requirements_slices.txt",
        ImportWarning,
    )
