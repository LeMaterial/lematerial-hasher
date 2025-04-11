# Copyright 2025 Entalpic
from material_hasher.hasher.entalpic import EntalpicMaterialsHasher, ShortenedEntalpicMaterialsHasher
from material_hasher.hasher.example import SimpleCompositionHasher

import warnings
warnings.filterwarnings('always')

__all__ = ["EntalpicMaterialsHasher"]

HASHERS = {
    "Entalpic": EntalpicMaterialsHasher,
    "Shortened Entalpic": ShortenedEntalpicMaterialsHasher,
}


try:
    from material_hasher.hasher.slices import SLICESHasher
    HASHERS.update({"SLICES": SLICESHasher})
except ImportError:
    warnings.warn('Failed to import SLICES. If you would like to use this module, please consider running uv pip install -r requirements_slices.txt', ImportWarning)
