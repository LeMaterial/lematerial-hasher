# Copyright 2025 Entalpic
from material_hasher.hasher.bawl import BAWLHasher, ShortBAWLHasher
from material_hasher.hasher.pdd import PointwiseDistanceDistributionHasher

__all__ = ["BAWLHasher"]

HASHERS = {
    "BAWL": BAWLHasher,
    "Short-BAWL": ShortBAWLHasher,
    "PDD": PointwiseDistanceDistributionHasher,
}


try:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ImportWarning)
        from material_hasher.hasher.slices import SLICESHasher

    HASHERS.update({"SLICES": SLICESHasher})
except ImportError:
    pass
