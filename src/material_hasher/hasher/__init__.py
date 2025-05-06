# Copyright 2025 Entalpic
from material_hasher.hasher.bawl import BAWLHasher, BAWLHasherLegacy, ShortBAWLHasher
from material_hasher.hasher.pdd import PointwiseDistanceDistributionHasher

__all__ = ["BAWLHasher"]

HASHERS = {
    "BAWL": BAWLHasher,
    "Short-BAWL": ShortBAWLHasher,
    "BAWL-Legacy": BAWLHasherLegacy,
    "PDD": PointwiseDistanceDistributionHasher,
}


try:
    from material_hasher.hasher.slices import SLICESHasher

    HASHERS.update({"SLICES": SLICESHasher})
except ImportError:
    pass
