# Copyright 2025 Entalpic
from .structure_matchers import PymatgenStructureSimilarity

__all__ = ["PymatgenStructureSimilarity"]

SIMILARITY_MATCHERS = {
    "pymatgen": PymatgenStructureSimilarity,
}

try:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ImportWarning)
        from .eqv2 import EquiformerV2Similarity

        __all__.append("EquiformerV2Similarity")
        SIMILARITY_MATCHERS["eqv2"] = EquiformerV2Similarity  # type: ignore
except ImportError:
    pass
