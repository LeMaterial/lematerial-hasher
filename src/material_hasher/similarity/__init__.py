# Copyright 2025 Entalpic
import logging

from .structure_matchers import PymatgenStructureSimilarity

__all__ = ["PymatgenStructureSimilarity"]

SIMILARITY_MATCHERS = {
    "pymatgen": PymatgenStructureSimilarity,
}

try:
    from .eqv2 import EquiformerV2Similarity

    __all__.append("EquiformerV2Similarity")
    SIMILARITY_MATCHERS["eqv2"] = EquiformerV2Similarity  # type: ignore
except ImportError:
    logging.warning(
        "EquiformerV2Similarity is not available. This is a known issue on MacOS. Otherwise you need to installed the optional dependencies for this feature."
    )
