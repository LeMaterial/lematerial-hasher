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
except ImportError as e:
    logging.warning(
        "EquiformerV2Similarity is not available. You need to install fairchem-core and its dependencies. "
        "This issue is known to affect MacOS systems. "
        "If you're not using MacOS, please ensure the optional dependencies required for this feature are installed. uv sync --extra fairchem and uv sync --extra geometric"
        "For more information, refer to issue #4: https://github.com/Entalpic/material-hasher/issues/4",
        f"Error: {e}",
    )
