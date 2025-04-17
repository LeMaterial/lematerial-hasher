# Copyright 2025 Entalpic
from pymatgen.core import Structure

from moyopy import MoyoDataset
from moyopy.interface import MoyoAdapter


def reduce_structure(self, structure: Structure) -> Structure:
    """Reduce the structure to its primitive cell."""
    if self.primitive_reduction:
        return MoyoAdapter.get_structure(
            MoyoDataset(MoyoAdapter.from_structure(structure)).prim_std_cell
        )
    return structure
