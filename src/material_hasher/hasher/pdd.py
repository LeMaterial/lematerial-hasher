# Copyright 2025 Entalpic
from hashlib import sha256

import numpy as np
from amd import PDD, PeriodicSet
from pymatgen.core import Structure

from material_hasher.hasher.base import HasherBase


class PointwiseDistanceDistributionHasher(HasherBase):
    def __init__(self, cutoff: float = 100.0):
        """
        Initialize the PDD Generator.

        Parameters:
        cutoff (float): Cutoff distance for PDD calculation. Default is 100.
        """
        self.cutoff = int(cutoff)  # Ensure cutoff is an integer

    def periodicset_from_structure(self, structure: Structure) -> PeriodicSet:
        """Convert a pymatgen Structure object to a PeriodicSet.

        Parameters
        ----------
        structure : pymatgen.Structure
            A pymatgen Structure object representing a crystal.

        Returns
        -------
        :class:`amd.PeriodicSet`
            Represents the crystal as a periodic set, consisting of a finite
            set of points (motif) and lattice (unit cell).

        Raises
        ------
        ValueError
            Raised if the structure has no valid sites.
        """

        # Unit cell
        cell = np.array(structure.lattice.matrix)

        # Coordinates and atomic numbers
        coords = np.array(structure.cart_coords)
        atomic_numbers = np.array([site.specie.number for site in structure.sites])

        # Check if the resulting motif is valid
        if len(coords) == 0:
            raise ValueError("The structure has no valid sites after filtering.")

        # Map coordinates to the unit cell (fractional positions mod 1)
        frac_coords = np.mod(structure.lattice.get_fractional_coords(coords), 1)

        motif = frac_coords

        return PeriodicSet(
            motif=motif,
            cell=cell,
            types=atomic_numbers,
        )

    def get_material_hash(self, structure: Structure) -> str:
        """
        Generate a hashed string for a single pymatgen structure based on its
        Point-wise Distance Distribution (PDD).

        Parameters
        ----------
        structure : pymatgen.Structure
            A pymatgen Structure object representing the crystal structure.

        Returns
        -------
        str
            A SHA256 hash string generated from the calculated PDD.
        """
        periodic_set = self.periodicset_from_structure(structure)

        pdd = PDD(periodic_set, int(self.cutoff), collapse=False)

        # Round the PDD values to 4 decimal places for numerical stability and consistency.
        pdd = np.round(pdd, decimals=4)

        # PDD hash array to PDD hash string
        string_pdd = pdd.tobytes()
        string_pdd = sha256(string_pdd).hexdigest()

        return string_pdd
