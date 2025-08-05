# Copyright 2025 Entalpic
from typing import Optional
import warnings

from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import EconNN, NearNeighbors
from pymatgen.core import Structure
from networkx import Graph
import spglib
from moyopy import MoyoDataset
from moyopy.interface import MoyoAdapter


def get_primitive_reduction(
    structure: Structure,
    symprec: float = 0.01,
    rad_angle_tolerance: Optional[float] = None,
) -> Structure:
    """Get primitive reduced structure from Pymatgen Structure object

    Args:
        structure (Structure): Pymatgen Structure object
        symprec_pr (float, optional): Distance tolerance in Angstroms 
            to find primitive reduction. Defaults  to 0.25, 
            default `tolerance` for pytmatgen's `Structure.get_primitive_structure`.
        rad_angle_tolerance (float, optional): Tolerance of angle between 
            basis vectors in radians to be tolerated in the symmetry 
            finding. Value in radians. Defaults to None, since
            the internet suggests not to use this variable: 
            https://github.com/spglib/spglib/issues/567

    Returns:
        Structure: Primitive reduced structure
    """
    cell = MoyoAdapter.from_structure(structure)
    mdata = MoyoDataset(
        cell,
        symprec=symprec,
        angle_tolerance=rad_angle_tolerance
    )
    prim_std_cell = mdata.prim_std_cell
    if prim_std_cell is None:
        raise RuntimeError("prim_std_cell (primitive reduced structure) was none.")
    else:
        return MoyoAdapter.get_structure(prim_std_cell)


def get_primitive_reduction_pymatgen(
    structure: Structure,
    tolerance: float = 0.25,
) -> Structure:
    """Get primitive reduced structure from Pymatgen Structure object

    Args:
        structure (Structure): Pymatgen Structure object
        symprec_pr (float, optional): Distance tolerance in Angstroms 
            to find primitive reduction. Defaults  to 0.25, 
            default `tolerance` for pytmatgen's `Structure.get_primitive_structure`.
        angle_tolerance (float, optional): Tolerance of angle between 
            basis vectors in radians to be tolerated in the symmetry 
            finding. Value in degrees. Defaults to -1, since
            the internet suggests not to use this variable: 
            https://github.com/spglib/spglib/issues/567

    Returns:
        Structure: Primitive reduced structure
    """
    try:
        struc = struc.get_primitive_structure(tolerance=tolerance)
    except:
        print("Failed to get primitive structure, using original structure.")
    try:
        struc = struc.get_reduced_structure()
    except:
        print("Failed to get reduced structure, using original structure.")
    return struc


def get_structure_graph(
    structure: Structure,
    bonding_kwargs: dict = {},
    bonding_algorithm: NearNeighbors = EconNN,
) -> Graph:
    """Method to build networkx graph object based on
    bonding algorithm from Pymatgen Structure

    Args:
        structure (Structure): Pymatgen Structure object
        bonding_kwargs (dict, optional): kwargs to pass to
            NearNeighbor class. Defaults to {}.
        bonding_algorithm (NearNeighbors, optional): NearNeighbor
            class to build bonded structure. Defaults to EconNN.

    Returns:
        Graph: networkx Graph object
    """
    assess_structure = structure.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        structure_graph = StructureGraph.from_local_env_strategy(
            structure=assess_structure,
            strategy=bonding_algorithm(**bonding_kwargs),
        )
    for n, site in zip(range(len(assess_structure)), assess_structure):
        structure_graph.graph.nodes[n]["specie"] = site.specie.name
    for edge in structure_graph.graph.edges:
        structure_graph.graph.edges[edge]["voltage"] = structure_graph.graph.edges[
            edge
        ]["to_jimage"]

    return structure_graph.graph
