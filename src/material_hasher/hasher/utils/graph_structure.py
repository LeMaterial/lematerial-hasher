# Copyright 2025 Entalpic
import warnings

from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import EconNN, NearNeighbors
from pymatgen.core import Structure
from networkx import Graph
from moyopy import MoyoDataset
from moyopy.interface import MoyoAdapter


def get_structure_graph(
    structure: Structure,
    bonding_kwargs: dict = {},
    bonding_algorithm: NearNeighbors = EconNN,
    primitive_reduction: bool = False,
    symprec: float = 0.1,
    angle_tolerance: float | None = 5,
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
    assess_structure = (
        MoyoAdapter.get_structure(
            MoyoDataset(
                MoyoAdapter.from_structure(structure),
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            ).prim_std_cell
        )
        if primitive_reduction
        else structure.copy()
    )
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
