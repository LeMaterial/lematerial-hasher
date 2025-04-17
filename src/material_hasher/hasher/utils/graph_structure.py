# Copyright 2025 Entalpic
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import EconNN, NearNeighbors
from pymatgen.core import Structure
from networkx import Graph


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
    structure_graph = StructureGraph.with_local_env_strategy(
        structure=structure,
        strategy=bonding_algorithm(**bonding_kwargs),
    )
    for n, site in zip(range(len(structure)), structure):
        structure_graph.graph.nodes[n]["specie"] = site.specie.name
    for edge in structure_graph.graph.edges:
        structure_graph.graph.edges[edge]["voltage"] = structure_graph.graph.edges[
            edge
        ]["to_jimage"]

    return structure_graph.graph
