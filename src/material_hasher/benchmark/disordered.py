# Copyright 2025 Entalpic
import logging
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tqdm
from datasets import Dataset, load_dataset
from pymatgen.core import Structure

from material_hasher.benchmark.utils import get_structure_from_hf_row
from material_hasher.hasher.base import HasherBase
from material_hasher.types import StructureEquivalenceChecker

logger = logging.getLogger(__name__)

HF_DISORDERED_PATH = "LeMaterial/sqs_materials"


def get_group_structures_from_data(
    hf_data: Dataset, data_groupby: pd.DataFrame, group_column: str, no_unique: int = 2
) -> Dict[str, List[Structure]]:
    """Get the structures grouped by a column in the data.

    Parameters
    ----------
    hf_data : Dataset
        Hugging Face dataset containing the structures.
    data_groupby : pd.DataFrame
        Dataframe containing the column to group the structures by and ordered
        the same way as the hf_data.
    group_column : str
        Column to group the structures by.
    no_unique : int
        Minimum number of unique structures to consider a group.

    Returns
    -------
    groups_dict : dict
        Dictionary containing the structures grouped by the column.
    """

    assert (
        group_column in data_groupby.columns
    ), f"Column {group_column} not found in data_groupby."

    hf_data = hf_data.select_columns(
        ["lattice_vectors", "species_at_sites", "cartesian_site_positions"]
    )

    groups = data_groupby.groupby(group_column).indices

    groups = {k: v for k, v in groups.items() if len(v) > no_unique}

    logger.info(f"Found {len(groups)} groups with more than {no_unique} structures.")

    hf_data = hf_data.select(np.concatenate(list(groups.values()))).to_pandas()

    new_groups = {}
    cumsum = 0
    for group, indices in groups.items():
        new_groups[group] = np.arange(cumsum, cumsum + len(indices))
        cumsum += len(indices)

    groups_dict = {}
    for group, indices in (
        pbar := tqdm.tqdm(new_groups.items(), desc="Loading groups")
    ):
        pbar.set_postfix_str(str(len(indices)))
        group_rows = hf_data.loc[indices]
        rows = [get_structure_from_hf_row(row) for _, row in group_rows.iterrows()]
        groups_dict[group] = rows

    return groups_dict


def download_disordered_structures(
    hf_disordered_path: str = HF_DISORDERED_PATH,
) -> Dict[str, List[Structure]]:
    """Download disordered structures from the HF dataset.

    Parameters
    ----------
    hf_disordered_path : str
        Path to the HF dataset containing disordered structures.

    Returns
    -------
    groups_dict : dict
        Dictionary containing the structures grouped by chemical formula.
    """
    hf_data = load_dataset(hf_disordered_path, split="train")
    dataset = hf_data.to_pandas()

    return get_group_structures_from_data(
        hf_data, dataset, "chemical_formula_descriptive"
    )


def get_dissimilar_structures(
    groups_dict: Dict[str, List[Structure]], n_picked_per_pair=10, seed=0
) -> Tuple[List[Tuple[int, int]], List[Structure]]:
    """Get dissimilar structures from the groups dictionary.

    Parameters
    ----------
    groups_dict : dict
        Dictionary containing the structures grouped by chemical formula.
    n_picked_per_pair : int
        Number of pairs of structure to pick for two disjoint groups of structures.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    dissimilar_structures : list[Tuple[int, int]]
        List of couples of dissimilar structures. Each couple is represented
        by the index of the structures in the unique_structures list.
    unique_structures : list[Structure]
        List of unique structures.
    """
    n_picked_per_pair = 40
    np.random.seed(seed)
    picked_structures = {}
    # keep track of the unique structures (to avoid storing structures)
    unique_structures = []
    # list of pairs of dissimilar structures (number associated to unique_structures id)
    dissimilar_structures = []

    all_group_names = list(groups_dict.keys())
    # list of all pairs of two different groups
    all_pairs = list(combinations(all_group_names, 2))

    for pair in all_pairs:
        group1 = groups_dict[pair[0]]
        group2 = groups_dict[pair[1]]
        for _ in range(n_picked_per_pair):
            structure1 = np.random.choice(list(range(len(group1))))
            if f"{pair[0]}_{structure1}" not in picked_structures:
                unique_structures.append(group1[structure1])
                picked_structures[f"{pair[0]}_{structure1}"] = (
                    len(unique_structures) - 1
                )
                structure1_idx = len(unique_structures) - 1
            else:
                structure1_idx = picked_structures[f"{pair[0]}_{structure1}"]
            structure2 = np.random.choice(list(range(len(group2))))
            if f"{pair[1]}_{structure2}" not in picked_structures:
                unique_structures.append(group2[structure2])
                picked_structures[f"{pair[1]}_{structure2}"] = (
                    len(unique_structures) - 1
                )
                structure2_idx = len(unique_structures) - 1
            else:
                structure2_idx = picked_structures[f"{pair[1]}_{structure2}"]
            dissimilar_structures.append((structure1_idx, structure2_idx))

    return dissimilar_structures, unique_structures


def get_group_structure_results(
    structure_checker: StructureEquivalenceChecker, structures: List[Structure]
) -> dict:
    """Get classification metrics from a list of structures.
    This function computes the pairwise equivalence matrix and then the classification metrics.

    Parameters
    ----------
    structure_checker : StructureEquivalenceChecker
        Structure equivalence checker.
    structures : List[Structure]
        List of structures to compute pairwise equivalence on.

    Returns
    -------
    metrics : dict
        Dictionary containing the classification metrics.
    """

    if isinstance(structure_checker, HasherBase):
        all_hashes = np.array(structure_checker.get_materials_hashes(structures))

        # creates a pairwise equivalence matrix
        pairwise_equivalence = all_hashes[:, None] == all_hashes[None, :]
    else:
        pairwise_equivalence = structure_checker.get_pairwise_equivalence(structures)

    # we only need the upper triangular part of the matrix
    triu_indices = np.triu_indices(len(structures), k=1)
    equivalence = np.array(pairwise_equivalence)[triu_indices].astype(int)
    metrics = get_classification_results(equivalence)
    return metrics


def get_classification_results(equivalence: np.ndarray) -> dict:
    """Get classification metrics from the pairwise equivalence matrix.
    Since all samples are labeled similar in this case, only the success rate is interesting

    Parameters
    ----------
    equivalence : np.ndarray
        Pairwise equivalence matrix.

    Returns
    -------
    metrics : dict
        Dictionary containing the classification metrics.
    """
    TP = np.sum(equivalence)
    FN = np.sum(equivalence == 0)
    success_rate = TP / (TP + FN)
    metrics = {"success_rate": success_rate}
    return metrics


def get_classification_results_dissimilar(
    structure_checker: StructureEquivalenceChecker,
    dissimilar_structures: List[List[Tuple[int, int]]],
    unique_structures: list[List[Structure]],
) -> Dict[str, List[float]]:
    """Get classification metrics from the dissimilar structures. Takes a list of lists of dissimilar structures for each seed.
    Only the success rate is interesting in this case because all samples are labeled dissimilar (so positive in this case).

    Parameters
    ----------
    structure_checker : StructureEquivalenceChecker
        Structure equivalence checker.
    dissimilar_structures : List[List[Tuple[Structure, Structure]]]
        List of dissimilar structures in the form of a list of tuples of
        the index of the structures on unique_structures for each seed.
    unique_structures : List[Structure]
        List of unique structures.

    Returns
    -------
    metrics : Dict[str, List[float]]
        Dictionary containing the classification metrics as a list of success rates for each seed.
    """
    success_rates = []
    for dissimilar_structures_seed, unique_structures_seed in tqdm.tqdm(
        zip(dissimilar_structures, unique_structures), desc="Dissimilar"
    ):
        TP = 0
        FN = 0

        all_hashes_seed = {}  # for pyright
        if isinstance(structure_checker, HasherBase):
            all_hashes_seed = np.array(
                structure_checker.get_materials_hashes(unique_structures_seed)
            )

        for structure1_idx, structure2_idx in dissimilar_structures_seed:
            if isinstance(structure_checker, HasherBase):
                is_equivalent = (
                    all_hashes_seed[structure1_idx] == all_hashes_seed[structure2_idx]
                )
            else:
                is_equivalent = structure_checker.is_equivalent(
                    unique_structures_seed[structure1_idx],
                    unique_structures_seed[structure2_idx],
                )
            TP += int(
                not is_equivalent
            )  # The structures are not equivalent, so the prediction is correct
            FN += int(is_equivalent)

        success_rate = TP / (TP + FN)
        success_rates.append(success_rate)
    return {"success_rate": success_rates}
