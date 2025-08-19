# Copyright 2025 Entalpic
import os
import tempfile
from collections import Counter
from functools import partial
from itertools import combinations, islice
from multiprocessing import Pool
from time import time
from typing import Tuple
import logging

import numpy as np
import pandas as pd
import requests
from datasets import Dataset
from pandas import DataFrame
from pymatgen.core import Structure

from material_hasher.hasher.bawl import BAWLHasher
from material_hasher.similarity.structure_matchers import PymatgenStructureSimilarity

logger = logging.getLogger(__name__)


def download_and_merge_github_datasets(dataset_name: str) -> DataFrame:
    """
    Download and merge the datasets from the github repository. These datasets
    are split in train, val and test csv files. Thus, we need to merge them.

    Parameters
    ----------
    dataset_name : str
        the name of the dataset to download, should be one of the following:
        ["mpts", "mp-20", "perov-5", "carbon-24"]

    Returns
    -------
    DataFrame
        the merged dataframes
    """
    base_url = f"https://raw.githubusercontent.com/jiaor17/DiffCSP-PP/49a72521bce428c91f5ccafcf18f614b3426e769/data/{dataset_name}/"
    merged_dataset = pd.DataFrame()

    for file in ["train.csv", "val.csv", "test.csv"]:
        url = base_url + file
        response = requests.get(url)
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=True, suffix=".csv"
        ) as tmp_file:
            tmp_file.write(response.text)
            tmp_file.flush()

            data = pd.read_csv(tmp_file.name)
            merged_dataset = pd.concat([merged_dataset, data], ignore_index=True)

    return merged_dataset


def build_pymatgen_structures(df: DataFrame):
    """builds the pymatgen structures from the df. This operations avoids to
    build the structures n^2 times in the following script.

    Parameters
    ----------
    df : Dataframe
        dataframe of structures, containing the material_id and the cif, and
        some properties

    Returns
    -------
    dict
        keys: material_id, values: pymatgen structure
    """
    logger.info("building structures")

    material_id_list = df["material_id"].tolist()
    pymatgen_structures = {}

    for material_id in material_id_list:
        structure = Structure.from_str(
            df.loc[df["material_id"] == material_id]["cif"].values[0], fmt="cif"
        )
        pymatgen_structures[material_id] = structure

    logger.info("structures built")

    return pymatgen_structures


def compare_single_pair_of_structure(
    input_tuple: tuple, pymatgen_structures_dict: dict
):
    """compares a pair of structures with pymatgen

    Parameters
    -------
    input tuple: (material_id_1, material_id_2)
        a tuple of two material ids

    pymatgen_structures_dict: dict
        a dictionary of pymatgen structures, keys: material_id, values: pymatgen
        structures


    Returns
    -------

    tuple
        a tuple of (material_id_1, material_id_2, matching), where matching is a
        boolean value indicating whether the two structures match
    """

    material_id_1, material_id_2 = input_tuple

    # access the pre-built structures from the dictionary
    structure_1 = pymatgen_structures_dict[material_id_1]
    structure_2 = pymatgen_structures_dict[material_id_2]

    matching = PymatgenStructureSimilarity().is_equivalent(structure_1, structure_2)
    output_tuple = (material_id_1, material_id_2, matching)

    return output_tuple


def compare_pairs_of_structure_with_pymatgen(
    df: DataFrame, output_dir: str, batch_size: int = 30000000
):
    """main function to orchestrate the comparison of structures with pymatgen
    iterates over the combinations of material ids and compares the structures

    as the combinations might get very large, the function batches the
    iterations on the combinations, processes the batch and saves the results.

    Parameters
    ----------
    df : DataFrame
        dataframe of structures, containing the material_id and the cif, and
        some properties

    output_dir : str
        path to the directory where to save the results

    Returns
    -------
    None
    """
    pymatgen_structures = build_pymatgen_structures(df)
    material_id_list = df["material_id"].tolist()

    logger.info("comparing structures")

    # create the iterator of combinations (couples of material ids)
    iterator = combinations(material_id_list, 2)

    batch_number = 0
    batched_combinations_of_ids = list(islice(iterator, batch_size))

    while batched_combinations_of_ids:
        _compare_single_pair_of_structure = partial(
            compare_single_pair_of_structure,
            pymatgen_structures_dict=pymatgen_structures,
        )

        cpu_count = os.cpu_count() - 2

        with Pool(cpu_count) as p:
            try:
                results = p.map(
                    _compare_single_pair_of_structure,
                    batched_combinations_of_ids,
                    chunksize=batch_size,
                )

            except Exception as e:
                logger.error(f"An error occurred: {e}")
                continue

        results = pd.DataFrame(
            results,
            columns=[
                "material_id_1",
                "material_id_2",
                "matching",
            ],
        )
        results.to_parquet(
            os.path.join(output_dir, f"matching_results_{batch_number}.parquet"),
            index=True,
            engine="pyarrow",
        )
        logger.info(f"batch {batch_number} done")

        batch_number += 1
        batched_combinations_of_ids = list(islice(iterator, batch_size))


def process_hash(couple_mat_id_structure: tuple, primitive: bool = False):
    """process a single hash for a couple (mat_id, structure)
    It avoids to compute it n^2 times in the following script.

    Parameters
    ----------
    couple : tuple
        material_id, cif structure
    primitive : bool, optional
        whether to use the primitive cell, by default False
    Returns
    -------
    Dict
        keys : material_id, hash
    """
    mat_id, structure = couple_mat_id_structure[0], couple_mat_id_structure[1]
    crystal = Structure.from_str(structure, fmt="cif")

    hash_result = BAWLHasher().get_material_hash(crystal)

    return {"material_id": mat_id, "hash": hash_result}


def process_all_hashes_and_save_csv(df: pd.DataFrame, output_dir: str) -> None:
    """function the hash of all the structures in the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe containing the material_id and the cif
    output_dir : str
        path to the directory where to save the processed hashes
    """

    reduced_df = [(row["material_id"], row["cif"]) for _, row in df.iterrows()]

    workers = os.cpu_count() - 1
    logger.info("Processing the dataframe")
    with Pool(workers) as p:
        processed_results = p.map(process_hash, reduced_df)

    results_df = pd.DataFrame(processed_results, columns=["material_id", "hash"])

    output_file = os.path.join(output_dir, "processed_hash.csv")
    results_df.to_csv(output_file, index=False)


def shorten_hash(hash):
    split = hash.split("_")
    return split[0] + "_" + split[2]


def shorten_hash_and_composition(hash):
    split = hash.split("_")
    return split[0]


def get_duplicates_from_hash(
    path_to_dataset_hash: str,
    get_only_graph_hash: bool = True,
    get_shortened_hash: bool = False,
) -> DataFrame:
    """
    This function reads the dataset with the hashes and returns the duplicates
    as a Dtaframe with columns material_id_1, material_id_2, matching (True)

    Parameters
    ----------
    path_to_dataset_hash : str
        The path to the dataset with the pre-processed hashes
    get_shortened_hash : bool, optional
        whether to shorten the hash, by default True

    Returns
    -------
    DataFrame
        The dataframe containing the material_id of the duplicates
    """

    hashed_results = pd.read_csv(path_to_dataset_hash)

    if "hash_result" in hashed_results.columns:
        hashed_results.rename(columns={"hash_result": "hash"}, inplace=True)

    hashed_results["hash"] = hashed_results["hash"].apply(lambda x: x.replace(" ", ""))

    # we shorthen the hash to avoid the pmg symmetry label
    if get_only_graph_hash:
        hashed_results["hash"] = hashed_results["hash"].apply(
            shorten_hash_and_composition
        )
    elif get_shortened_hash:
        hashed_results["hash"] = hashed_results["hash"].apply(shorten_hash)

    duplicates_sorted = hashed_results[
        hashed_results.duplicated(subset=["hash"], keep=False)
    ]

    # group the duplicates by hash and get the list of grouped material_id under
    # the same hash
    grouped_duplicates = (
        duplicates_sorted.groupby("hash")
        .apply(lambda group: pd.Series({"material_id": group["material_id"].tolist()}))
        .reset_index()
    )

    matching_ids = []

    for _, row in grouped_duplicates.iterrows():
        # get across all the list of matching hash material ids
        material_ids = row["material_id"]
        for comb in combinations(material_ids, 2):
            matching_ids.append(
                {"material_id_1": comb[0], "material_id_2": comb[1], "matching": True}
            )

    return pd.DataFrame(matching_ids)


def concatenate_parquet_files_and_get_duplicates_from_pymatgen(
    subdir: str,
) -> DataFrame:
    """concatenates all the parquet files in a directory, where each parquet file
    contains the pairs of structures, with a column 'matching' indicating if the
    structures are matching. This directory is supposed to be the output of the
    compare_structures

    Parameters
    ----------
    subdir : str
        the subdir containing the parquet files of processed pairs of structures

    Returns
    -------
    DataFrame
        a dataframe containing all the matching pairs according to Pymatgen
    """
    # get all the parquet files in the directory
    parquet_files = [
        os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith(".parquet")
    ]

    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        df = df[df["matching"]]
        dfs.append(df)

    # only one single concat to avoid multiple copies of the same df
    df_matching = pd.concat(dfs, ignore_index=True)

    return df_matching


def compare_duplicates(
    pymatgen_duplicates: DataFrame, hashing_duplicates: DataFrame
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """compare the duplicates found by Pymatgen and the ones found by hashing
    sort it in common rows, unique to Pymatgen and unique to hashing (compounds of a Venn diagram)

    Parameters
    ----------
    pymatgen_duplicates : DataFrame
        dataframe containing the pairs duplicates found by Pymatgen
    hashing_duplicates : DataFrame
        dataframe containing the pairs duplicates found by the hash function

    Returns
    -------
    tuple[DataFrame, DataFrame, DataFrame]
        3 dataframe, for the common rows, the unique to Pymatgen and the unique to hashing
    """
    # create a frozenset of pairs to compare the duplicates in each dataset
    pymatgen_duplicates["pair"] = pymatgen_duplicates.apply(
        lambda row: frozenset([row["material_id_1"], row["material_id_2"]]), axis=1
    )
    hashing_duplicates["pair"] = hashing_duplicates.apply(
        lambda row: frozenset([row["material_id_1"], row["material_id_2"]]), axis=1
    )

    # get the common rows and delete useless columns
    common_rows = (
        pd.merge(pymatgen_duplicates, hashing_duplicates, on=["pair", "matching"])
        .drop(columns=["pair", "material_id_1_y", "material_id_2_y"])
        .rename(
            columns={
                "material_id_1_x": "material_id_1",
                "material_id_2_x": "material_id_2",
            }
        )
    )

    # get the deduplicated rows, with a _merge column to indicate the origin of
    # the row

    all_rows = pd.merge(
        pymatgen_duplicates,
        hashing_duplicates,
        on=["pair", "matching"],
        how="outer",
        indicator=True,
    )  # indicator = True for the _merge column

    # get the unique matching rows from the pymatgen matching, and drop/rename the columns
    unique_to_pymatgen = (
        all_rows[all_rows["_merge"] == "left_only"]
        .drop(columns=["_merge, pair, material_id_1_y", "material_id_2_y"])
        .rename(
            columns={
                "material_id_1_x": "material_id_1",
                "material_id_2_x": "material_id_2",
            }
        )
    )

    # get the unique matching rows from the hashing matching, and drop/rename the columns
    unique_to_hash = (
        all_rows[all_rows["_merge"] == "right_only"]
        .drop(columns=["_merge, pair, material_id_1_x", "material_id_2_x"])
        .rename(
            columns={
                "material_id_1_y": "material_id_1",
                "material_id_2_y": "material_id_2",
            }
        )
    )

    # Concatenate the material_id columns as a list

    unique_to_pymatgen["material_id"] = unique_to_pymatgen.apply(
        lambda x: [x["material_id_1"], x["material_id_2"]], axis=1
    )
    unique_to_hash["material_id"] = unique_to_hash.apply(
        lambda x: [x["material_id_1"], x["material_id_2"]], axis=1
    )
    common_rows["material_id"] = common_rows.apply(
        lambda x: [x["material_id_1"], x["material_id_2"]], axis=1
    )

    return common_rows, unique_to_pymatgen, unique_to_hash


def compute_hash_and_get_duplicates(df: DataFrame, hash_to_compare: str) -> None:
    """
    Computes all the

    Parameters
    ----------
    df : DataFrame
        the inout dataframe with the cif files
    hash_to_compare : str
        the hash to find in the dataset

    Returns
    -------
    None

    """
    df["hash_result"] = df.apply(
        lambda row: BAWLHasher().get_material_hash(
            Structure.from_str(row["cif"], fmt="cif")
        ),
        axis=1,
    )
    df["hash_result"] = df["hash_result"].apply(shorten_hash)
    df["equals_input_hash"] = df["hash_result"].apply(lambda x: x == hash_to_compare)


def build_structure_and_compare(structure_to_compare: Structure, cif_file: str) -> bool:
    """
    builds an Structure from an input cif file, and compares it to the
    structure_to_compare.

    Parameters
    ----------
    structure_to_compare : Structure
        The structure we are comparing to the structures in the dataset
    cif_file : str
        the cif file of the current row in the dataset

    Returns
    -------
    bool
        if the two structures are similar
    """
    structure = Structure.from_str(cif_file, fmt="cif")
    return PymatgenStructureSimilarity().is_equivalent(structure, structure_to_compare)


def get_number_of_duplicates_pymatgen(
    structure_to_compare: Structure, df: DataFrame
) -> None:
    """
    Process the search for duplicates in the dataset, using the pymatgen
    structure matcher

    Parameters
    ----------
    structure_to_compare : Structure
        The structure that we compare to the entire dataset
    df : DataFrame
        the dataframe containing the cif files and all the structures to compare
    """
    df["equals_input_pymatgen"] = df["cif"].apply(
        lambda x: build_structure_and_compare(structure_to_compare, x)
    )


def process_times_with_different_shape_datasets(
    dataset: DataFrame,
    hash_to_compare: str,
    structure_to_compare: Structure,
    sizes: list,
    repeats=3,
):
    """compares the time taken to identify all the duplicates in a dataset,
    accordinfg to StructureMatcher or hash comparison. The dataset size is
    varying in order to assess the scalability of the two methods (sizes:
    multiple of the original dataset size)


    Parameters
    ----------
    dataset : DataFrame
        the dataset containing the cif files
    hash_to_compare : str
        the hash to compare to the dataset
    structure_to_compare : Structure
        the structure to compare to the dataset
    sizes : list
        the list of sizes that the dataset will be multiplied by
    repeats : int, optional
        the number of repeats to get the average time, by default 3 (useful to
        statistics)

    Returns
    -------
    DataFrame
        a dataset with the aggregated times and stds for the two methods
    """

    results = []
    for size in sizes:
        logger.info(f"-------------------Size : {size}")
        df = pd.concat([dataset] * size, ignore_index=True)

        hash_times = []
        pymatgen_times = []

        for _ in range(repeats):
            # search the duplicates with the hash function and get the time
            time_start = time()
            compute_hash_and_get_duplicates(df, hash_to_compare)
            hash_times.append(time() - time_start)

            # search the duplicates with pymatgen and get the time
            time_start = time()
            get_number_of_duplicates_pymatgen(structure_to_compare, df)
            pymatgen_times.append(time() - time_start)

        results.append(
            {
                "size": len(df),
                "hash_mean_time": np.mean(hash_times),
                "hash_std_time": np.std(hash_times),
                "pymatgen_mean_time": np.mean(pymatgen_times),
                "pymatgen_std_time": np.std(pymatgen_times),
            }
        )

    results_df = pd.DataFrame(results)

    return results_df


def get_new_structure_with_gaussian_noise(
    structure: Structure, sigma: float, noise_type: str
) -> Structure:
    """
    Returns new structure with gaussian noise on either fractional coordinates,
    lattice or both.

    Parameters:
    ----------

        structure (Structure): Input structure to modify sigma (float,
        optional): Noise applied to atomic position. Defaults to 0.001.

    Returns:
    -------
        Structure: new structure with modification
    """

    if noise_type == "coords":  # apply noise to the coordinates
        return Structure(
            structure.lattice,
            structure.species,
            structure.cart_coords
            * (1 + np.random.normal(np.zeros(structure.frac_coords.shape), sigma)),
            coords_are_cartesian=True,
        )

    if noise_type == "lattice":  # apply noise to the lattice
        perturbed_lattice = structure.lattice.matrix + np.random.normal(
            0, sigma, size=(3, 3)
        )
        return Structure(
            perturbed_lattice,
            structure.species,
            structure.cart_coords,
            coords_are_cartesian=True,
        )

    if noise_type == "both":  # apply noise to the coordinates and lattice
        perturbed_lattice = structure.lattice.matrix + np.random.normal(
            0, sigma, size=(3, 3)
        )
        return Structure(
            perturbed_lattice,
            structure.species,
            structure.cart_coords
            * (1 + np.random.normal(np.zeros(structure.frac_coords.shape), sigma)),
            coords_are_cartesian=True,
        )


def apply_noise_to_structures_and_compare(
    std: int, df_apply_noise: DataFrame, noise_type: str
) -> dict:
    """applies noise with a certain standard deviation to 30 structures and
    compares the noisy structures with the original structures. We compare the
    pairs of structures using pymatgen, hash comparison and full hash
    comparison. We also calculate the rmsd between the noisy and original
    structures.

    We perform a total of 10 comparisons for each structure to get the result on
    the average of the added noise.

    Parameters
    ----------
    std : int
        standard deviation of the noise to apply to the structures

    df_apply_noise : DataFrame
        DataFrame containing the structures to apply the noise to

    noise_type : str
        type of noise to apply to the structures. Can be either "coords",
        "lattice" or "both".

    Returns
    -------
    Dict
        Dict of the mean of the matching of (noisy_struct, original_struct) for
        each structure out of the 30 structures
    """

    logger.info(f"-----------------------{std}-----------------------")

    list_equality = []
    pymatgen_list = []
    hash_list = []
    full_hash_list = []
    rmsd_list = []

    for idx in range(len(df_apply_noise)):
        pymatgen_comparison_list = []
        hash_comparison_list = []
        full_hash_comparison_list = []
        rmsd_values = []

        # the initial structure to apply the noise to, that is the reference
        initial_structure = Structure.from_str(
            df_apply_noise.iloc[idx]["cif"],
            fmt="cif",
        )
        try:
            for _ in range(
                10
            ):  # iterate 10 times to get the average on the added noise
                noisy_structure = get_new_structure_with_gaussian_noise(
                    initial_structure, std, noise_type
                )

                pymatgen_comparison = PymatgenStructureSimilarity().is_equivalent(
                    initial_structure, noisy_structure
                )  # pymatgen comparison
                rmsd = PymatgenStructureSimilarity().get_similarity_score(
                    initial_structure, noisy_structure
                )  # rmsd comparison
                hash_comparison = BAWLHasher(
                    shorten_hash=True
                ).is_equivalent(
                    initial_structure, noisy_structure
                )  # short hash comparison
                full_hash_comparison = BAWLHasher(
                    shorten_hash=False
                ).is_equivalent(
                    initial_structure, noisy_structure
                )  # full hash comparison

                pymatgen_comparison_list.append(pymatgen_comparison)
                hash_comparison_list.append(hash_comparison)
                full_hash_comparison_list.append(full_hash_comparison)

                if rmsd:
                    rmsd_values.append(rmsd[0])

            if len(pymatgen_comparison_list) > 0:
                pymatgen_list.append(
                    sum(pymatgen_comparison_list) / len(pymatgen_comparison_list)
                )
            if len(hash_comparison_list) > 0:
                hash_list.append(sum(hash_comparison_list) / len(hash_comparison_list))
            if len(full_hash_comparison_list) > 0:
                full_hash_list.append(
                    sum(full_hash_comparison_list) / len(full_hash_comparison_list)
                )
            if len(rmsd_values) > 0:
                rmsd_list.append(sum(rmsd_values) / len(rmsd_values))

        except Exception as e:
            logger.error(e)
            continue

    return {
        "std": std,
        "pymatgen": pymatgen_list,
        "hash": hash_list,
        "list_equality": list_equality,
        "full_hash": full_hash_list,
        "rmsd": rmsd_list,
    }


def get_equality_on_trajectory(
    data: DataFrame,
) -> Tuple[list[bool], list[bool], list[bool], list[int], str]:
    """
    Function that goes across the trajectory, and assess the matching between
    the final relaxed structure and the structures at each ionic step

    Parameters
    ----------
    data : DataFrame
        The dataframe of a single trajectory

    Returns
    -------
    Tuple[List[bool], List[bool], List[bool], List[int], str]

        pymatgen_equality_list : List[bool]
            A list of boolean that indicates if the structure at the
            ionic_step[i] is equal to the final structure, according to pymatgen
            structurematcher

        hash_equality_list : List[bool]
            A list of boolean that indicates if the structure at the
            ionic_step[i] is equal to the final structure, according to the
            shortened hash of the structure

        full_hash_equality_list : List[bool]
            A list of boolean that indicates if the structure at the
            ionic_step[i] is equal to the final structure, according to the full
            hash of the structure

        ionic_step_list : List[int]
            A list of the ionic steps of the trajectory

        final_row_id : str
            The material id of the final structure
    """

    pymatgen_equality_list = []
    hash_equality_list = []
    full_hash_equality_list = []
    ionic_step_list = []

    final_row = data.iloc[0]
    final_row_id = final_row["mp_id"]

    for idx in range(len(data)):
        # Get the structure at each ionic step, and compare it to the final structure

        on_traj_row = data.iloc[idx]

        final_structure = Structure(
            lattice=final_row["cell"],
            species=final_row["numbers"],
            coords=final_row["positions"],
        )

        on_traj_structure = Structure(
            lattice=on_traj_row["cell"],
            species=on_traj_row["numbers"],
            coords=on_traj_row["positions"],
        )

        pymatgen_equality = PymatgenStructureSimilarity().is_equivalent(
            final_structure, on_traj_structure
        )
        hash_equality = BAWLHasher(shorten_hash=True).is_equivalent(
            final_structure, on_traj_structure
        )
        full_hash_equality = BAWLHasher(shorten_hash=False).is_equivalent(
            final_structure, on_traj_structure
        )

        pymatgen_equality_list.append(pymatgen_equality)
        hash_equality_list.append(hash_equality)
        full_hash_equality_list.append(full_hash_equality)
        ionic_step_list.append(on_traj_row["ionic_step"])

    return (
        pymatgen_equality_list,
        hash_equality_list,
        full_hash_equality_list,
        ionic_step_list,
        final_row_id,
    )


def get_relevant_ids(dataset: Dataset) -> list[str]:
    """
    This function filters the dataset to return the list of materials that have more than 30 occurrences in the dataset. It avoids to study trajectories with very little trajectories.

    Parameters
    ----------
    dataset : Dataset
        The dataset to filter

    Returns
    -------
    list[str]
        The list of material ids of structures that are relevant to study
    """

    train_data = dataset["train"]

    element_counts = Counter(
        train_data["mp_id"]
    )  # Count the number of occurrences of each material id
    elements_with_more_than_30_occurrences = {
        element: count for element, count in element_counts.items() if count > 30
    }

    first_ids = list(elements_with_more_than_30_occurrences.keys())

    return first_ids


def filter_dataset_and_return_dataframe(
    dataset: Dataset, id: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function filters the dataset to keep only the trajectories of the
    material id passed as argument that have more than 10 occurences, and
    returns the filtered dataset as a pandas dataframe

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple of the matching conditions and the filtered dataframe
    """
    filtered_dataset = dataset.filter(
        lambda x: x["mp_id"] == id
    )  # Filter the dataset to keep only the trajectories of the current material id
    filtered_data_list = list(filtered_dataset["train"])
    filtered_dataframe = pd.DataFrame(
        filtered_data_list, columns=filtered_data_list[0].keys()
    )  # turn the  filter dataset into a pandas dataframe

    grouped_dataframe = filtered_dataframe.groupby(
        ["calc_id"]
    ).size()  # group the dataframe by calc_id and count the number of occurrences
    sufficient_conditions = grouped_dataframe[
        grouped_dataframe >= 10
    ]  # keep only the calc_id that have more than 10 occurrences

    return sufficient_conditions, filtered_dataframe


def study_trajectories(dataset, max_number_of_traj):
    results_list = []
    first_ids = get_relevant_ids(dataset)
    current_number_of_traj = 0

    for id in first_ids:
        # iterate over all the materials that have more than 30 occurrences
        if current_number_of_traj == max_number_of_traj:
            # Stop the loop if we reached the maximum number of trajectories
            # (enough samples to study)
            return results_list

        try:
            # Filter the dataset for the current material id, and see if we have
            # enough trajectories to study it
            sufficient_conditions, data = filter_dataset_and_return_dataframe(
                dataset, id
            )

            if not sufficient_conditions.empty:
                logger.info(
                    f"Iteration {current_number_of_traj}: Found calc_id and ionic_step"
                )

                # As some trajectories might have different ionic steps, we need
                # to choose one of them. We want to keep the ionic step has the
                # best weighting between the number of ionic steps and the
                # average ionic step (close to the end of the traj)
                stats = data.groupby("calc_id").agg(
                    row_count=("ionic_step", "size"),
                    avg_ionic_step=("ionic_step", "mean"),
                )
                stats["score"] = stats["row_count"] * stats["avg_ionic_step"]

                # Get the index (the ionic step) of the trajectory that has the
                # best score
                selected_calc_id = stats["score"].idxmax()
                # Filter the dataset to keep only the trajectory with the best
                # score
                filtered_data = data[data["calc_id"] == selected_calc_id]
                # Ascending = False :  we keep the end of the trajectory at the
                # beginning
                data = filtered_data.sort_values(by="ionic_step", ascending=False)

                (
                    pymatgen_equality_list,
                    hash_equality_list,
                    full_hash_equality_list,
                    ionic_step_list,
                    material_id,
                ) = get_equality_on_trajectory(data)

                results = {
                    "pymatgen_equality": pymatgen_equality_list,
                    "hash_equality": hash_equality_list,
                    "full_hash_equality": full_hash_equality_list,
                    "ionic_step": ionic_step_list,
                    "material_id": material_id,
                }

                results_list.append(results)
                current_number_of_traj += 1

        except Exception as e:
            logger.error(e)
            continue
