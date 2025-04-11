# Copyright 2025 Entalpic
import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from datasets import Dataset, VerificationMode, load_dataset
from pymatgen.core import Structure

from material_hasher.benchmark.transformations import ALL_TEST_CASES, get_test_case
from material_hasher.benchmark.utils import get_structure_from_hf_row
from material_hasher.hasher import HASHERS
from material_hasher.hasher.base import HasherBase
from material_hasher.similarity import SIMILARITY_MATCHERS
from material_hasher.similarity.base import SimilarityMatcherBase
from material_hasher.types import StructureEquivalenceChecker

logger = logging.getLogger(__name__)

STRUCTURE_CHECKERS = {**HASHERS, **SIMILARITY_MATCHERS}


def get_hugging_face_dataset(
    token: Optional[str] = None, n_rows: Optional[int] = None
) -> Dataset:
    """
    Only returns the dataset from Hugging Face where all the subsets are concatenated.

    Parameters
    ----------
    token : str, optional
        The authentication token required to access the dataset.
        Optional if the dataset is public or you have already configured the Hugging Face CLI.
    n_rows : int, optional
        Number of rows to load from the dataset.

    Returns
    -------
    Dataset
        The concatenated dataset from Hugging Face.
    """

    split = "train"
    if n_rows is not None:
        split += f"[:{n_rows}]"

    return load_dataset(
        "LeMaterial/LeMat-Bulk",
        "compatible_pbe",
        split=split,
        token=token,
        verification_mode=VerificationMode.NO_CHECKS,
    )


def get_data_from_hugging_face(
    token: Optional[str] = None,
    n_test_elements: int = 100,
    n_rows: Optional[int] = None,
    seed: int = 0,
) -> list[Structure]:
    """
    Downloads and processes structural data from the Hugging Face `datasets` library.

    This function fetches a dataset from Hugging Face, extracts relevant structural information,
    and converts it into a list of pymatgen Structure objects.

    Parameters
    ----------
    token : str, optional
        The authentication token required to access the dataset.
        Optional if the dataset is public or you have already configured the Hugging Face CLI.
    n_test_elements : int
        Number of elements to select from the dataset to run the benchmark on. Default is 100.
        This is used to run the transformation benchmark only a subset of LeMat-Bulk.
    n_rows : int, optional
        Number of rows to load from the dataset. This will load them in sequential order.
    seed : int
        Random seed for selecting a subset of the dataset. Default is 0.

    Returns
    -------
    list[Structure]
        A list of pymatgen Structure objects extracted and processed from the dataset.

    Raises
    ------
    ValueError
        If the dataset fails to load or the structures cannot be processed.

    Notes
    -----
    - The dataset is fetched from the `LeMaterial/LeMat-Bulk` repository using the
      `compatible_pbe` configuration.
    - Only the first entry of the dataset is selected for processing.
    - Errors during the transformation process are logged but do not halt execution.
    """

    ds = get_hugging_face_dataset(token, n_rows=n_rows)

    # Convert dataset to Pandas DataFrame
    logger.info("Loaded dataset:", len(ds))
    np.random.seed(seed)
    range_select = np.random.choice(len(ds), n_test_elements, replace=False)
    df = ds.select(range_select)

    # Transform dataset int pymatgen Structure objects
    structure_data = []
    for row in df:
        try:
            # Construct the Structure object
            struct = get_structure_from_hf_row(row)
            structure_data.append(struct)

        except Exception as e:
            # Log errors without interrupting processing
            logger.info(f"Error processing row : {e}")

    # Display the total number of successfully loaded structures
    logger.info(f"structure_data size: {len(structure_data)}")

    # Return the list of pymatgen Structure objects
    return structure_data


def apply_transformation(
    structure: Structure,
    test_case: str,
    parameter: tuple[str, any],
) -> list[Structure]:
    """
    Applies transformations to a structure using a specified test case and parameter.

    Parameters
    ----------
    structure : Structure
        Input structure to be transformed.
    test_case : str
        Test case to be applied.
    parameter : tuple[str, any]
        Name and value of the parameter to be used for the transformation.

    Returns
    -------
    list[Structure]
        List of transformed structures.

    Raises
    ------
    ValueError
        If no valid test case is provided.
    """
    if not structure:
        raise ValueError("No structure was provided.")

    if not test_case:
        raise ValueError("No test case was provided.")

    if not parameter:
        raise ValueError("No parameter was provided.")

    # Load the test case
    func, params = get_test_case(test_case)

    transformed_structures = []

    # Extract parameter name and value
    param_name, param_value = parameter
    kwargs = {param_name: param_value}

    # Apply the transformation
    result = func(structure, **kwargs)
    if isinstance(result, list):
        # If the result is a list, extend the transformed_structures list
        transformed_structures.extend(result)
    else:
        for _ in range(2):
            result = func(structure, **kwargs)
            transformed_structures.append(result)

    return transformed_structures


def hasher_sensitivity(
    structure: Structure,
    transformed_structures: list[Structure],
    structure_checker: StructureEquivalenceChecker,
) -> float:
    """
    Computes the proportion of transformed structures with hashes equal to the original structure's hash.

    Parameters
    ----------
    structure : Structure
        The original structure.
    transformed_structures : list[Structure]
        List of transformed structures.
    structure_checker : StructureEquivalenceChecker
        Structure equivalence checker

    Returns
    -------
    float
        Proportion of hashes equal to the original hash.
    """
    # This dichotomy allows to avoid hashing the original structure at every comparison
    # It can speed up the benchmark but just by a constant factor so it could be removed
    if isinstance(structure_checker, HasherBase):
        # Compute hash for the original structure
        original_hash = structure_checker.get_material_hash(structure)
        # Compute hashes for transformed structures
        transformed_hashes = [
            structure_checker.get_material_hash(s) for s in transformed_structures
        ]

        # Calculate the proportion of hashes matching the original hash
        matching_hashes = sum(1 for h in transformed_hashes if h == original_hash)
    elif isinstance(structure_checker, SimilarityMatcherBase):
        # Compute similarity for transformed structures
        matching_with_transformed = [
            structure_checker.is_equivalent(structure, s)
            for s in transformed_structures
        ]

        # Calculate the proportion of similarities matching the original similarity
        matching_hashes = sum(1 for h in matching_with_transformed if h)
    else:
        raise ValueError("Unknown structure checker")

    return (
        matching_hashes / len(transformed_structures)
        if len(transformed_structures) > 0
        else 0
    )


def mean_sensitivity(
    structure_data: list[Structure],
    test_case: str,
    parameter: tuple[str, any],
    structure_checker: StructureEquivalenceChecker,
) -> float:
    """
    Computes the mean sensitivity for all structures in the dataset.

    Parameters
    ----------
    structure_data : list[Structure]
        List of structures to be processed.
    test_case : str
        Test case to be applied.
    parameter : tuple[str, any]
        Name and value of the parameter to be used for the transformation.
    structure_checker : StructureEquivalenceChecker
        Structure equivalence checker

    Returns
    -------
    float
        Mean sensitivity across all structures.
    """
    sensitivities = []

    for structure in structure_data:
        # Apply transformation
        transformed_structures = apply_transformation(structure, test_case, parameter)
        # Compute sensitivity
        sensitivity = hasher_sensitivity(
            structure, transformed_structures, structure_checker
        )
        sensitivities.append(sensitivity)

    # Calculate and return mean sensitivity
    return sum(sensitivities) / len(sensitivities) if len(sensitivities) > 0 else 0


def sensitivity_over_parameter_range(
    structure_data: list[Structure],
    test_case: str,
    structure_checker: StructureEquivalenceChecker,
) -> dict[float, float]:
    """
    Computes mean sensitivity for a range of parameter values from PARAMETERS.

    Parameters
    ----------
    structure_data : list[Structure]
        List of structures to be processed.
    test_case : str
        Test case to be applied.
    structure_checker : StructureEquivalenceChecker
        Structure equivalence checker

    Returns
    -------
    dict[float, float]
        Dictionary mapping each parameter value to its mean sensitivity.
    """
    # Load parameters from test cases
    _, params = get_test_case(test_case)
    param_name = list(params.keys())[0]  # Generalize to fetch the first parameter name

    param_range = params[param_name]

    results = {}
    for param_value in param_range:
        parameter = (param_name, param_value)
        mean_sens = mean_sensitivity(
            structure_data, test_case, parameter, structure_checker
        )
        results[param_value] = mean_sens

    return results


def benchmark_transformations(
    structure_checker: StructureEquivalenceChecker,
    structure_data: list[Structure],
    test_case: Optional[str],
) -> dict[str, dict[float, float]]:
    """
    Benchmarks the given structure_checker on the provided structure data for a given structure checker.

    Parameters
    ----------
    structure_checker : StructureEquivalenceChecker
        Structure equivalence checker. This can be a hasher or a similarity matcher.
    structure_data : list[Structure]
        List of structures to be processed.
    test_case : str
        Test case to be applied.

    Returns
    -------
    dict[str, dict[float, float]]
        Dictionary containing results for each hasher.
    """
    all_results = {}
    start_time = time.time()
    if not (test_case):
        for test_case in ALL_TEST_CASES:
            results = sensitivity_over_parameter_range(
                structure_data, test_case, structure_checker
            )
            all_results[test_case] = results
    else:
        results = sensitivity_over_parameter_range(
            structure_data, test_case, structure_checker
        )
        all_results[test_case] = results

    end_time = time.time()

    return all_results, end_time - start_time


def diagram_sensitivity(
    structure_data: list[Structure],
    test_case: str,
    dataset_name: str,
    noise_type: str,
    output_dir: str,
):
    """
    Generates and saves sensitivity diagrams for all hashers.

    Parameters
    ----------
    structure_data : list[Structure]
        List of structures to be processed.
    test_case : str
        Test case to be applied.
    dataset_name : str
        Name of the dataset.
    noise_type : str
        Type of noise added.
    output_dir : str
        Directory to save the output plot.
    """

    results = {}
    for hasher_key, hasher in STRUCTURE_CHECKERS.items():
        hasher = hasher()
        results_hasher = benchmark_transformations(hasher, structure_data, test_case)[
            0
        ][test_case]
        results[hasher_key] = results_hasher

    logger.info("final dict results : ", results)

    plt.figure(figsize=(10, 6))
    for hasher_name, data in results.items():
        param_range = list(data.keys())
        sensitivities = list(data.values())
        plt.plot(param_range, sensitivities, label=hasher_name, marker="o")

    plt.xlabel("Parameter Value")
    plt.ylabel("Mean Sensitivity")
    plt.title(f"{dataset_name} with noise on {noise_type}")
    plt.legend()
    plt.grid(True)
    plt.xscale("log")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_path_figure = os.path.join(
        output_dir, f"{dataset_name}_{noise_type}_sensitivity_diagram.png"
    )

    plt.savefig(output_path_figure, dpi=600, bbox_inches="tight", format="png")

    # Convert results to DataFrame and save as CSV
    df = pd.DataFrame(results)
    output_path_csv = os.path.join(
        output_dir, f"{dataset_name}_{noise_type}_sensitivity_results.csv"
    )
    df.to_csv(output_path_csv, index=True)

    plt.show()


def main():
    """
    Run the benchmark for hashers.

    This function provides a command-line interface to benchmark hashers.

    Get help with:

    .. code-block:: bash

        $ python -m material_hasher.benchmark.run_transformations --help
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Benchmark hashers.")
    parser.add_argument(
        "--algorithm",
        choices=list(STRUCTURE_CHECKERS.keys()) + ["all"],
        help="The name of the structure checker to benchmark.",
    )
    parser.add_argument(
        "--test-cases",
        nargs="+",
        help="The test cases to run. If not provided, all test cases will be run.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for the results. Default: 'results/'",
        default="results/",
    )
    parser.add_argument(
        "--n-test-elements",
        type=int,
        help="Number of elements to select from the dataset to run the benchmark on. Default is 100.",
        default=100,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for selecting a subset of the dataset. Default is 0.",
        default=0,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Name of the .yaml configuration file to use for the hyperparameters of the hasher. Defaults to default.yaml",
        default="default.yaml",
    )
    args = parser.parse_args()

    # TODO: Combine that + the flag parser with other benchmarks into benchmark utils functions
    # Could be done in the same script with a flag to specify the benchmark type
    config = yaml.safe_load(open(Path("configs") / args.config, "r"))
    output_path = Path(args.output_path) / datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    os.makedirs(output_path, exist_ok=True)
    with open(output_path / "config.yaml", "w") as fh:
        yaml.dump(config, fh)

    if args.algorithm not in STRUCTURE_CHECKERS and args.algorithm != "all":
        raise ValueError(
            f"Invalid algorithm: {args.algorithm}. Must be one of: {list(STRUCTURE_CHECKERS.keys()) + ['all']}"
        )

    # TODO: Create a separate benchmark dataset?
    structure_data = get_data_from_hugging_face(
        n_test_elements=args.n_test_elements, seed=args.seed
    )

    all_results = {}
    for structure_checker_name, structure_checker_class in STRUCTURE_CHECKERS.items():
        if args.algorithm != "all" and structure_checker_name != args.algorithm:
            continue

        structure_checker = structure_checker_class(
            **config.get(structure_checker_name, {})
        )
        results_dict, structure_checker_time = benchmark_transformations(
            structure_checker, structure_data, args.test_cases
        )
        all_results[structure_checker_name] = results_dict
        with open(
            output_path / f"{structure_checker_name}_results_disordered.json", "w"
        ) as fh:
            json.dump(
                results_dict,
                fh,
            )

        logger.info(f"{structure_checker_name}: {structure_checker_time:.3f} s")

    if args.algorithm == "all":
        with open(output_path / "all_results_disordered.json", "w") as fh:
            json.dump(
                all_results,
                fh,
            )


if __name__ == "__main__":
    main()
