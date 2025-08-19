import datetime
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import yaml
from pymatgen.core import Structure

from material_hasher.benchmark.disordered import get_group_structure_results
from material_hasher.benchmark.run_transformations import get_hugging_face_dataset
from material_hasher.benchmark.utils import get_structure_from_hf_row
from material_hasher.hasher import HASHERS
from material_hasher.similarity import SIMILARITY_MATCHERS
from material_hasher.types import StructureEquivalenceChecker

STRUCTURE_CHECKERS = {**HASHERS, **SIMILARITY_MATCHERS}


def get_benchmark_data(n_structures: int, seed: Optional[int] = 0) -> List[Structure]:
    """Get the benchmark data for the given number of structures.

    Parameters
    ----------
    n_structures : int
        Number of structures to use for benchmarking.

    Returns
    -------
    structures : List[Structure]
        List of structures for benchmarking.
    """

    np.random.seed(seed)
    hf_dataset = get_hugging_face_dataset()
    random_indices = np.random.choice(len(hf_dataset), n_structures, replace=False)
    hf_dataset = hf_dataset.select(random_indices).to_pandas()
    structures = [get_structure_from_hf_row(row) for _, row in hf_dataset.iterrows()]  # type: ignore
    return structures


def plot_runtimes(file_path, output_path):
    data = pd.read_csv(file_path, index_col=0)

    # Convert the column names (which look like "10", "100", etc.) to integers
    sequence_lengths = [int(x) for x in data.columns[1:]]

    plt.figure(figsize=(7, 5))

    for algorithm in data.index:
        means = []
        std_devs = []
        for col in data.columns[1:]:
            # Convert the string of runtimes to a list of floats
            algorithm_runtimes = eval(data.at[algorithm, col])
            means.append(np.mean(algorithm_runtimes))
            std_devs.append(np.std(algorithm_runtimes))

        # Plot the mean runtimes with error bars (standard deviation)
        plt.errorbar(
            sequence_lengths,
            means,
            yerr=std_devs,
            capsize=5,
            marker="o",
            linestyle="-",
            label=algorithm,
        )

    plt.title("Algorithm Runtimes to compare all Structure pairs", fontsize=16)
    plt.xlabel("Number of Structures", fontsize=14)
    plt.ylabel("Average Runtime (s)", fontsize=14)
    plt.legend(title="Algorithm", fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    output_file = Path(output_path) / "runtimes.pdf"
    plt.savefig(output_file, dpi=300)

    plt.show()


def benchmark_time(
    structure_checker: StructureEquivalenceChecker,
    seeds: List[int] = [0, 1, 2, 3, 4],
) -> Tuple[pd.DataFrame, float]:
    """Benchmark the time taken to compute pairwise equivalence
    of structures using the given hasher or similarity matcher.

    Parameters
    ----------
    hasher : StructureEquivalenceChecker
        Hasher to use for benchmarking.

    Returns
    -------
    df_results : pd.DataFrame
        Dataframe containing the results of the benchmarking.
    total_time : float
        Total time taken for the benchmark
    """
    print("Downloading benchmark data...")
    start_script_time = time.time()
    all_structures = [get_benchmark_data(1000, seed) for seed in seeds]

    lengths_to_test = np.linspace(2, 1000, 20).astype(int)
    # we will stop here because 200**2 is already a lot of comparisons!

    results = defaultdict(list)

    for length in tqdm.tqdm(lengths_to_test):
        print(f"Benchmarking {length} structures...")
        for seed in seeds:
            structures_length_seed = all_structures[seed][:length]
            start_time = time.time()
            _ = get_group_structure_results(structure_checker, structures_length_seed)
            total_time = time.time() - start_time
            results[length].append(total_time)

    # the dataframe is just a one row dataframe with the list of times for all seeds
    df_results = pd.DataFrame({length: [results[length]] for length in lengths_to_test})
    print(df_results)

    total_time = time.time() - start_script_time

    return df_results, total_time


def main():
    """
    Run the benchmark for the disordered structures.

    This function provides a command-line interface to benchmark hashers and similarity matchers.

    Get help with:

    .. code-block:: bash

        $ python -m material_hasher.benchmark.run_time --help
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Benchmark hashers and similarity matchers for time taken to compare structures."
    )
    parser.add_argument(
        "--algorithm",
        choices=list(STRUCTURE_CHECKERS.keys()) + ["all"],
        help=f"The name of the structure checker to benchmark. One of: {list(STRUCTURE_CHECKERS.keys()) + ['all']}",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for the results. Default: 'results/'",
        default="results/",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Name of the .yaml configuration file to use for the hyperparameters of the hasher. Defaults to default.yaml",
        default="default.yaml",
    )
    args = parser.parse_args()

    config = yaml.safe_load(open(Path("configs") / args.config, "r"))
    output_path = Path(args.output_path) / datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    os.makedirs(output_path, exist_ok=True)
    yaml.dump(config, open(output_path / "config.yaml", "w"))

    if args.algorithm not in STRUCTURE_CHECKERS and args.algorithm != "all":
        raise ValueError(
            f"Invalid algorithm: {args.algorithm}. Must be one of: {list(STRUCTURE_CHECKERS.keys()) + ['all']}"
        )

    all_results = {}
    for structure_checker_name, structure_checker_class in STRUCTURE_CHECKERS.items():
        if args.algorithm != "all" and structure_checker_name != args.algorithm:
            continue

        structure_checker = structure_checker_class(
            **config.get(structure_checker_name, {})
        )
        df_results, structure_checker_time = benchmark_time(structure_checker)
        df_results.to_csv(output_path / f"{structure_checker_name}_results_time.csv")
        all_results[structure_checker_name] = df_results
        print(f"{structure_checker_name}: {structure_checker_time:.3f} s")

    if args.algorithm == "all":
        all_results = pd.concat(all_results, names=["algorithm"])
        file_path = output_path / "all_results_time.csv"
        all_results.to_csv(file_path)
    else:
        file_path = output_path / f"{args.algorithm}_results_time.csv"

    plot_runtimes(file_path, output_path)


if __name__ == "__main__":
    main()
