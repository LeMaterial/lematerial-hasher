# Copyright 2025 Entalpic
import inspect
from typing import Optional

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ALL_TEST_CASES = [
    "gaussian_noise",
    "isometric_strain",
    "strain",
    "translation",
    "symm_ops",
]

PARAMETERS = {
    "gaussian_noise": {"sigma": np.logspace(0.0001, 0.5, 15, base=0.0000001)},
    "isometric_strain": {"pct": [1, 1.05, 1.1, 1.2, 1.5]},
    "strain": {"sigma": np.logspace(0.001, 0.5, 10, base=0.0000001)},
    "translation": {"sigma": np.logspace(0.0001, 0.5, 15, base=0.0000001)},
    "symm_ops": {"structure_symmetries": ["all_symmetries_found"]},
}


def get_new_structure_with_gaussian_noise(
    structure: Structure, sigma: float
) -> Structure:
    """Returns new structure with gaussian noise on atomic positions

    Args:
        structure (Structure): Input structure to modify
        sigma (float, optional): Noise applied to atomic position. Defaults to 0.001.

    Returns:
        Structure: new structure with modification
    """
    return Structure(
        structure.lattice,
        structure.species,
        structure.cart_coords
        + np.random.normal(np.zeros(structure.cart_coords.shape), sigma),
        coords_are_cartesian=True,
    )


def get_new_structure_with_isometric_strain(
    structure: Structure, pct: float
) -> Structure:
    """
    Args:
        structure (Structure): Input structure to modify
        pct (float): Strain applied to lattice in all direction

    Returns:
        Structure: new structure with modification
    """
    s = structure.copy()
    return s.scale_lattice(structure.volume * pct)


def get_new_structure_with_strain(structure: Structure, sigma: float) -> Structure:
    """
    Args:
        structure (Structure): Input structure to modify
        sigma (float): Percent noise applied to lattice vectors

    Returns:
        Structure: new structure with modification
    """
    s = structure.copy()
    return s.apply_strain(np.random.normal(np.zeros(3), sigma))


def get_new_structure_with_translation(structure: Structure, sigma: float) -> Structure:
    """
    Args:
        structure (Structure): Input structure to modify
        sigma (float): Noise to apply to lattice vectors

    Returns:
        Structure: new structure with modification
    """
    s = structure.copy()
    return s.translate_sites(
        [n for n in range(len(structure))], np.random.normal(np.zeros(3), sigma)
    )


def get_new_structure_with_symm_ops(
    structure: Structure,
    structure_symmetries: str,
) -> list[Structure]:
    """
    Modify a structure using symmetry operations.

    Parameters
    ----------
    structure : Structure
        Input structure to modify.
    structure_symmetries : str, optional
        Determines how symmetry operations are applied. It apply all symmetries by default.

    Returns
    -------
    list[Structure]
        A list of modified structures.

    Raises
    ------
    ValueError
        If an invalid value for `symm_ops` is provided.
    """
    # Analyze symmetry operations
    analyzer = SpacegroupAnalyzer(structure)
    symmetry_operations = analyzer.get_symmetry_operations()

    return [structure.copy().apply_operation(op) for op in symmetry_operations]


def make_test_cases(
    test_cases: Optional[list[str]] = None,
    ignore_test_cases: Optional[list[str]] = None,
) -> list[str]:
    """Utility function to generate a list of test cases to run based on the specified test cases and ignored test cases.

    The procedure is as follows:

    1. If ``test_cases`` is not ``None``, include only the specified test cases.
    2. Otherwise, if ``test_cases`` is ``None``, include all test cases (from :const:`ALL_TEST_CASES`).
    3. If ``ignore_test_cases`` is not ``None``, filter the list of test cases to exclude the specified test cases.

    Parameters
    ----------
    test_cases : Optional[list[str]], optional
        List of test cases the user wants, by default ``None``
    ignore_test_cases : Optional[list[str]], optional
        List of test to ignore, by default ``None``

    Returns
    -------
    list[str]
        List of test cases to run.

    Raises
    ------
    ValueError
        If an unknown test case is specified in ``test_cases`` or ``ignore_test_cases``.
    ValueError
        If the resulting list of test cases is empty.
    """
    all_test_cases = ALL_TEST_CASES.copy()

    if test_cases is None:
        test_cases = all_test_cases

    if ignore_test_cases is None:
        ignore_test_cases = []

    for t in test_cases + ignore_test_cases:
        if t not in all_test_cases:
            raise ValueError(f"Unknown test case: {t}")

    if test_cases is not None:
        all_test_cases = [tc for tc in all_test_cases if tc in test_cases]
    if ignore_test_cases is not None:
        all_test_cases = [tc for tc in all_test_cases if tc not in ignore_test_cases]

    if not all_test_cases:
        raise ValueError("No test cases to run.")

    return all_test_cases


def get_test_case(test_case: str) -> tuple:
    """
    Utility function to get test data for a given test case.

    Parameters
    ----------
    test_case : str
        Name of the test case.

    Returns
    -------
    tuple
        Function and corresponding parameters for the test case.

    Raises
    ------
    ValueError
        If the test case or corresponding function is not found.
    """
    # Get the current module
    module = inspect.getmodule(inspect.currentframe())

    # Create the function name dynamically
    func_name = f"get_new_structure_with_{test_case}"

    # Check if the function exists in the module
    if not hasattr(module, func_name):
        raise ValueError(f"Function {func_name} not found in module {module.__name__}")

    # Check if test_case exists in PARAMETERS
    if test_case not in PARAMETERS:
        raise ValueError(f"Test case {test_case} not found in PARAMETERS")

    # Retrieve the function and parameters
    function = getattr(module, func_name)
    return function, PARAMETERS[test_case]
