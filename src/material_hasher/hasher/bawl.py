# Copyright 2025 Entalpic
from pymatgen.analysis.local_env import EconNN, NearNeighbors
from pymatgen.core import Structure

from material_hasher.hasher.base import HasherBase
from material_hasher.hasher.utils.graph import get_weisfeiler_lehman_hash
from material_hasher.hasher.utils.graph_structure import get_structure_graph
from material_hasher.hasher.utils.symmetry import (
    AFLOWSymmetry,
    MoyoSymmetry,
    SPGLibSymmetry,
)


class BAWLHasher(HasherBase):
    """BAWL (Bonding Algorithm Weisfeiler Lehman) is a materials fingerprint method
    based on bonding graph structure, composition, and symmetry [1, 2].

    Parameters
    ----------
    graphing_algorithm (str, optional): Graph hashing algorithm.
        Currently only Weisfeiler-Lehman (WL) is implemented.
        Defaults to "WL".
    bonding_algorithm (NearNeighbors, optional): Pymatgen
        NearNeighbors class to compute bonds and create
        bonded structure. Defaults to EconNN.
    bonding_kwargs (dict, optional): kwargs passed to the
        Near Neighbor class. Defaults to {}.
    include_composition (bool, optional): Whether to
        include composition in the hash. Defaults to False.
    symmetry_labeling (str, optional): Method for deciding symmetry
        label. Only AFLOW, SPGLib, or moyo implemented. AFLOW requires
        AFLOW python packages. SPGLib requires SPGLib python packages.
        moyo requires moyo python packages. Defaults to "moyo".
    shorten_hash (bool, optional): Whether to shorten the hash.
        The shortened hash does not include the symmetry label.
        Defaults to False.

    References
    ----------
    [2] Martin Siron, et al. LeMat-Bulk: aggregating, and de-duplicating quantum chemistry
    materials databases. In AI for Accelerated Materials Design - ICLR 2025.
    [1] Ongari, Daniele, et al. "Data-driven matching of experimental crystal structures
    and gas adsorption isotherms of Metal–Organic frameworks." Journal of Chemical &
    Engineering Data 67.7 (2022): 1743-1756.
    """

    def __init__(
        self,
        graphing_algorithm: str = "WL",
        bonding_algorithm: NearNeighbors = EconNN,
        bonding_kwargs: dict = {"tol": 0.2, "cutoff": 10, "use_fictive_radius": True},
        include_composition: bool = True,
        symmetry_labeling: str = "moyo",
        shorten_hash: bool = False,
    ):
        self.graphing_algorithm = graphing_algorithm
        self.bonding_algorithm = bonding_algorithm
        self.bonding_kwargs = bonding_kwargs
        self.include_composition = include_composition
        self.symmetry_labeling = symmetry_labeling
        self.shorten_hash = shorten_hash

    def get_bawl_materials_data(
        self, structure: Structure, symmetry_label: int | str | None = None
    ) -> dict:
        """Gets various hash components for given Pymatgen structure.

        Parameters
        ----------
        structure (Structure): pymatgen structure object
        symmetry_label (int | str | None): symmetry label if already computed.
            Otherwise, the symmetry label will be computed using the class
            symmetry_labeling attribute.

        Raises
        ------
        ValueError: if non-implemented hash method
            is requested.
        ValueError: if non implemented symmetry method
            is requested.

        Returns
        -------
        dict: data dictionary with all hash components
        """
        data = dict()
        if self.graphing_algorithm == "WL":
            graph = get_structure_graph(
                structure,
                bonding_kwargs=self.bonding_kwargs,
                bonding_algorithm=self.bonding_algorithm,
            )
            data["bonding_graph_hash"] = get_weisfeiler_lehman_hash(graph)
        else:
            raise ValueError(
                "Graphing algorithm {} not implemented".format(self.graphing_algorithm)
            )
        if not self.shorten_hash:
            match (self.symmetry_labeling, symmetry_label):
                case (_, label) if label is not None:
                    data["symmetry_label"] = label
                case ("AFLOW", _):
                    data["symmetry_label"] = AFLOWSymmetry().get_symmetry_label(
                        structure
                    )
                case ("SPGLib", _):
                    data["symmetry_label"] = SPGLibSymmetry().get_symmetry_label(
                        structure
                    )
                case ("moyo", _):
                    data["symmetry_label"] = MoyoSymmetry().get_symmetry_label(
                        structure
                    )
                case (unknown, _):
                    raise ValueError(f"Symmetry algorithm {unknown} not implemented")
        if self.include_composition:
            data["composition"] = structure.composition.formula.replace(" ", "")
        return data

    def get_material_hash(self, structure: Structure) -> str:
        """Returns a hash of the structure.

        Parameters
        ----------
        structure : Structure
            Structure to hash.

        Returns
        -------
        str
            Hash of the structure.
        """
        data = self.get_bawl_materials_data(structure)
        return "_".join([str(v) if v is not None else "" for k, v in data.items()])


class ShortBAWLHasher(BAWLHasher):
    def __init__(
        self,
    ):
        super().__init__(shorten_hash=True)


class BAWLHasherLegacy(BAWLHasher):
    def __init__(self):
        super().__init__(
            graphing_algorithm="WL",
            bonding_algorithm=EconNN,
            bonding_kwargs={"tol": 0.2, "cutoff": 10, "use_fictive_radius": True},
            include_composition=True,
            symmetry_labeling="SPGLib",
            shorten_hash=False,
        )
