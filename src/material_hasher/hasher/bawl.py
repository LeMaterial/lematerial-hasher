# Copyright 2025 Entalpic
import math
from typing import Optional
from pymatgen.analysis.local_env import EconNN, NearNeighbors
from pymatgen.core import Structure

from material_hasher.hasher.base import HasherBase
from material_hasher.hasher.utils.graph import get_weisfeiler_lehman_hash
from material_hasher.hasher.utils.graph_structure import get_primitive_reduction, get_primitive_reduction_pymatgen, get_structure_graph
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
    primitive_reduction (str, optional): Whether to reduce the
        structure to its primitive cell before computing the hash.
        Defaults to None.
    shorten_hash (bool, optional): Whether to shorten the hash.
        The shortened hash does not include the symmetry label.
        Defaults to False.
    symprec_pr (float, optional): Distance tolerance in Angstroms 
        to find primitive reduction. Defaults  to 0.25, 
        default `tolerance` for pytmatgen's `Structure.get_primitive_structure`.
    angle_tolerance_pr (float, optional): Tolerance of angle between 
        basis vectors in degrees to be tolerated in the symmetry 
        finding. Value in degrees. Defaults to None, since
        the internet suggests not to use this variable: 
        https://github.com/spglib/spglib/issues/567
    symprec_sg (float, optional): Distance tolerance in Angstroms 
        to find crystal symmetry. May not be supported 
        for all backends. Defaults  to 0.01, default `symprec` for 
        pytmatgen's `SpacegroupAnalyzer`.
    angle_tolerance_sg (float, optional): Tolerance of angle between 
        basis vectors in degrees to be tolerated in the symmetry 
        finding. Value in degrees. May not be supported for all 
        backends. Defaults to None, since the internet suggests not 
        to use this variable: https://github.com/spglib/spglib/issues/567

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
        symmetry_labeling: str = "SPGLib",
        primitive_reduction: Optional[Literal["moyo", "pymatgen"]] = None,
        shorten_hash: bool = False,
        symprec_pr: float = 0.25,
        angle_tolerance_pr: Optional[float] = None,
        symprec_sg: float = 0.01,
        angle_tolerance_sg: Optional[float] = None,
    ) -> None:
        self.graphing_algorithm = graphing_algorithm
        self.bonding_algorithm = bonding_algorithm
        self.bonding_kwargs = bonding_kwargs
        self.include_composition = include_composition
        self.symmetry_labeling = symmetry_labeling
        self.primitive_reduction = primitive_reduction
        self.shorten_hash = shorten_hash
        self.symprec_pr = symprec_pr
        self.angle_tolerance_pr = angle_tolerance_pr
        self.symprec_sg = symprec_sg
        self.angle_tolerance_sg = angle_tolerance_sg
    

    @property
    def rad_angle_tolerance_pr(self) -> float:
        """Angle tolerance in radians for primitive reduction."""
        if self.angle_tolerance_pr is None:
            return None
        else:
            return self.angle_tolerance_pr * math.pi / 180
    
    @property
    def rad_angle_tolerance_sg(self) -> float:
        """Angle tolerance in radians for symmetry finding."""
        if self.angle_tolerance_sg is None:
            return None
        else:
            return self.angle_tolerance_sg * math.pi / 180


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
        if self.primitive_reduction == "moyo":
            structure = get_primitive_reduction(
                structure,
                symprec=self.symprec_pr,
                rad_angle_tolerance=self.rad_angle_tolerance_pr,
            )
        elif self.primitive_reduction == "pymatgen":
            structure = get_primitive_reduction_pymatgen(
                structure,
                symprec=self.symprec_pr,
            )
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
                    if self.symprec_sg is not None:
                        RuntimeWarning(f"{self.symprec_sg=} however, AFLOW does not support it.")
                    if self.angle_tolerance_sg is not None:
                        RuntimeWarning(f"{self.angle_tolerance_sg=} however, AFLOW does not support it.")
                    data["symmetry_label"] = AFLOWSymmetry().get_symmetry_label(
                        structure
                    )
                case ("SPGLib", _):
                    data["symmetry_label"] = SPGLibSymmetry(
                        symprec=self.symprec_sg,
                        angle_tolerance=-1 if self.angle_tolerance_sg is None else self.angle_tolerance_sg,
                    ).get_symmetry_label(
                        structure
                    )
                case ("moyo", _):
                    data["symmetry_label"] = MoyoSymmetry(
                        symprec=self.symprec_sg,
                        rad_angle_tolerance=self.rad_angle_tolerance_sg,
                    ).get_symmetry_label(
                        structure
                    )
                case (unknown, _):
                    raise ValueError(f"Symmetry algorithm {unknown} not implemented")
        if self.include_composition:
            data["composition"] = structure.composition.reduced_formula.replace(" ", "")
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
