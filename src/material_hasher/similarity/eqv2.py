# Copyright 2025 Entalpic
import logging
import os
from pathlib import Path
from typing import Optional, Union

import ase
import numpy as np
import yaml
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

# fmt: off
try:
    from fairchem.core import OCPCalculator
except ImportError:
    import warnings
    warnings.warn(
        "EquiformerV2Similarity is not available. You need to install fairchem-core and its dependencies. "
        "This issue is known to affect MacOS systems. "
        "If you're not using MacOS, please ensure the optional dependencies required for this feature are installed. uv sync --extra fairchem and uv sync --extra geometric"
        "For more information, refer to issue #4: https://github.com/Entalpic/material-hasher/issues/4",
        ImportWarning,
    )
# fmt: on

from huggingface_hub import hf_hub_download
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from material_hasher.similarity.base import SimilarityMatcherBase

HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "fairchem/OMAT24")
HF_MODEL_PATH = os.getenv("HF_MODEL_PATH", "eqV2_31M_omat_mp_salex.pt")
logger = logging.getLogger(__name__)


class EquiformerV2Similarity(SimilarityMatcherBase):
    """EquiformerV2 Embedder for structure similarity comparison.
    Designed for EquiformerV2 models trained on the OMAT24 dataset.
    These models can be found on the Hugging Face model hub at
    https://huggingface.co/fairchem/OMAT24

    Parameters
    ----------
    trained : bool
        Whether the model was trained or not
    cpu : bool
        Whether to use the cpu to run inference on or the gpu if one is found
    threshold : float, optional
        Threshold to determine similarity, by default 0.01
    n_relaxation_steps : int, optional
        Number of relaxation steps to perform on the atoms object before computing the embeddings of the atoms, by default 0 (no relaxations).
    model_path : Optional[str], optional
        Path to the model checkpoint if downloaded, by default None
    load_from_hf : bool, optional
        Whether to download the model from the Hugging Face model hub, by default True. Note that you need to have access to the model on the Hugging Face model hub to download it.
    agg_type : str
        Aggregation type to use for the embeddings, by default "sum" which sums the normalized embeddings.
    """

    def __init__(
        self,
        trained: bool = True,
        cpu: bool = False,
        threshold: float = 0.999,
        n_relaxation_steps: int = 0,
        model_path: Optional[Union[str, Path]] = None,
        load_from_hf: bool = True,
        agg_type: str = "sum",
    ):
        self.model_path = model_path
        self.load_from_hf = load_from_hf

        self.trained = trained
        self.cpu = cpu

        self.threshold = threshold
        self.n_relaxation_steps = n_relaxation_steps

        assert agg_type in ["sum", "mean"], "Aggregation type not supported"
        self.agg_type = agg_type

        self.calc = None
        self.features = {}
        self._load_model()

    def _load_model(self):
        """Load the model from the model path.
        The calculator is then saved as an attribute of the class and a hook is added to the model to extract the sum of the normalized embeddings.
        """
        if self.load_from_hf:
            try:
                self.model_path = hf_hub_download(
                    repo_id=HF_MODEL_REPO_ID, filename=HF_MODEL_PATH
                )
            except Exception as e:
                logger.error(
                    f"Failed to download the model from the Hugging Face model hub: {e}. Note that this method requires access to the EquiformerV2 model trained on OMAT24 through [OMAT24 Hugging Face page](https://huggingface.co/fairchem/OMAT24). You then need to connect to your Hugging Face account via `huggingface-cli login` the first time before you run the code."
                )

        if not self.trained:
            logger.warning(
                "⚠️ Loading an untrained model because trained is set to False."
            )
            calc = OCPCalculator(checkpoint_path=self.model_path, cpu=self.cpu)
            config = calc.trainer.config

            config["dataset"] = {
                "train": {"src": "dummy"}
            }  # for compatibility with yaml loading
            with open("/tmp/config.yaml", "w") as fh:
                yaml.dump(config, fh)
            self.calc = OCPCalculator(config_yml="/tmp/config.yaml", cpu=self.cpu)
        else:
            self.calc = OCPCalculator(checkpoint_path=self.model_path, cpu=self.cpu)

        self.add_model_hook()

    def add_model_hook(self):
        """Add a hook to the model to extract the sum of the normalized embeddings.
        The hook is added to the last norm block of the Interaction part of the model.

        Embeddings are stored in the features attribute of the class.
        """
        assert self.calc is not None, "Model not loaded"

        def hook_norm_block(m, input_embeddings, output_embeddings):  # noqa
            self.features["sum_norm_embeddings"] = (
                output_embeddings.narrow(1, 0, 1)
                .squeeze(1)
                .sum(0)
                .detach()
                .cpu()
                .numpy()
            )

            self.features["mean_norm_embeddings"] = (
                output_embeddings.narrow(1, 0, 1)
                .squeeze(1)
                .mean(0)
                .detach()
                .cpu()
                .numpy()
            )

        self.calc.trainer.model.backbone.norm.register_forward_hook(hook_norm_block)

    def relax_atoms(self, atoms: ase.Atoms) -> ase.Atoms:
        """Relax the atoms using the FIRE optimizer
        WARNING: This function modifies the atoms object

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object to relax
        """
        atoms.calc = self.calc

        dyn = FIRE(FrechetCellFilter(atoms))
        dyn.run(steps=self.n_relaxation_steps)

        return atoms

    def get_structure_embeddings(self, structure: Structure) -> np.ndarray:
        """Get the embeddings of the structure.

        Parameters
        ----------
        structure : Structure
            Structure to get the embeddings of.

        Returns
        -------
        np.ndarray
            Embeddings of the structure.
        """
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms = self.relax_atoms(atoms)

        if self.agg_type == "mean":
            return self.features["mean_norm_embeddings"]
        elif self.agg_type == "sum":
            return self.features["sum_norm_embeddings"]

    def get_similarity_embeddings(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> float:
        """Get the similarity score between two embeddings.
        Uses the cosine similarity between the embeddings.

        Parameters
        ----------
        embeddings1 : np.ndarray
            First embeddings to compare.
        embeddings2 : np.ndarray
            Second embeddings to compare.

        Returns
        -------
        float
            Similarity score between the two embeddings.
        """
        embeddings1_norm = np.linalg.norm(embeddings1)
        embeddings2_norm = np.linalg.norm(embeddings2)

        if embeddings1_norm == 0 or embeddings2_norm == 0:
            return 0.0

        return np.dot(embeddings1, embeddings2) / (embeddings1_norm * embeddings2_norm)

    def get_similarity_score(
        self, structure1: Structure, structure2: Structure
    ) -> float:
        """Get the similarity score between two structures.
        Uses the cosine similarity between the embeddings of the structures.

        Parameters
        ----------
        structure1 : Structure
            First structure to compare.
        structure2 : Structure
            Second structure to compare.

        Returns
        -------
        float
            Similarity score between the two structures.
        """

        embeddings1 = self.get_structure_embeddings(structure1)
        embeddings2 = self.get_structure_embeddings(structure2)

        return self.get_similarity_embeddings(embeddings1, embeddings2)

    def is_equivalent(
        self,
        structure1: Structure,
        structure2: Structure,
        threshold: Optional[float] = None,
    ) -> bool:
        """Returns True if the two structures are equivalent according to the
        implemented algorithm.
        Uses a threshold to determine equivalence if provided and the algorithm
        does not have a built-in threshold.

        Parameters
        ----------
        structure1 : Structure
            First structure to compare.
        structure2 : Structure
            Second structure to compare.

        Returns
        -------
        bool
            True if the two structures are similar, False otherwise.
        """
        score = self.get_similarity_score(structure1, structure2)

        if threshold is None:
            threshold = self.threshold

        return score >= threshold

    def get_pairwise_equivalence(
        self, structures: list[Structure], threshold: Optional[float] = None
    ) -> np.ndarray:
        """Returns a matrix of equivalence between structures.

        Parameters
        ----------
        structures : list[Structure]
            List of structures to compare.
        threshold : float, optional
            Threshold to determine similarity, by default None and the
            algorithm's default threshold is used if it exists.

        Returns
        -------
        np.ndarray
            Matrix of equivalence between structures.
        """
        if threshold is None:
            threshold = self.threshold

        all_embeddings = np.array(
            [self.get_structure_embeddings(s) for s in structures]
        )

        return self.get_pairwise_similarity_scores_from_embeddings(all_embeddings)
