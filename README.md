# material-hasher

`material-hasher` provides access to comprehensive benchmarks for material-fingerprinting and similarity methods, as well as implementation of fingerprint and similarity methods from the community.

## Benchmarks

In this release, we include the following benchmarks:

-   Transformations

    -   Noise on atomistic coordinates

        ```python
        from material_hasher.benchmark.transformations import get_new_structure_with_gaussian_noise
        ```

    -   Noise on lattice vectors

        ```python
        from material_hasher.benchmark.transformations import get_new_structure_with_strain
        ```

    -   Isometric strain on lattice

        ```python
        from material_hasher.benchmark.transformations import get_new_structure_with_isometric_strain
        ```

    -   Translations

        ```python
        from material_hasher.benchmark.transformations import get_new_structure_with_translation
        ```

    -   Application of symmetry operations

        ```python
        from material_hasher.benchmark.transformations import get_new_structure_with_symm_ops
        ```

-   Disordered materials

    -   Includes comprehensive test cases of structures generated via [Supercell](https://orex.github.io/supercell/) program and from Supercell paper to test whether various fingerprint or similarity metrics recognize disordered materials

        ```python
        from material_hasher.benchmark.run_disordered import benchmark_disordered_structures
        ```

## Fingerprinting methods

We include the following fingerprint methods:

-   a Structure graph, hashed via Weisfeiler-Lehman with and without symmetry labeling from SPGLib and composition

    ```python
    from material_hasher.hasher.bawl import BAWLHasher
    ```

-   SLICES

    ```python
    from material_hasher.hasher.slices import SLICESHasher
    ```

## Similarity methods

We include the following structure similarity methods:

-   Using GNN embeddings from trained and untrained EquiformerV2

    ```python
    from material_hasher.eqv2 import EquiformerV2Similarity
    ```

Note that this method requires access to the EquiformerV2 model trained on OMAT24 through [OMAT24 Hugging Face page](https://huggingface.co/fairchem/OMAT24). You then need to connect to your Hugging Face account via `huggingface-cli login` the first time before you run the code.

-   Pymatgen's StructureMatcher

    ```python
    from material_hasher.similarity.structure_matchers import PymatgenStructureSimilarity
    ```

## How to run benchmarks:

### Disordered benchmark

To test all the hasher and similarity methods on disordered materials dataset, seeing if each method can match the varying amount of disorder across a set of curated materials:

-   typical run (test disordered materials benchmark on all algorithms):

    ```bash
    $ python -m material_hasher.benchmark.run_disordered --algorithm all
    ```

-   get help:

    ```bash
    $ python -m material_hasher.benchmark.run_disordered --help
    ```

### Transformation benchmark

To test all the hasher and similarity methods on varying transformations applied to the structures across materials sampled from LeMat-Bulk:

-   typical run (test BAWL fingerprint on all test cases for a single structure):

    ```bash
    $ python -m material_hasher.benchmark.run_transformations --algorithm BAWL --n-test-elements 1
    ```

-   get help:

    ```bash
    $ python -m material_hasher.benchmark.run_transformations --help
    ```

## How to utilize a fingerprint method:

Here is a sample code to get a hash result:

```python
import numpy as np
from pymatgen.core import Structure

from material_hasher.hasher.bawl import BAWLHasher

# create a structure
structure = Structure(np.eye(3) * 3, ["Si"], [[0, 0, 0]])

# initialize the hasher
emh = BAWLHasher()

# get the hash
print(emh.get_material_hash(structure))
```

## Installation

We utilize [`uv`](https://docs.astral.sh/uv/getting-started/installation/) as our package manager ([why?](https://docs.astral.sh/uv/#highlights)).

```bash
# Either
$ uv add git+https://github.com/lematerial/material-hasher.git
# Or
$ uv pip install git+https://github.com/lematerial/material-hasher.git
# Or
$ pip install git+https://github.com/lematerial/material-hasher.git
```

For local development, please run:

```bash
$ git clone https://github.com/lematerial/material-hasher.git
# or
$ git clone git@github.com:lematerial/material-hasher.git
# then
$ cd material-hasher
$ uv sync
# Optionally, you can install the package in the editable mode
$ uv pip install -e .
```

To utilize `EquiformerV2Similarity`, please run: 

```bash
uv sync --extra fairchem
```

To utilize SLICES, please run:

```bash
uv pip install -r requirements_slices.txt
```

## Citation

We are working on a pre-print describing our fingerprint method.

If your work makes use of the varying fingerprint methods, please consider citing:
SLICES:

```
@article{xiao2023invertible,
  title={An invertible, invariant crystal representation for inverse design of solid-state materials using generative deep learning},
  author={Xiao, Hang and Li, Rong and Shi, Xiaoyang and Chen, Yan and Zhu, Liangliang and Chen, Xi and Wang, Lei},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={7027},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

PDD:

```
@article{widdowson2021pointwise,
  title={Pointwise distance distributions of periodic point sets},
  author={Widdowson, Daniel and Kurlin, Vitaliy},
  journal={arXiv preprint arXiv:2108.04798},
  year={2021}
}
```

If your work makes use of varying similarity methods, please consider citing:
Pymatgen:

```
@article{ong2013python,
  title={Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis},
  author={Ong, Shyue Ping and Richards, William Davidson and Jain, Anubhav and Hautier, Geoffroy and Kocher, Michael and Cholia, Shreyas and Gunter, Dan and Chevrier, Vincent L and Persson, Kristin A and Ceder, Gerbrand},
  journal={Computational Materials Science},
  volume={68},
  pages={314--319},
  year={2013},
  publisher={Elsevier}
}
```

EquiformerV2:

```
@article{liao2023equiformerv2,
  title={Equiformerv2: Improved equivariant transformer for scaling to higher-degree representations},
  author={Liao, Yi-Lun and Wood, Brandon and Das, Abhishek and Smidt, Tess},
  journal={arXiv preprint arXiv:2306.12059},
  year={2023}
}
```
