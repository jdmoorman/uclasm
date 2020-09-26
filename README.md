<div align="center">
<img src="logo.png" alt="logo">
</div>

<h2 align="center">Subgraph Matching on Multiplex Networks</h2>

<div align="center">
<a href="https://zenodo.org/badge/latestdoi/148378128"><img alt="Zenodo Archive" src="https://zenodo.org/badge/148378128.svg"></a>
<a href="https://pypi.org/project/ucla-subgraph-matching/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/ucla-subgraph-matching.svg"></a>
<a href="https://pypi.org/project/ucla-subgraph-matching/"><img alt="Supported Python Versions" src="https://img.shields.io/pypi/pyversions/ucla-subgraph-matching.svg"></a>
</div>

To reproduce our experiments, you will need at least Python 3.7 and a few packages installed. You can check your python version with

```bash
$ python --version
```
and install the necessary packages with
```bash
$ python -m pip install numpy scipy pandas tqdm matplotlib networkx
```

You will also need a local copy of our code either cloned from GitHub or downloaded from a Zenodo archive. To install our package from your local copy of the code, change to the code directory and use pip.

```bash
$ cd ucla-subgraph-matching
$ python -m pip install .
```

#### Erdős–Rényi Experiments

Running the experiments will take a while depending on your hardware.

```bash
$ cd experiments
$ python run_erdos_renyi.py
$ python plot_erdos_renyi.py
```

This will generate figures called XXX which correspond to figures XXX in the paper.

THOMAS FILL IN THE DETAILS.

#### Sudoku Experiments

Running the experiments will take a while depending on your hardware.

```bash
$ cd experiments
$ python run_sudoku.py
$ python plot_sudoku_times.py
```

This will generate figures called XXX which correspond to figures XXX in the paper.

THOMAS FILL IN THE DETAILS.

### TODO
* **Update Zenodo badge after new release if necessary, not sure how that really works.**
* **Add PyPI version badge.**
* **Add supported python versions badge.**

<!-- <div align="center">
<a href="https://pypi.org/project/kaczmarz-algorithms/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/kaczmarz-algorithms.svg"></a>
<a href="https://pypi.org/project/kaczmarz-algorithms/"><img alt="Supported Python Versions" src="https://img.shields.io/pypi/pyversions/kaczmarz-algorithms.svg"></a>
<a href="https://github.com/jdmoorman/kaczmarz-algorithms/actions"><img alt="Build Status" src="https://github.com/jdmoorman/kaczmarz-algorithms/workflows/CI/badge.svg"></a>
<a href="https://codecov.io/gh/jdmoorman/kaczmarz-algorithms"><img alt="Code Coverage" src="https://codecov.io/gh/jdmoorman/kaczmarz-algorithms/branch/master/graph/badge.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</div> -->
