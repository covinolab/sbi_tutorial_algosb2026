# Simulation-based inference: hands-on tutorial, AlgoSB 2026

Two notebooks that turn the SBI lecture of [AlgoSB 2026](https://sites.google.com/view/algosb2026) into practice. They are written for PhD students who know Python with NumPy and basic probability and have never used SBI, PyTorch or neural density estimation.

| | Notebook | What you do | Open |
| --- | --- | --- | --- |
| 1 | [SBI from scratch on a 2D toy problem](01_sbi_toy_2d.ipynb) | Run the full SBI workflow where the exact posterior is known, and learn three ways to check a posterior | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/covinolab/sbi_tutorial_algosb2026/blob/main/01_sbi_toy_2d.ipynb) |
| 2 | [SBI for time series: Brownian motion in a double well](02_sbi_double_well.ipynb) | Infer a barrier height and a diffusion coefficient from one trajectory, with summary statistics you invent | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/covinolab/sbi_tutorial_algosb2026/blob/main/02_sbi_double_well.ipynb) |

Each notebook takes 75 minutes. Every part is a challenge with a check cell that prints ✅ PASS or ❌ NOT YET. How you reach the target is up to you: trial and error, changing the given code, thinking, or asking an LLM for code.

## Running the notebooks

**On Google Colab (recommended).** Click a button above. The first code cell installs `sbi` and fetches this repository. A free CPU runtime is enough; no GPU is needed.

**On your own machine.** You need Python 3.10 or newer. [INSTALL.md](INSTALL.md) sets up a virtual environment in a few commands, for Linux, macOS and Windows:

```bash
git clone https://github.com/covinolab/sbi_tutorial_algosb2026.git
cd sbi_tutorial_algosb2026
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux only
pip install -r requirements.txt
jupyter lab
```

## What is in this repository

```
01_sbi_toy_2d.ipynb         Notebook 1
02_sbi_double_well.ipynb    Notebook 2
tutorial_utils.py           simulators, the exact posterior, plotting helpers, the check cells
data/                       observations, blind trajectories, test set, pre-trained networks,
                            and benchmarks.json with the targets of all check cells
solutions/                  both notebooks with every challenge solved and explained
reference/                  for the instructors: scripts that produce data/ and the notebooks,
                            and our reference summary statistics for Notebook 2
tests/                      unit tests, and a test that executes all four notebooks
INSTALL.md                  running the notebooks without Colab
requirements.txt            the packages INSTALL.md installs
CHANGES.md                  what changed since 14 November 2026, and why
```

Stuck on a challenge? Ask an instructor first. The worked answers in `solutions/` give the reasoning, not only the code.

## For instructors

Everything below is for whoever maintains or teaches the tutorial.

**The notebooks are generated.** Their sources are `reference/notebooks/*.py`, one file per notebook, with cells tagged `student` or `solution`. Edit the sources, then rebuild:

```bash
python reference/build_notebooks.py                    # the four notebooks, without outputs
python reference/build_notebooks.py --run-solutions    # also execute the two solutions notebooks (about 12 minutes on two CPU cores)
```

The student notebooks ship without outputs. The solutions notebooks ship executed. Two bonus answers train for several minutes each and sit behind a `RUN_SLOW_BONUS` switch in the solutions notebooks. The one of Notebook 1 is on, and the tests switch it off. The one of Notebook 2 (a CNN on the raw trajectory) is off: it is kept for reference, and the bonus itself uses a learned summary that trains in seconds.

**The data are generated.** Two scripts simulate the data in `data/`, train the networks that are handed to the students, and measure the benchmark targets. Each fixes its seeds and ends on a printed check of our reference solution against every benchmark.

```bash
python reference/make_reference_toy.py            # Notebook 1, about 5 minutes
python reference/make_reference_double_well.py    # Notebook 2, about 10 minutes
```

The networks are pickled, so run the scripts with the `sbi` version the notebooks pin.

**Tests.**

```bash
pip install pytest nbclient ipykernel
pytest tests/test_utils.py -q             # about a minute
pytest tests/test_notebooks.py -q -s      # executes all four notebooks, about 12 minutes
```

The notebook test checks that the student notebooks run from top to bottom with every challenge still open, and that the solutions notebooks pass every check. With `-s` it prints the slowest cells.

**Benchmark targets** are in `data/benchmarks.json`. The scores of Notebook 2 (bronze, silver, gold) are measured and written by `make_reference_double_well.py`; the other targets are set by hand in that file.

**If the repository moves**, change its address in two places: `GITHUB_REPO` in `reference/build_notebooks.py` (then rebuild the notebooks), and the links in this README.

## Sources

- Notebook 1 builds on [A tutorial on simulation-based inference](https://astroautomata.com/blog/simulation-based-inference/) by Miles Cranmer, which introduced this toy problem.
- Notebook 2 builds on [covinolab/sbi_for_smfe_tutorial](https://github.com/covinolab/sbi_for_smfe_tutorial): the same double-well model, without the linker and the measured extension.
- Both notebooks use the [`sbi`](https://github.com/sbi-dev/sbi) toolkit, version 0.27.0.
