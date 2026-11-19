# Running the notebooks without Colab

For anyone who cannot use Colab. You need Python 3.10 or newer, about 2 GB of free disk space, and an internet connection while installing. No GPU is needed: everything runs on the CPU of a laptop.

## 1. Get the tutorial

```bash
git clone https://github.com/covinolab/sbi_tutorial_algosb2026.git
cd sbi_tutorial_algosb2026
```

Without git: on the repository's GitHub page, click **Code → Download ZIP**, unpack it, and open a terminal in the unpacked folder.

## 2. Create a virtual environment and install

Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux only: the CPU build of PyTorch, much smaller
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs `sbi` 0.27.0, which brings PyTorch, NumPy, SciPy and Matplotlib, plus JupyterLab. Keep `sbi` at 0.27.0: the pre-trained networks in `data/` were saved with it.

## 3. Check and start

```bash
python -c "import sbi, torch; print('sbi', sbi.__version__, '| torch', torch.__version__)"
jupyter lab
```

The first line should print `sbi 0.27.0`. In JupyterLab, open `01_sbi_toy_2d.ipynb`. Its first code cell installs packages on Colab only; on your computer it does nothing.

Next time, open a terminal in the folder, activate the environment again (`source .venv/bin/activate`, or `.venv\Scripts\Activate.ps1` on Windows) and run `jupyter lab`.

## If something fails

- **No `python3` or `py`.** Install Python from python.org. With conda instead: `conda create -n sbi-tutorial python=3.12`, `conda activate sbi-tutorial`, then the `pip install` lines of step 2.
- **PowerShell refuses the activation script.** Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use the Command Prompt with `.venv\Scripts\activate.bat`.
- **"Could not load the pre-trained network".** The installed `sbi` or PyTorch differs too much from the versions the networks were saved with. Check the versions (step 3) and tell an instructor.

Tested on Linux from a fresh clone with Python 3.10 (PyTorch 2.14.1, NumPy 2.2.6) and Python 3.12 (PyTorch 2.15.0, NumPy 2.5.5), both with sbi 0.27.0: the installation takes about a minute and 1.6 GB, and both notebooks run from top to bottom. Not yet tried on Windows or macOS.
