"""Execute all four notebooks from top to bottom. Takes about 13 minutes.

    pytest tests/test_notebooks.py -q -s

The student notebooks must run without an error and leave their challenges open.
The solutions notebooks must pass every check cell. With -s the slowest cells are
printed: the target is under a minute per cell on a free Colab CPU runtime.
"""

import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
import build_notebooks  # noqa: E402

NAMES = ["01_sbi_toy_2d", "02_sbi_double_well"]


def printed(notebook):
    """Everything the code cells printed, as one string."""
    chunks = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                chunks.append(output.text)
            elif output.output_type == "error":
                chunks.append("\n".join(output.traceback))
    return "".join(chunks)


def on_stderr(notebook):
    """Everything written to stderr: progress bars are fine there, warnings are not."""
    return "".join(o.text for c in notebook.cells for o in c.get("outputs", [])
                   if o.output_type == "stream" and o.name == "stderr")


def report_slowest(notebook, name, top=4):
    timed = sorted(((seconds, i) for i, seconds in build_notebooks.cell_seconds(notebook)), reverse=True)
    for seconds, i in timed[:top]:
        first_line = notebook.cells[i].source.strip().splitlines()[0][:70]
        print(f"  {name}: cell {i:3d} took {seconds:5.1f} s   {first_line}")
    return timed[0][0] if timed else 0.0


@pytest.mark.parametrize("name", NAMES)
def test_notebooks_are_built_from_their_sources(name):
    """The committed notebooks hold exactly the cells of reference/notebooks/*.py."""
    for variant in ("student", "solution"):
        path, fresh = build_notebooks.build(name, variant, write=False)
        committed = nbformat.read(ROOT / path, as_version=4)
        assert [c.source for c in committed.cells] == [c.source for c in fresh.cells], (
            f"{path} is out of date: run python reference/build_notebooks.py")


@pytest.mark.parametrize("name", NAMES)
def test_student_notebook_runs_and_leaves_the_challenges_open(name):
    _, notebook = build_notebooks.build(name, "student", write=False)
    build_notebooks.execute(notebook, ROOT)
    text = printed(notebook)
    print()
    report_slowest(notebook, name)
    assert "Traceback" not in text
    assert text.count("❌ NOT YET ·") >= 4, "every check cell should report on its challenge"
    assert "✅ PASS" not in text, "no starting cell should pass its check"
    assert "Warning" not in on_stderr(notebook), "no warnings on screen"


@pytest.mark.parametrize("name", NAMES)
def test_solutions_notebook_passes_every_check(name):
    _, notebook = build_notebooks.build(name, "solution", write=False)
    for cell in notebook.cells:  # two bonus answers train for minutes: skipped here, run by the build
        cell.source = cell.source.replace("RUN_SLOW_BONUS = True", "RUN_SLOW_BONUS = False")
    build_notebooks.execute(notebook, ROOT / "solutions")
    text = printed(notebook)
    print()
    report_slowest(notebook, name + " (solutions)")
    assert "Traceback" not in text
    scorecard = [line for line in text.splitlines() if "checks passed." in line][-1]
    passed, total = (int(n) for n in scorecard.replace(" checks passed.", "").split(" of "))
    assert passed == total and total >= 4, scorecard
    bonus = [line for line in text.splitlines() if "· Bonus ·" in line]
    assert len(bonus) >= 2 and all("✅ PASS" in line for line in bonus), bonus
    if name == "02_sbi_double_well":
        assert "🥈 silver" in text and "🥇 gold" in text, "the reference is silver, the gold answer gold"


@pytest.mark.parametrize("name", NAMES)
def test_committed_solutions_notebook_is_executed_and_clean(name):
    """The solutions notebooks ship with their outputs: every check passed, nothing raised."""
    notebook = nbformat.read(ROOT / f"solutions/{name}_solutions.ipynb", as_version=4)
    outputs = [o for c in notebook.cells for o in c.get("outputs", [])]
    assert len(outputs) > 20, "run python reference/build_notebooks.py --run-solutions"
    assert not [o for o in outputs if o.output_type == "error"]
    scorecard = [line for line in printed(notebook).splitlines() if "checks passed." in line][-1]
    passed, total = (int(n) for n in scorecard.replace(" checks passed.", "").split(" of "))
    assert passed == total, scorecard
