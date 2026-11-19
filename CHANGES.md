# Changes since 14 November 2026

The repository as it was before these changes is tagged `before-changes-2026-11-19`.

- To look at it: `git checkout before-changes-2026-11-19`
- To undo one change on `main`: `git revert <commit>`. Each section below is one commit.

## Notebook 2: the ensemble and learned-summary bonuses

**What changed.**

- Bonus 3 (ensemble posteriors) has three steps and a check: look at two single networks and their ensemble for the blind trajectory; measure how much the ensemble gains over the mean score of its five members, with 200 and with 2,000 simulations; compare how often the 50% intervals contain the truth. It passes when the gain with 200 simulations is above 0.2 and at least twice the gain with 2,000.
- Bonus 4 (learned summaries) puts a small network (`FCEmbedding`) on the histogram of transitions between consecutive frames, standardized as one object (`z_score_x="structured"`), and passes above a score of 2.5. The histogram of positions alone follows as a second step: it is blind to D. The CNN on the raw trajectory stays in the solutions as a reference cell, switched off.
- New helpers in `tutorial_utils.py`: `log_probs_at_truth`, `interval_coverage`, `check_ensemble`, `check_learned_summary`. Their targets are in `data/benchmarks.json`, and two new tests cover them.

**Why.**

- Bonus 3 asked students to look at an ensemble, without a question to answer or a check. What an ensemble buys can be measured, and the answer depends on the training data: in our runs (ten networks per row) an ensemble of five gained 0.41 over its members with 200 simulations and 0.09 with 2,000.
- Bonus 4 asked to beat the hand-made statistics with a CNN on the raw trajectory. Our own answer reached 1.86 against 3.54, and a CNN trains for minutes to tens of minutes on two CPU cores. On the transition histogram a small network scores 3.35 to 3.43 (ten seeds) and trains in about ten seconds. The lesson stays the same and becomes reachable in an afternoon: a learned summary works once its input keeps what matters.

## Notebook 2: every score is the mean of three trainings

**What changed.**

- `train_and_score` trains the network three times, with the seeds `seed`, `seed + 1` and `seed + 2`, and reports the mean of the three scores. It returns the posterior of the first training. The medal targets are unchanged.
- `num_seeds=1` trains once. The cells that need only a posterior use it (Part 4, the members of the ensemble in bonus 3, the CNN reference cell), and so does `reference/make_reference_double_well.py`, so its benchmark numbers come out as before.
- Every training starts from its own copy of the untrained network. sbi's `posterior_nn` hands the same embedding network to every network it builds; without the copies, the second and third trainings of bonus 4 would start from the embedding network that the first one trained. A new test checks that the score is the mean of three independent trainings.
- The text of Part 2 says how the score is made and how much it still moves. In the table of bonus 4, the score of one network on the transition histogram reads 3.3 to 3.4, which also covers the trainings of the executed solutions.

**Why.** A single training is partly luck. Over ten training seeds our reference statistics scored 3.37 to 3.60 (standard deviation 0.07): one bronze, six silver, three gold. A medal, and "the best score in the room", partly measured the seed. The mean of three moves by about ±0.04: the reference statistics are silver in 97% of the 120 possible seed triples (gold in the other 3%), and our gold statistics are gold in all of them. The price is time: an attempt in Part 2 takes about 27 seconds instead of about 10 on two CPU cores, and the solutions notebook runs in about 6 minutes instead of 4.

## Notebook 1: the board counts three passes in a row

**What changed.** The check of Part 2 counts the passes in a row with the same number of simulations and prints the count under its verdict. The board asks for the lowest number of simulations that passes three runs in a row. Checking the same samples twice does not count as a new run, and the repair bonus, which reuses the check, has no board. The challenge itself still passes with a single pass. The solutions show how often each density estimator passed in our runs, and a new test covers the count.

**Why.** Near the record a pass is partly luck. In ten runs per setting, each with new simulations and a new training, the mixture density network passed 2 times with 100 simulations, 9 times with 200, 7 times with 500, and every time from 1,000 on. A record set by a single pass rewards the luckiest run rather than the best setting. The solutions also no longer say that this network "often passes with about 100".

## Running the notebooks without Colab

**What changed.** `INSTALL.md` sets up a virtual environment for Linux, macOS and Windows in a few commands, from `requirements.txt` (sbi 0.27.0, JupyterLab, ipywidgets). On Linux it installs the CPU build of PyTorch first. The README points to it, lists the new files and gives the current run times, and git ignores `.venv/`.

**Why.** Some students may not reach Colab. On Linux, `pip install sbi` alone pulls the CUDA build of PyTorch with its NVIDIA libraries, several gigabytes; the CPU build is all these notebooks need. ipywidgets draws sbi's progress bars in Jupyter. Tested on Linux from a fresh clone with Python 3.10 and 3.12: the installation takes about a minute and 1.6 GB, and all tests pass in both, including the four notebooks from top to bottom (about 13 minutes).

## Notebook 1: Part 5 trains on its own simulations

**What changed.**

- The worked cell of Part 5 simulates 2,000 training pairs of its own, `theta_nle` and `x_nle`, instead of training NLE on whatever `theta` and `x` Part 2 left behind. The text above it, the shortcut to the stored training set (`theta_nle, x_nle = tu.load_toy_training_set()`), the solution and the NRE bonus use the new names.
- The rejection ABC loop of Part 1 calls its draws `theta_proposal` and `x_proposal`, so running it again no longer replaces the training set.
- The training cell of bonus 3 (NPE on sets of points) starts from a fixed seed, and its text says how often the recipe passes: in six of ten trainings.
- One line of the Part 4 solutions showed as a large heading (one `#` too many). It is plain text again.
- The test file and the README give the time of the notebook tests as about 13 minutes, as measured.

**Why.**

- *Part 5.* With the board of Part 2, students end Part 2 with as few simulations as pass three times in a row, often 200 or fewer. NLE trained on a few hundred pairs gives intervals about 50% too wide (see the solutions of Part 5), so Part 5 would fail for a reason that has nothing to do with Part 5. Running the ABC loop again after Part 2 broke Part 5 the same way, by leaving single draws in `theta` and `x`. Now Part 5 starts from the same 2,000 pairs for everyone, the starting point its solutions describe.
- *Bonus 3.* Its training was not seeded, so its result depended on the random numbers the cells above it had drawn. The first change shifted them, and the executed solutions then failed the bonus's check: 34% too wide at n = 25, where the target is 25%. We trained the recipe with ten seeds. Up to n = 10 the widths stayed within 15% of the answer key every time. At n = 25 they passed in six trainings and were 26% to 59% too wide in four. The text said "within 10%". With seed 0, one of the six that pass, the result in the solutions no longer depends on the cells above it.
