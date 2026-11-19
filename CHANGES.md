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
