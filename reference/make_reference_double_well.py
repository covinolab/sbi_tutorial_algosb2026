"""Reference data and networks for Notebook 2 (double well). For us, not for the students.

    python reference/make_reference_double_well.py

Run once, commit what it writes to data/, and the notebook loads it from there:

    dw_test_set.pt                 the common test set: 200 parameters and raw trajectories
    dw_trajectories.pt             practice trajectory (Part 1), blind trajectory (Part 3),
                                   three more blind trajectories for the bonus, one from a tilted well
    dw_hidden.pt                   the hidden parameters of those trajectories
    dw_reference_training_set.pt   10,000 trajectories reduced to the baseline and reference statistics
    dw_posterior_baseline.pkl      NPE on mean and standard deviation of x (bronze)
    dw_posterior_reference.pkl     NPE on our eight reference statistics (silver)
    dw_posterior_static.pkl        NPE on the four static statistics only (Part 4)
    benchmarks.json                the measured targets of the check cells (section "double_well")

The posteriors are pickled, so run this with the sbi version the notebooks pin.
The script ends on a printed check of the reference solution against every benchmark.
"""

import json
import os
import pickle
import sys
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # everything here is small: stay on the CPU

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_statistics as rs
import tutorial_utils as tu

warnings.filterwarnings("ignore")

# Hidden parameters: (barrier height in k_BT, log10 D), and the seed of each trajectory.
PRACTICE = dict(theta=[3.4, np.log10(0.55)], seed=11)
BLIND = dict(theta=[4.2, np.log10(0.32)], seed=21)
BONUS = [  # three more blind trajectories; exactly one comes from a tilted double well
    dict(theta=[3.6, np.log10(0.70)], seed=31, tilt=0.0),
    dict(theta=[3.0, np.log10(0.60)], seed=32, tilt=-0.75),  # right well lower by 1.5 k_BT
    dict(theta=[2.8, np.log10(0.35)], seed=33, tilt=0.0),
]
NUM_TEST, SEED_TEST = 200, 123
NUM_LARGE, SEED_LARGE = 10_000, 7
SCORE_SEEDS = range(5)


def simulate_one(theta, seed, tilt=0.0):
    return tu.simulate_double_well(torch.tensor([theta], dtype=torch.float32), seed=seed, tilt=tilt)[0]


def student_training_set():
    """Exactly what the notebook simulates live: 2,000 trajectories, both seeds 0."""
    torch.manual_seed(0)
    theta = tu.dw_prior().sample((2000,))
    return theta, tu.simulate_double_well(theta, seed=0)


def save_pickle(obj, name):
    with open(tu.DATA_DIR / name, "wb") as f:
        pickle.dump(obj, f)


def predictive_p_value(posterior, summary, x_observed, statistic, num=200, seed=0):
    """Posterior predictive check: how extreme is the observed statistic among simulations
    from the posterior? Returns the two-sided tail probability."""
    samples = posterior.sample((num,), x=tu.summarize(summary, x_observed), show_progress_bars=False)
    simulated = tu.simulate_double_well(samples, seed=seed).numpy()
    t_sim, t_obs = statistic(simulated), statistic(x_observed.numpy()[None])[0]
    tail = (t_sim >= t_obs).mean()
    return 2 * min(tail, 1 - tail)


def main():
    tu.DATA_DIR.mkdir(exist_ok=True)
    prior = tu.dw_prior()

    # ---- trajectories with hidden parameters --------------------------------------
    trajectories = {
        "practice": simulate_one(**PRACTICE),
        "blind": simulate_one(**BLIND),
        "bonus": torch.stack([simulate_one(**b) for b in BONUS]),
    }
    torch.save(trajectories, tu.DATA_DIR / "dw_trajectories.pt")
    torch.save({
        "practice": torch.tensor(PRACTICE["theta"], dtype=torch.float32),
        "blind": torch.tensor(BLIND["theta"], dtype=torch.float32),
        "bonus": torch.tensor([b["theta"] for b in BONUS], dtype=torch.float32),
        "bonus_tilted_index": int(np.flatnonzero([b["tilt"] != 0 for b in BONUS])[0]),
    }, tu.DATA_DIR / "dw_hidden.pt")

    # ---- common test set ------------------------------------------------------------
    torch.manual_seed(SEED_TEST)
    theta_test = prior.sample((NUM_TEST,))
    x_test = tu.simulate_double_well(theta_test, seed=SEED_TEST)
    torch.save({"theta": theta_test, "x": x_test}, tu.DATA_DIR / "dw_test_set.pt")

    # ---- benchmark scores, on the training set the students simulate ----------------
    theta, x = student_training_set()
    scores = {}
    for name, summary in (("baseline", rs.baseline_summary), ("reference", rs.reference_summary)):
        scores[name] = [tu.train_and_score(summary, theta, x, seed=s, num_seeds=1, verbose=False)[1]
                        for s in SCORE_SEEDS]
        print(f"{name:9s} statistics, 2,000 simulations: score "
              f"{np.mean(scores[name]):.2f} ± {np.std(scores[name]):.2f} over {len(SCORE_SEEDS)} seeds")
    benchmarks = tu.load_benchmarks()
    benchmarks["double_well"].update({
        "score_prior": round(float(-np.log(4.0)), 2),
        "score_baseline": round(float(np.mean(scores["baseline"])), 2),
        "score_reference": round(float(np.mean(scores["reference"])), 2),
        # Retraining the same statistics moves the score by about ±0.04, and so does the number
        # of CPU threads (seed 0: 3.45 on one thread, 3.54 on four). Bronze beats the baseline,
        # silver reaches the reference within that noise, gold beats its average by more.
        "score_bronze": round(float(np.mean(scores["baseline"])) + 0.1, 2),
        "score_silver": round(float(np.mean(scores["reference"])) - 0.15, 2),
        "score_gold": round(float(np.mean(scores["reference"])) + 0.05, 2),
    })
    with open(tu.DATA_DIR / "benchmarks.json", "w") as f:
        json.dump(benchmarks, f, indent=2)
        f.write("\n")

    # ---- fallback: 10,000 simulations, reduced, and three pre-trained posteriors ------
    torch.manual_seed(SEED_LARGE)
    theta_large = prior.sample((NUM_LARGE,))
    x_large = tu.simulate_double_well(theta_large, seed=SEED_LARGE)
    torch.save({
        "theta": theta_large,
        "s_baseline": tu.summarize(rs.baseline_summary, x_large),
        "s_reference": tu.summarize(rs.reference_summary, x_large),
    }, tu.DATA_DIR / "dw_reference_training_set.pt")
    posteriors = {}
    for name, summary in (("baseline", rs.baseline_summary), ("reference", rs.reference_summary),
                          ("static", rs.static_summary)):
        posteriors[name], score = tu.train_and_score(summary, theta_large, x_large, num_seeds=1, verbose=False)
        save_pickle(posteriors[name], f"dw_posterior_{name}.pkl")
        print(f"{name:9s} statistics, 10,000 simulations: score {score:.2f}")

    # ---- printed check: does the reference solution pass every benchmark? -------------
    b = benchmarks["double_well"]
    print(f"\nPart 2  bronze > {b['score_bronze']}, silver ≥ {b['score_silver']}, gold > {b['score_gold']}")

    posterior_2k, score_2k = tu.train_and_score(rs.reference_summary, theta, x, num_seeds=1, verbose=False)
    for label, posterior in (("2,000", posterior_2k), ("10,000", posteriors["reference"])):
        samples = posterior.sample((10_000,), x=tu.summarize(rs.reference_summary, trajectories["blind"]),
                                   show_progress_bars=False)
        area, inside = tu.region_90(samples, BLIND["theta"])
        print(f"Part 3  reference posterior ({label} simulations) on the blind trajectory: "
              f"truth inside = {inside}, area = {area:.2%} of the prior (target < {b['blind_area_max']:.0%}); "
              f"mean = {samples.mean(0).tolist()}, truth = {BLIND['theta']}")
        assert inside and area < b["blind_area_max"], "choose another blind trajectory"

    posterior_static, _ = tu.train_and_score(rs.static_summary, theta, x, num_seeds=1, verbose=False)
    for label, posterior in (("2,000", posterior_static), ("10,000", posteriors["static"])):
        width_dG, width_logD = tu.posterior_widths(posterior, rs.static_summary, verbose=False)
        print(f"Part 4  static posterior ({label} simulations): ΔG‡ width {width_dG:.0%} of the prior's "
              f"(target < {b['static_dG_width_max']:.0%}), log D width {width_logD:.0%} "
              f"(target > {b['static_logD_width_min']:.0%})")
        assert width_dG < b["static_dG_width_max"] and width_logD > b["static_logD_width_min"]

    # Bonus: a posterior predictive check on the share of time spent in the right well
    def time_right(x):
        return (x > 0).mean(axis=1)

    p_values = [predictive_p_value(posterior_2k, rs.reference_summary, x_b, time_right)
                for x_b in trajectories["bonus"]]
    for b_, x_b, p in zip(BONUS, trajectories["bonus"], p_values):
        print(f"Bonus   trajectory with tilt {b_['tilt']:+.2f}: {int(rs.count_hops(x_b.numpy()[None])[0])} hops, "
              f"{(x_b > 0).float().mean():.0%} of the time in the right well, predictive p = {p:.3f}")
    tilted = int(np.flatnonzero([b_["tilt"] != 0 for b_ in BONUS])[0])
    assert int(np.argmin(p_values)) == tilted and p_values[tilted] < 0.02, "the tilt should stand out"
    assert all(p > 0.1 for i, p in enumerate(p_values) if i != tilted), "the others should look ordinary"

    hops = {name: int(rs.count_hops(trajectories[name].numpy()[None])[0]) for name in ("practice", "blind")}
    print(f"\npractice trajectory: {hops['practice']} hops; blind trajectory: {hops['blind']} hops")


if __name__ == "__main__":
    main()
