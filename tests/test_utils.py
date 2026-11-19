"""Unit tests of tutorial_utils: the answer key, the simulator, the helpers, the check cells.

    pytest tests/test_utils.py -q
"""

import numpy as np
import pytest
import torch

import reference_statistics as rs
import tutorial_utils as tu


# ---------------------------------------------------------------- toy answer key
def test_exact_posterior_is_gaussian_around_the_observation():
    exact = tu.ExactPosterior(torch.tensor([1.0, -2.0]))
    samples = exact.sample((200_000,), seed=0)
    assert torch.allclose(samples.mean(0), torch.tensor([1.0, -2.0]), atol=0.01)
    assert torch.allclose(samples.std(0), torch.tensor([0.5, 0.5]), atol=0.01)
    assert torch.allclose(exact.std(), torch.tensor([0.5, 0.5]), atol=1e-4)
    assert torch.allclose(exact.width(0.9), torch.tensor([1.645, 1.645]), atol=1e-3)


def test_exact_posterior_narrows_with_iid_observations():
    x = torch.tensor([[3.0, -1.0], [3.4, -2.0], [2.6, -1.5], [3.0, -1.5]])
    exact = tu.ExactPosterior(x)
    assert np.allclose(exact.loc, [3.0, -1.5])
    assert exact.scale == pytest.approx(0.25)
    assert torch.allclose(exact.width(0.9), torch.tensor([1.645, 1.645]) / 2, atol=1e-3)


def test_exact_posterior_is_cut_off_at_the_prior_box():
    exact = tu.ExactPosterior(torch.tensor([5.5, 0.0]))
    samples = exact.sample((50_000,), seed=0)
    assert samples[:, 0].max() <= 5.0
    assert samples[:, 0].mean() < 5.0 - 0.2  # piled up against the wall, not centred on 5.5
    assert exact.log_prob(torch.tensor([5.2, 0.0])) == -torch.inf
    # the density integrates to one over the box
    axis = torch.linspace(-5, 5, 801)
    grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1)
    integral = exact.log_prob(grid).exp().sum() * (axis[1] - axis[0]) ** 2
    assert integral == pytest.approx(1.0, abs=0.02)


def test_exact_posterior_region_membership():
    exact = tu.ExactPosterior(torch.tensor([0.0, 0.0]))
    radius_95 = 0.5 * np.sqrt(-2 * np.log(0.05))  # 1.224
    assert exact.contains(torch.tensor([radius_95 - 0.05, 0.0]), 0.95)
    assert not exact.contains(torch.tensor([radius_95 + 0.05, 0.0]), 0.95)


def test_interval_width_from_samples():
    samples = torch.randn(200_000, 2, generator=torch.Generator().manual_seed(0)) * 0.5
    assert torch.allclose(tu.interval_width(samples), torch.tensor([1.645, 1.645]), atol=0.01)


# ---------------------------------------------------------------- toy check cells
def test_check_by_hand(capsys):
    x_o = torch.tensor([3.2, -1.1])
    tu.check_by_hand(x_o, x_o + torch.tensor([1.1, 0.0]), x_o)
    assert tu.RESULTS["Part 1 · best θ"] and tu.RESULTS["Part 1 · furthest θ"]
    tu.check_by_hand(torch.tensor([3.0, -1.5]), x_o + torch.tensor([1.1, 0.0]), x_o)
    assert not tu.RESULTS["Part 1 · best θ"]  # the truth is not the best fit
    tu.check_by_hand(x_o, x_o + torch.tensor([0.5, 0.0]), x_o)
    assert not tu.RESULTS["Part 1 · furthest θ"]  # not far enough
    tu.check_by_hand(x_o, x_o + torch.tensor([1.5, 0.0]), x_o)
    assert not tu.RESULTS["Part 1 · furthest θ"]  # outside the 95% region
    assert "NOT YET" in capsys.readouterr().out


def test_check_match_tells_right_from_wrong():
    x_o = torch.tensor([3.2, -1.1])
    exact = tu.ExactPosterior(x_o)
    assert tu.mismatch_score(exact.sample((2000,), seed=5), exact.sample((2000,), seed=1)) < 0.56
    tu.check_match(exact.sample((2000,), seed=5), 1000, x_o)
    assert tu.RESULTS["Part 2 · match"]
    too_wide = x_o + torch.randn(2000, 2) * 1.0
    tu.check_match(too_wide, 1000, x_o)
    assert not tu.RESULTS["Part 2 · match"]
    tu.check_match(torch.zeros(10, 2), 1000, x_o)  # too few samples: a message, no crash
    assert not tu.RESULTS["Part 2 · match"]


def test_check_iid_accepts_the_answer_key_and_rejects_drift():
    x_iid = tu.load_toy_observations()
    n_values = [1, 5, 10, 25, 50, 100]
    exact = torch.stack([tu.ExactPosterior(x_iid[:n]).width() for n in n_values])
    tu.check_iid(n_values, exact, 25)
    assert tu.RESULTS["Part 5 · pin θ down"]
    tu.check_iid(n_values, exact[:, 0], 25)  # one width per n is fine too
    assert tu.RESULTS["Part 5 · pin θ down"]
    tu.check_iid(n_values, exact * 1.3, 25)  # too wide
    assert not tu.RESULTS["Part 5 · pin θ down"]
    tu.check_iid(n_values, exact, 50)  # wrong reading
    assert not tu.RESULTS["Part 5 · pin θ down"]
    tu.check_iid([1, 10, 100], exact[:3], 25)  # too few values of n
    assert not tu.RESULTS["Part 5 · pin θ down"]
    few = [1, 2, 3, 4, 5]  # n = 25 cannot be read off a plot that ends at n = 5
    tu.check_iid(few, torch.stack([tu.ExactPosterior(x_iid[:n]).width() for n in few]), 25)
    assert not tu.RESULTS["Part 5 · pin θ down"]


def test_check_walls_and_calibration_use_the_hidden_answers():
    hidden = tu._load("toy_hidden.pt")
    tu.check_walls(hidden["wall_low"] + 0.2, hidden["wall_high"] - 0.2)
    assert tu.RESULTS["Part 3 · hidden walls"]
    tu.check_walls(hidden["wall_low"] - 0.3, hidden["wall_high"] + 0.3)  # exactly on the tolerance
    assert tu.RESULTS["Part 3 · hidden walls"]
    tu.check_walls([-5, -5], [5, 5])
    assert not tu.RESULTS["Part 3 · hidden walls"]
    tu.check_calibration(hidden["calibration"])
    assert tu.RESULTS["Part 4 · which network lies"]
    tu.check_calibration({"A": "fine", "B": "fine", "C": "fine"})
    assert not tu.RESULTS["Part 4 · which network lies"]


def test_check_bimodal():
    both = torch.cat([torch.randn(1000, 2) + 3, torch.randn(1000, 2) - 3])
    tu.check_bimodal(both)
    assert tu.RESULTS["Bonus · bend the simulator"]
    tu.check_bimodal(torch.randn(1000, 2) + 3)
    assert not tu.RESULTS["Bonus · bend the simulator"]
    tu.check_bimodal(torch.rand(4000, 2) * 10 - 5)  # the prior is symmetric too, but has no gap
    assert not tu.RESULTS["Bonus · bend the simulator"]
    tu.check_bimodal(torch.randn(4000, 2) * 0.3)  # one blob at zero
    assert not tu.RESULTS["Bonus · bend the simulator"]


# ---------------------------------------------------------------- double well
def test_force_is_minus_the_gradient_of_the_potential():
    x = np.linspace(-1.8, 1.8, 3601)
    numeric = -np.gradient(tu.double_well_potential(x, 4.0), x)
    assert np.allclose(tu.double_well_force(x, 4.0)[5:-5], numeric[5:-5], atol=0.05)
    assert tu.double_well_potential(0.0, 4.0) == pytest.approx(4.0)
    assert tu.double_well_potential(np.array([-1.0, 1.0]), 4.0) == pytest.approx([0.0, 0.0])
    # a tilt adds a constant force
    assert np.allclose(tu.double_well_force(x, 4.0, tilt=0.7), tu.double_well_force(x, 4.0) - 0.7)


def test_simulator_shape_seed_and_physics():
    theta = torch.tensor([[2.0, 0.0]] * 40)
    x = tu.simulate_double_well(theta, seed=1)
    assert x.shape == (40, 5000) and x.dtype == torch.float32
    assert torch.equal(x, tu.simulate_double_well(theta, seed=1))
    assert not torch.equal(x, tu.simulate_double_well(theta, seed=2))
    # Boltzmann statistics: the barrier region |x| < 0.5 holds about 11% of the frames at 2 k_BT
    assert (x.abs() < 0.5).float().mean().item() == pytest.approx(0.11, abs=0.015)
    # both wells are visited equally often
    assert x.mean().abs() < 0.1


def test_hop_count_ignores_recrossings():
    x = np.array([[-1.0, -0.2, 0.2, -0.2, -1.0, 0.3, 1.0, 1.0, 0.1, -0.1, 0.9, -0.8]], np.float32)
    assert rs.count_hops(x).tolist() == [2]


def test_static_statistics_do_not_see_the_time_order():
    x = tu.simulate_double_well(torch.tensor([[3.0, -0.3]] * 3), seed=0).numpy()
    shuffled = np.random.default_rng(0).permuted(x, axis=1)
    assert np.allclose(rs.static_summary(x), rs.static_summary(shuffled), atol=1e-5)
    assert not np.allclose(rs.dynamic_summary(x), rs.dynamic_summary(shuffled), atol=1e-2)
    assert rs.reference_summary(x).shape == (3, 8)


def test_summarize_checks_the_students_function():
    x = torch.randn(7, 5000)
    assert tu.summarize(rs.baseline_summary, x).shape == (7, 2)
    assert tu.summarize(rs.baseline_summary, x[0]).shape == (1, 2)
    assert tu.summarize(lambda x: x.mean(axis=1), x).shape == (7, 1)
    with pytest.raises(ValueError, match="one row per trajectory"):
        tu.summarize(lambda x: x.mean(), x)
    with pytest.raises(ValueError, match="NaN or infinite"):
        tu.summarize(lambda x: np.stack([x.mean(axis=1), np.log(0 * x[:, 0])], axis=1), x)


def test_region_90_of_a_known_gaussian():
    rng = np.random.default_rng(0)
    samples = np.array([4.0, -0.5]) + rng.normal(size=(20_000, 2)) * np.array([0.2, 0.05])
    area, inside = tu.region_90(samples, truth=[4.1, -0.52])
    exact_area = np.pi * (-2 * np.log(0.1)) * 0.2 * 0.05 / 4.0  # ellipse / prior box
    assert area == pytest.approx(exact_area, rel=0.15)
    assert inside
    assert not tu.region_90(samples, truth=[5.0, -0.5])[1]


def test_dw_check_cells(capsys):
    hidden = tu._load("dw_hidden.pt")
    dG, D = hidden["practice"][0].item(), 10 ** hidden["practice"][1].item()
    tu.check_practice([(3.0, 0.1), (dG + 0.3, D * 1.3)])
    assert tu.RESULTS["Part 1 · by hand"]
    capsys.readouterr()
    tu.check_practice([(dG + 1.0, D)])
    assert not tu.RESULTS["Part 1 · by hand"]
    said = capsys.readouterr().out  # no hint on which of the two parameters is off
    assert "ΔG‡ within" not in said and "D within" not in said and "D not within" not in said
    tu.check_practice([(dG, D)] * 11)  # right, but too many tries
    assert not tu.RESULTS["Part 1 · by hand"]

    b = tu.load_benchmarks()["double_well"]
    tu.check_score(b["score_bronze"] - 0.1)
    assert not tu.RESULTS["Part 2 · summary statistics"]
    tu.check_score(b["score_gold"] + 0.1)
    assert tu.RESULTS["Part 2 · summary statistics"]

    truth = hidden["blind"]
    tight = truth + torch.randn(5000, 2) * torch.tensor([0.15, 0.04])
    tu.check_blind(tight)
    assert tu.RESULTS["Part 3 · blind trajectory"]
    tu.check_blind(tight + torch.tensor([1.0, 0.0]))  # narrow but wrong
    assert not tu.RESULTS["Part 3 · blind trajectory"]
    tu.check_blind(tu.dw_prior().sample((5000,)))  # contains the truth but learned nothing
    assert not tu.RESULTS["Part 3 · blind trajectory"]

    tu.check_sparse(score_200=2.9, score_200_start=1.7, score_2000=3.5)  # most of the drop won back
    assert tu.RESULTS["Bonus · sparse training data"]
    tu.check_sparse(score_200=1.8, score_200_start=1.7, score_2000=3.5)  # above bronze, nothing won back
    assert not tu.RESULTS["Bonus · sparse training data"]
    tu.check_sparse(score_200=-1.3, score_200_start=-2.5, score_2000=-1.0)  # won back, but below bronze
    assert not tu.RESULTS["Bonus · sparse training data"]

    tu.check_ensemble(gain_200=0.41, gain_2000=0.09)  # what we measured
    assert tu.RESULTS["Bonus · ensemble"]
    tu.check_ensemble(gain_200=0.15, gain_2000=0.05)  # three times larger, but too small
    assert not tu.RESULTS["Bonus · ensemble"]
    tu.check_ensemble(gain_200=0.30, gain_2000=0.20)  # large, but not twice the gain with 2,000
    assert not tu.RESULTS["Bonus · ensemble"]
    capsys.readouterr()
    tu.check_ensemble(gain_200=-0.20, gain_2000=0.05)  # log densities averaged instead of densities
    assert not tu.RESULTS["Bonus · ensemble"] and "Average the densities" in capsys.readouterr().out

    tu.check_learned_summary(b["learned_score_min"] + 0.1)
    assert tu.RESULTS["Bonus · learned summaries"]
    tu.check_learned_summary(1.7)  # a CNN on the raw trajectory
    assert not tu.RESULTS["Bonus · learned summaries"]

    tu.check_wrong_simulator(int(hidden["bonus_tilted_index"]))
    assert tu.RESULTS["Bonus · wrong simulator"]
    tu.check_wrong_simulator((int(hidden["bonus_tilted_index"]) + 1) % 3)
    assert not tu.RESULTS["Bonus · wrong simulator"]


def test_fallback_posteriors_load_and_sample():
    x_blind = tu.load_trajectory("blind")
    for name in ("baseline", "reference", "static"):
        samples = tu.load_fallback_posterior(name).sample((500,), x=x_blind)
        assert samples.shape == (500, 2)
    samples = tu.load_black_box().sample((300,), x=[0.0, 0.0])
    assert samples.shape == (300, 2)
    widths = tu.posterior_widths(tu.load_fallback_posterior("static"), None, verbose=False)
    assert widths[0] < 0.25 and widths[1] > 0.85  # the fallback brings its own statistics
    reduced = tu.load_reference_training_set()
    assert reduced["theta"].shape == (10_000, 2) and reduced["s_reference"].shape == (10_000, 8)
    networks = tu.load_three_networks()
    assert sorted(networks) == ["A", "B", "C"]


# ---------------------------------------------------------------- awkward input
def test_check_iid_explains_malformed_input(capsys):
    tu.check_iid([1, 2, 5, 10, 25, 50, 100], [1.0, 0.9, 0.5], 25)  # three widths for seven n
    assert not tu.RESULTS["Part 5 · pin θ down"]
    assert "one width per n" in capsys.readouterr().out
    tu.check_iid([1, 2, 5, 10, 25], [1.6, 1.2, 0.74, 0.52, 0.33], None)  # no reading of n yet
    assert not tu.RESULTS["Part 5 · pin θ down"]
    assert "n_needed" in capsys.readouterr().out


def test_black_box_takes_one_observation_at_a_time():
    box = tu.load_black_box()
    with pytest.raises(ValueError, match="one observation"):
        box.sample((10,), x=torch.zeros(5, 2))


class _LeakyPosterior:
    """Stands in for a network that puts no probability inside its prior for this observation."""

    def __init__(self):
        from sbi.utils import BoxUniform

        self.prior = BoxUniform(low=torch.tensor([-1.0, -2.0]), high=torch.tensor([1.0, 2.0]))

    def sample(self, sample_shape, x=None, show_progress_bars=True, reject_outside_prior=True,
               max_sampling_time=None, return_partial_on_timeout=False):
        if reject_outside_prior:  # what sbi 0.27 does when nothing is accepted within the time limit
            raise RuntimeError("Sampling time exceeded and no samples were accepted.")
        return torch.full((int(np.prod(sample_shape)), 2), 7.0)


def test_black_box_never_hangs_and_never_comes_back_short(capsys):
    samples = tu.BlackBox(_LeakyPosterior()).sample((50,), x=[9.0, 9.0])
    assert samples.shape == (50, 2)
    assert torch.equal(samples.max(dim=0).values, torch.tensor([1.0, 2.0]))  # moved onto the edge
    assert "no probability inside" in capsys.readouterr().out


def test_ensemble_helpers_score_and_cover(capsys):
    """Two quickly trained networks: the per-trajectory log densities average to the score, the ensemble
    is not below the mean of its members, and the coverage of a list is that of the pooled samples."""
    torch.manual_seed(0)
    theta = tu.dw_prior().sample((200,))
    x = tu.simulate_double_well(theta, seed=0).numpy()
    members, scores = zip(*[tu.train_and_score(rs.reference_summary, theta, x, seed=seed, num_seeds=1,
                                               verbose=False, max_num_epochs=30) for seed in range(2)])
    log_probs = torch.stack([tu.log_probs_at_truth(m, rs.reference_summary) for m in members])
    assert log_probs.shape == (2, 200)
    assert np.allclose(log_probs.mean(dim=1).numpy(), scores, atol=1e-4)
    score_ensemble = (torch.logsumexp(log_probs, dim=0) - np.log(2)).mean().item()
    assert score_ensemble >= np.mean(scores) - 1e-6

    wide = tu.interval_coverage(members[0], rs.reference_summary, level=0.9, num_samples=100)
    narrow = tu.interval_coverage(members[0], rs.reference_summary, level=0.5, num_samples=100, verbose=False)
    assert all(0.0 <= c <= 1.0 for c in wide + narrow) and all(w >= n for w, n in zip(wide, narrow))
    assert "single network" in capsys.readouterr().out
    both = tu.interval_coverage(list(members), rs.reference_summary, level=0.9, num_samples=100)
    assert "ensemble of 2" in capsys.readouterr().out and all(0.0 <= c <= 1.0 for c in both)


def test_score_is_the_mean_of_three_trainings_and_the_posterior_the_first(monkeypatch):
    """The medal score averages three independent trainings, also when the builder holds an embedding
    network, and the posterior returned is that of the first."""
    from sbi.neural_nets import posterior_nn
    from sbi.neural_nets.embedding_nets import FCEmbedding

    torch.manual_seed(0)
    theta = tu.dw_prior().sample((200,))
    x = tu.simulate_double_well(theta, seed=0).numpy()

    def builder():
        torch.manual_seed(1)  # the same untrained embedding network every time
        embedding = FCEmbedding(input_dim=8, output_dim=4, num_layers=1, num_hiddens=16)
        return posterior_nn("nsf", num_transforms=1, embedding_net=embedding, z_score_x="structured")

    settings = dict(verbose=False, max_num_epochs=20)
    singles = [tu.train_and_score(rs.reference_summary, theta, x, seed=seed, num_seeds=1,
                                  density_estimator=builder(), **settings) for seed in range(3)]
    first, score = tu.train_and_score(rs.reference_summary, theta, x, num_seeds=3,
                                      density_estimator=builder(), **settings)
    assert score == pytest.approx(np.mean([s for _, s in singles]), abs=1e-4)
    single_log_probs = tu.log_probs_at_truth(singles[0][0], rs.reference_summary)
    assert torch.allclose(tu.log_probs_at_truth(first, rs.reference_summary), single_log_probs, atol=1e-4)

    # Without the copies, the second training would start from the embedding network the first trained
    monkeypatch.setattr(tu, "_untrained_copy", lambda density_estimator: density_estimator)
    _, score_shared = tu.train_and_score(rs.reference_summary, theta, x, num_seeds=2,
                                         density_estimator=builder(), **settings)
    assert score_shared != pytest.approx(np.mean([s for _, s in singles[:2]]), abs=1e-4)
