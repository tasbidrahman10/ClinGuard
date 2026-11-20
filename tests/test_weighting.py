"""Matched-control weightings (plan section 9, Tier 1 "alternative weights"): loss-weighted and random.

Both solve  max_w  sum_i w_i s_i - KL(w || q)  on the simplex inside the box (1 - rho) q <= w <= (1 + rho) q,
the same geometry as the ClinGuard controller (manual Eq. 8 and 10).
"""
import json
import math

import pytest
import torch

from clinguard.engine.runner import RunConfig, run
from clinguard.engine.toy import ScriptedSite, make_regression_sites
from clinguard.engine.weighting import Weighting, fixed_q, tilt_in_box

PHI0 = torch.zeros(8, dtype=torch.float64)


def objective(w, q, s):
    return sum(wi * si for wi, si in zip(w, s)) - sum(wi * math.log(wi / qi) for wi, qi in zip(w, q))


def in_box(w, q, rho, tol=1e-12):
    return all((1 - rho) * qi - tol <= wi <= (1 + rho) * qi + tol for wi, qi in zip(w, q))


@pytest.mark.parametrize("seed", range(5))
def test_tilt_is_the_exact_optimum(seed):
    g = torch.Generator().manual_seed(seed)
    q = fixed_q((torch.rand(6, generator=g) * 900 + 100).tolist())
    s = (torch.randn(6, generator=g, dtype=torch.float64) * 0.6).tolist()
    rho = 0.4
    w = tilt_in_box(q, s, rho)
    assert sum(w) == pytest.approx(1.0, abs=1e-12) and in_box(w, q, rho)

    # KKT certificate: one nu shared by every interior weight; clipped weights sit on the correct side.
    upper = [abs(wi - (1 + rho) * qi) < 1e-12 for wi, qi in zip(w, q)]
    lower = [abs(wi - (1 - rho) * qi) < 1e-12 for wi, qi in zip(w, q)]
    nus = [si - math.log(wi / qi) for wi, qi, si, u, l in zip(w, q, s, upper, lower) if not (u or l)]
    assert nus, "test case should have at least one interior weight"
    nu = nus[0]
    assert all(x == pytest.approx(nu, abs=1e-9) for x in nus)
    for si, u, l in zip(s, upper, lower):
        if u:
            assert si - nu >= math.log(1 + rho) - 1e-9
        if l:
            assert si - nu <= math.log(1 - rho) + 1e-9

    # No other feasible weight vector does better.
    best = objective(w, q, s)
    for k in range(200):
        g2 = torch.Generator().manual_seed(10_000 + 100 * seed + k)
        other = tilt_in_box(q, torch.randn(6, generator=g2, dtype=torch.float64).tolist(), rho)
        assert objective(other, q, s) <= best + 1e-12


def test_tilt_edge_cases():
    q = [0.2, 0.3, 0.5]
    assert tilt_in_box(q, [0.0, 0.0, 0.0], 0.5) == pytest.approx(q, abs=1e-12)     # no signal -> q
    assert tilt_in_box(q, [3.0, 3.0, 3.0], 0.5) == pytest.approx(q, abs=1e-12)     # shift-invariant
    assert tilt_in_box(q, [5.0, -1.0, 2.0], 0.0) == q                              # rho = 0 -> exactly q
    w = tilt_in_box(q, [50.0, -50.0, 0.0], 0.5)                                    # strong signal -> box edges
    assert w[0] == pytest.approx(1.5 * 0.2) and w[1] == pytest.approx(0.5 * 0.3)
    assert sum(w) == pytest.approx(1.0, abs=1e-12)
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError):
            tilt_in_box(q, [0.0, 0.0, 0.0], bad)
    with pytest.raises(ValueError):
        tilt_in_box(q, [0.0, float("nan"), 0.0], 0.5)


def test_loss_weighting_favours_the_struggling_site():
    wt = Weighting("loss", [100, 100, 100], rho=0.5, temperature=1.0)
    w = wt(dict(round=0, calib_loss=[1.0, 1.0, 2.0]))
    assert w[2] > 1 / 3 > w[0] and w[0] == pytest.approx(w[1], abs=1e-12)
    assert wt(dict(round=0, calib_loss=[0.7, 0.7, 0.7])) == pytest.approx([1 / 3] * 3, abs=1e-12)
    hot = Weighting("loss", [100, 100, 100], rho=0.5, temperature=1e6)
    assert hot(dict(round=0, calib_loss=[1.0, 1.0, 2.0])) == pytest.approx([1 / 3] * 3, abs=1e-6)
    with pytest.raises(ValueError):
        wt(dict(round=0))
    with pytest.raises(ValueError):
        Weighting("loss", [100, 100], temperature=0.0)


def test_random_weighting_is_reproducible_and_bounded():
    a = Weighting("random", [100, 200, 300], rho=0.5, sigma=0.3, seed=11)
    b = Weighting("random", [100, 200, 300], rho=0.5, sigma=0.3, seed=11)
    draws = [a(dict(round=t)) for t in range(20)]
    assert draws == [b(dict(round=t)) for t in range(20)]                    # same seed, same weights
    assert len({tuple(round(x, 12) for x in d) for d in draws}) == 20       # a fresh draw every round
    assert draws != [Weighting("random", [100, 200, 300], rho=0.5, sigma=0.3, seed=22)(dict(round=t))
                     for t in range(20)]
    for d in draws:
        assert sum(d) == pytest.approx(1.0, abs=1e-12) and in_box(d, a.q, 0.5)
    flat = Weighting("random", [100, 200, 300], rho=0.5, sigma=0.0, seed=11)
    assert flat(dict(round=3)) == pytest.approx(flat.q, abs=1e-12)


def test_loss_weighting_run_matches_hand_computed_first_round(tmp_path):
    # q = (0.5, 0.5); calibration losses (0, ln 2), T = 1 -> w proportional to (1, 2) = (1/3, 2/3), inside
    # the box [0.25, 0.75]. d = (0.5, 1.5) -> d_bar = 1/6 + 1 = 7/6 -> phi_1 = -1.33 * 7/6.
    sites = [ScriptedSite("a", [[0.5]], calib_script=[0.0]),
             ScriptedSite("b", [[1.5]], calib_script=[math.log(2)])]
    cfg = RunConfig(H=1, total_steps_per_site=1, weighting="loss", rho=0.5, loss_temperature=1.0)
    phi, _ = run(cfg, sites, torch.zeros(1, dtype=torch.float64), str(tmp_path))
    assert phi.item() == pytest.approx(-1.33 * 7 / 6, abs=1e-12)
    row = json.loads(open(tmp_path / "rounds.jsonl").readline())
    assert row["weights"] == pytest.approx({"a": 1 / 3, "b": 2 / 3}, abs=1e-12)
    assert row["calib_loss"] == pytest.approx({"a": 0.0, "b": math.log(2)})


def test_measuring_calibration_loss_does_not_disturb_training(tmp_path):
    # With rho = 0 loss weighting returns exactly q, so the run must equal fixed-q DiLoCo bit for bit:
    # evaluating the calibration loss may not touch optimizer, sampler or random state.
    def final(**kw):
        cfg = RunConfig(H=10, total_steps_per_site=50, **kw)
        return run(cfg, make_regression_sites(seed=3), PHI0, str(tmp_path / kw["weighting"]))

    phi_q, m_q = final(weighting="fixed_q")
    phi_l, m_l = final(weighting="loss", rho=0.0)
    assert torch.equal(phi_q, phi_l) and torch.equal(m_q, m_l)


@pytest.mark.parametrize("kind", ["loss", "random"])
def test_matched_controls_stay_in_the_box_and_resume_exactly(tmp_path, kind):
    cfg = RunConfig(H=7, total_steps_per_site=42, weighting=kind, rho=0.5, loss_temperature=0.5,
                    random_sigma=0.3)
    phi_full, m_full = run(cfg, make_regression_sites(seed=5), PHI0, str(tmp_path / "full"))
    run(cfg, make_regression_sites(seed=5), PHI0, str(tmp_path / "part"), stop_after_round=2)
    phi_res, m_res = run(cfg, make_regression_sites(seed=5), PHI0, str(tmp_path / "part"))
    assert torch.equal(phi_full, phi_res) and torch.equal(m_full, m_res)

    def log(d):
        rows = [json.loads(l) for l in open(tmp_path / d / "rounds.jsonl")]
        for r in rows:
            for c in r["cost"].values():
                c.pop("train_sec"), c.pop("calib_sec")
        return rows

    rows = log("full")
    assert rows == log("part")
    q = fixed_q([s.n_train for s in make_regression_sites(seed=5)])
    ws = [list(r["weights"].values()) for r in rows]
    assert all(in_box(w, q, 0.5) and sum(w) == pytest.approx(1.0, abs=1e-12) for w in ws)
    assert any(max(abs(a - b) for a, b in zip(w, q)) > 1e-3 for w in ws)       # the weights actually move
