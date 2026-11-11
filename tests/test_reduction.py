"""A.3 Reduction test: the weighted pipeline collapses to the simpler methods exactly.

The ClinGuard half of this test (w = q with zero controller effect reproduces fixed-q DiLoCo)
is added at gate G6; the `external` weighting used here is the path the controller will feed.
"""
import pytest
import torch

from clinguard.engine import outer
from clinguard.engine.runner import RunConfig, run
from clinguard.engine.toy import ScriptedSite, make_regression_sites
from clinguard.engine.weighting import Weighting, fixed_q, uniform


def test_fixed_q_is_a_direct_weighted_sum():
    g = torch.Generator().manual_seed(0)
    deltas = [torch.randn(30, generator=g, dtype=torch.float64) for _ in range(3)]
    q = fixed_q([700, 900, 800])
    assert q == pytest.approx([700 / 2400, 900 / 2400, 800 / 2400])
    direct = q[0] * deltas[0] + q[1] * deltas[1] + q[2] * deltas[2]
    assert torch.allclose(outer.aggregate(deltas, q), direct, rtol=1e-13, atol=1e-13)


def test_equal_counts_give_uniform_weights():
    assert fixed_q([50, 50, 50, 50]) == uniform(4)


def test_bad_weights_are_rejected():
    d = [torch.zeros(2), torch.zeros(2)]
    with pytest.raises(ValueError):
        outer.aggregate(d, [0.7, 0.7])
    with pytest.raises(ValueError):
        outer.aggregate(d, [1.2, -0.2])
    with pytest.raises(ValueError):
        Weighting("external", [10, 10], external=[0.9, 0.3])


def test_external_weights_equal_to_q_reproduce_fixed_q_run(tmp_path):
    def final(weighting, external=()):
        sites = make_regression_sites(seed=3)
        cfg = RunConfig(H=10, total_steps_per_site=50, weighting=weighting, external_weights=external)
        return run(cfg, sites, torch.zeros(8, dtype=torch.float64), str(tmp_path / (weighting + str(len(external)))))

    q = tuple(fixed_q([s.n_train for s in make_regression_sites(seed=3)]))
    phi_q, m_q = final("fixed_q")
    phi_e, m_e = final("external", q)
    assert torch.equal(phi_q, phi_e) and torch.equal(m_q, m_e)
    phi_u, _ = final("uniform")
    assert not torch.equal(phi_q, phi_u)                    # sites have unequal sizes, so q != uniform


def test_uniform_run_matches_hand_computed_three_rounds(tmp_path):
    # Two sites, scalar phi, d = (0.5, 1.5) every round, so d_bar = 1 under uniform weights.
    # eta 0.7, mu 0.9:  m = 1, 1.9, 2.71   step = 0.7*(1 + 0.9*m) = 1.33, 1.897, 2.4073
    sites = [ScriptedSite("a", [[0.5]] * 3), ScriptedSite("b", [[1.5]] * 3)]
    cfg = RunConfig(H=1, total_steps_per_site=3, weighting="uniform")
    phi, m = run(cfg, sites, torch.zeros(1, dtype=torch.float64), str(tmp_path))
    assert m.item() == pytest.approx(2.71, abs=1e-12)
    assert phi.item() == pytest.approx(-(1.33 + 1.897 + 2.4073), abs=1e-12)


def test_fixed_q_run_matches_hand_computed_first_round(tmp_path):
    # q = (0.25, 0.75) -> d_bar = 0.25*0.5 + 0.75*1.5 = 1.25 -> phi_1 = -1.33 * 1.25
    sites = [ScriptedSite("a", [[0.5]], n_train=100), ScriptedSite("b", [[1.5]], n_train=300)]
    cfg = RunConfig(H=1, total_steps_per_site=1, weighting="fixed_q")
    phi, _ = run(cfg, sites, torch.zeros(1, dtype=torch.float64), str(tmp_path))
    assert phi.item() == pytest.approx(-1.33 * 1.25, abs=1e-12)
