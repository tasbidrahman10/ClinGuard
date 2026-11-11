"""End-to-end checks of the round loop on toy sites with conflicting objectives."""
import json

import torch

from clinguard.data.sampler import EpochSampler
from clinguard.engine.runner import RunConfig, run
from clinguard.engine.task import SiteTask
from clinguard.engine.toy import make_regression_sites

PHI0 = torch.zeros(8, dtype=torch.float64)


def weighted_loss(sites, phi):
    n = sum(s.n_train for s in sites)
    return sum(s.n_train / n * s.loss(phi) for s in sites)


def test_toy_sites_implement_the_interface():
    assert all(isinstance(s, SiteTask) for s in make_regression_sites())


def test_diloco_approaches_the_pooled_optimum(tmp_path):
    # The sites' objectives conflict, so the loss cannot reach zero. The reference is the pooled
    # least-squares optimum; the run must close at least 85% of the gap to it.
    sites = make_regression_sites(seed=1)
    x, y = torch.cat([s.x for s in sites]), torch.cat([s.y for s in sites])
    best = weighted_loss(sites, torch.linalg.lstsq(x, y).solution)
    start = weighted_loss(sites, PHI0)
    phi, _ = run(RunConfig(H=20, total_steps_per_site=400, weighting="fixed_q"), sites, PHI0, str(tmp_path))
    final = weighted_loss(sites, phi)
    assert best <= final < best + 0.15 * (start - best)


def test_fedavg_equals_the_weighted_average_of_local_models(tmp_path):
    # eta_out = 1, mu = 0:  phi_{t+1} = phi_t - sum_i w_i (phi_t - phi_i) = sum_i w_i phi_i
    sites = make_regression_sites(seed=2)
    q = [s.n_train / sum(x.n_train for x in sites) for s in sites]
    cfg = RunConfig(H=15, total_steps_per_site=15, eta_out=1.0, mu=0.0, weighting="fixed_q",
                    reset_inner_state=True, method="fedavg")
    phi, m = run(cfg, sites, PHI0, str(tmp_path))
    average = sum(qi * s.get_params() for qi, s in zip(q, sites))
    assert torch.allclose(phi, average, rtol=1e-12, atol=1e-12)


def test_round_log_and_byte_accounting(tmp_path):
    sites = make_regression_sites(seed=1)
    run(RunConfig(H=10, total_steps_per_site=30, weighting="uniform"), sites, PHI0, str(tmp_path))
    rows = [json.loads(l) for l in open(tmp_path / "rounds.jsonl")]
    assert [r["round"] for r in rows] == [0, 1, 2]
    meta = json.load(open(tmp_path / "run.json"))
    assert meta["n_params"] == 8 and len(meta["config_hash"]) == 64
    for r in rows:
        assert abs(sum(r["weights"].values()) - 1) < 1e-12
        for s in sites:
            c = r["cost"][s.name]
            assert c["download_bytes"] == 8 * 4                      # phi in fp32
            assert c["upload_bytes"] == 8 * 4 + 16                   # update in fp32 + loss/count summary
            assert c["examples"] == 10 * 8 and c["unique"] <= s.n_train


def test_sampler_uses_every_item_once_per_epoch_and_restores():
    s = EpochSampler(10, 4, seed=0)
    first_epoch = s.next() + s.next() + s.next()[:2]
    assert sorted(first_epoch) == list(range(10))
    a = EpochSampler(10, 4, seed=0)
    a.next()
    saved = a.state()
    expected = [a.next() for _ in range(6)]
    b = EpochSampler(10, 4, seed=99)
    b.load(json.loads(json.dumps(saved)))                            # state survives serialisation
    assert [b.next() for _ in range(6)] == expected
