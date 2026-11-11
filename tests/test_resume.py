"""A.7 Resume test: an interrupted and resumed run equals an uninterrupted run bit for bit
(global parameters, outer momentum, each site's AdamW state and sampler state)."""
import json

import pytest
import torch

from clinguard.engine.runner import RunConfig, run
from clinguard.engine.toy import make_regression_sites

CFG = dict(H=7, total_steps_per_site=42, weighting="fixed_q")       # 6 rounds
PHI0 = torch.zeros(8, dtype=torch.float64)


def assert_same(a, b):
    """Exact equality for nested dicts / lists / tensors."""
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("stop_after", [0, 2, 4])
def test_resumed_run_equals_uninterrupted_run(tmp_path, stop_after):
    full_sites = make_regression_sites(seed=5)
    phi_full, m_full = run(RunConfig(**CFG), full_sites, PHI0, str(tmp_path / "full"))

    part_dir = str(tmp_path / "part")
    run(RunConfig(**CFG), make_regression_sites(seed=5), PHI0, part_dir, stop_after_round=stop_after)
    resumed_sites = make_regression_sites(seed=5)                    # fresh objects, as in a new session
    phi_res, m_res = run(RunConfig(**CFG), resumed_sites, PHI0, part_dir)

    assert torch.equal(phi_full, phi_res)
    assert torch.equal(m_full, m_res)
    for a, b in zip(full_sites, resumed_sites):
        assert_same(a.state_dict(), b.state_dict())
        assert torch.equal(a.get_params(), b.get_params())

    def log(d):
        rows = [json.loads(l) for l in open(tmp_path / d / "rounds.jsonl")]
        for r in rows:
            for site_cost in r["cost"].values():
                site_cost.pop("train_sec")                           # wall time differs between runs
        return rows

    assert log("full") == log("part")                                # same 6 rounds, none duplicated


def test_resume_refuses_a_different_config(tmp_path):
    run(RunConfig(**CFG), make_regression_sites(seed=5), PHI0, str(tmp_path), stop_after_round=1)
    other = RunConfig(**{**CFG, "eta_out": 0.5})
    with pytest.raises(RuntimeError, match="different config"):
        run(other, make_regression_sites(seed=5), PHI0, str(tmp_path))


def test_finished_run_is_not_rerun(tmp_path):
    phi, m = run(RunConfig(**CFG), make_regression_sites(seed=5), PHI0, str(tmp_path))
    phi2, m2 = run(RunConfig(**CFG), make_regression_sites(seed=5), PHI0, str(tmp_path))
    assert torch.equal(phi, phi2) and torch.equal(m, m2)
    assert sum(1 for _ in open(tmp_path / "rounds.jsonl")) == 6
