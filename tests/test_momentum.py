"""A.2 Momentum test: our outer step equals torch.optim.SGD(nesterov=True), including the first
step, and Eq. (6) gives the direction actually taken."""
import pytest
import torch

from clinguard.engine import outer

ETA, MU = 0.7, 0.9


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-13), (torch.float32, 1e-5)])
def test_matches_torch_nesterov(dtype, tol):
    g = torch.Generator().manual_seed(0)
    phi0 = torch.randn(50, generator=g, dtype=dtype)
    d_bars = [torch.randn(50, generator=g, dtype=dtype) for _ in range(6)]

    p = torch.nn.Parameter(phi0.clone())
    opt = torch.optim.SGD([p], lr=ETA, momentum=MU, nesterov=True, dampening=0.0, weight_decay=0.0)
    phi, m = phi0.clone(), torch.zeros_like(phi0)
    for d_bar in d_bars:
        p.grad = d_bar.clone()
        opt.step()
        phi, m = outer.nesterov_step(phi, m, d_bar, ETA, MU)
        assert torch.allclose(phi, p.detach(), rtol=tol, atol=tol)


def test_first_step_is_1_33_times_the_average_change():
    phi = torch.zeros(3, dtype=torch.float64)
    d_bar = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    phi_next, m = outer.nesterov_step(phi, torch.zeros_like(phi), d_bar, ETA, MU)
    assert torch.allclose(m, d_bar)
    assert torch.allclose(phi_next, -ETA * (1 + MU) * d_bar)        # 0.7 * 1.9 = 1.33


def test_eq6_direction():
    g = torch.Generator().manual_seed(1)
    phi = torch.randn(20, generator=g, dtype=torch.float64)
    m_prev = torch.randn(20, generator=g, dtype=torch.float64)
    deltas = [torch.randn(20, generator=g, dtype=torch.float64) for _ in range(3)]
    w = [0.5, 0.3, 0.2]
    d_bar = outer.aggregate(deltas, w)
    phi_next, _ = outer.nesterov_step(phi, m_prev, d_bar, ETA, MU)
    v = MU ** 2 * m_prev + (1 + MU) * sum(wi * d for wi, d in zip(w, deltas))
    assert torch.allclose(outer.nesterov_direction(m_prev, d_bar, MU), v, rtol=1e-13, atol=1e-13)
    assert torch.allclose(phi_next, phi - ETA * v, rtol=1e-13, atol=1e-13)


def test_inputs_are_not_modified():
    phi, m, d = torch.ones(4), torch.ones(4), torch.ones(4)
    outer.nesterov_step(phi, m, d, ETA, MU)
    assert torch.equal(phi, torch.ones(4)) and torch.equal(m, torch.ones(4)) and torch.equal(d, torch.ones(4))
