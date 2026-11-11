"""A.1 Sign test: a descent-like pseudo-gradient must lower the loss when momentum is zero."""
import torch

from clinguard.engine import outer
from clinguard.engine.toy import ToySite


def test_scalar_quadratic_sign():
    a = torch.tensor([1.0], dtype=torch.float64)
    loss = lambda p: 0.5 * ((p - a) ** 2).sum().item()
    phi = torch.zeros(1, dtype=torch.float64)
    phi_local = torch.tensor([0.1], dtype=torch.float64)       # one local move towards a
    d = outer.pseudo_gradient(phi, phi_local)
    assert ((phi - a) * d).item() > 0                           # L'(phi) * d > 0: descent-like
    phi_next, _ = outer.nesterov_step(phi, torch.zeros_like(phi), d, eta_out=0.7, mu=0.0)
    assert loss(phi_next) < loss(phi)


def test_trained_site_update_lowers_its_loss():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 4, generator=g, dtype=torch.float64)
    y = x @ torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)
    site = ToySite("s", x, y, batch_size=8, lr=0.01, seed=0)
    phi = torch.zeros(4, dtype=torch.float64)
    site.set_params(phi)
    site.train(20)
    d = outer.pseudo_gradient(phi, site.get_params())
    phi_next, _ = outer.nesterov_step(phi, torch.zeros_like(phi), d, eta_out=0.5, mu=0.0)
    assert site.loss(phi_next) < site.loss(phi)
