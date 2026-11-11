"""Toy sites for testing the engine on CPU in seconds. Not part of any reported experiment.

ToySite      least-squares regression on site-specific data, AdamW inner optimizer, mini-batches
             from an EpochSampler. Sites with different true coefficients have conflicting optima.
ScriptedSite returns pre-written pseudo-gradients, so outer-step results can be checked by hand.
"""
import torch

from clinguard.data.sampler import EpochSampler


def make_regression_sites(k=3, n=96, p=8, batch_size=8, lr=0.05, seed=0, spread=1.0, dtype=torch.float64):
    """k sites whose true coefficient vectors differ by `spread`, so their objectives conflict."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(p, generator=g, dtype=dtype)
    sites = []
    for i in range(k):
        n_i = n + 16 * i                                   # unequal sizes, so q != uniform
        a_i = base + spread * torch.randn(p, generator=g, dtype=dtype)
        x = torch.randn(n_i, p, generator=g, dtype=dtype)
        y = x @ a_i + 0.1 * torch.randn(n_i, generator=g, dtype=dtype)
        sites.append(ToySite(f"site{i}", x, y, batch_size, lr, seed=1000 * seed + i))
    return sites


class ToySite:
    def __init__(self, name, x, y, batch_size, lr, seed):
        self.name, self.x, self.y, self.lr = name, x, y, lr
        self.n_train = len(x)
        self.phi = torch.nn.Parameter(torch.zeros(x.shape[1], dtype=x.dtype))
        self.sampler = EpochSampler(self.n_train, batch_size, seed)
        self.reset_inner_state()

    def get_params(self):
        return self.phi.detach().clone()

    def set_params(self, phi):
        with torch.no_grad():
            self.phi.copy_(phi)

    def loss(self, phi=None):
        """Full-data loss at phi (default: the site's current parameters)."""
        with torch.no_grad():
            p = self.phi if phi is None else phi
            return 0.5 * ((self.x @ p - self.y) ** 2).mean().item()

    def train(self, steps):
        self.sampler.reset_seen()
        total = 0.0
        for _ in range(steps):
            idx = self.sampler.next()
            self.opt.zero_grad(set_to_none=True)
            loss = 0.5 * ((self.x[idx] @ self.phi - self.y[idx]) ** 2).mean()
            loss.backward()
            self.opt.step()
            total += loss.item()
        return dict(loss=total / steps, examples=self.sampler.drawn, unique=len(self.sampler.seen))

    def reset_inner_state(self):
        self.opt = torch.optim.AdamW([self.phi], lr=self.lr, weight_decay=0.0)

    def state_dict(self):
        return dict(opt=self.opt.state_dict(), sampler=self.sampler.state())

    def load_state_dict(self, state):
        self.opt.load_state_dict(state["opt"])
        self.sampler.load(state["sampler"])


class ScriptedSite:
    """Round t ends at phi_t - script[t], whatever phi_t is."""

    def __init__(self, name, script, n_train=1):
        self.name, self.n_train = name, n_train
        self.script = [torch.as_tensor(d, dtype=torch.float64).reshape(-1) for d in script]
        self.t = 0
        self.phi = None

    def get_params(self):
        return self.phi.clone()

    def set_params(self, phi):
        self.phi = phi.clone()

    def train(self, steps):
        self.phi = self.phi - self.script[self.t]
        self.t += 1
        return dict(loss=0.0, examples=0, unique=0)

    def reset_inner_state(self):
        pass

    def state_dict(self):
        return dict(t=self.t)

    def load_state_dict(self, state):
        self.t = state["t"]
