"""Site weights for the outer aggregation.

Kinds:
  uniform    w_i = 1/K                                                  (vanilla DiLoCo)
  fixed_q    w_i = q_i, proportional to eligible training-study counts (primary comparator)
  external   weights supplied from outside (the path the ClinGuard controller feeds)
  loss       matched control: tilt q towards sites with a higher calibration loss
  random     matched control: random tilt of q, same box, fresh draw every round

`loss` and `random` use the same geometry as ClinGuard (manual Eq. 8 and 10): weights stay close to q
in KL divergence and inside the box (1 - rho) q_i <= w_i <= (1 + rho) q_i. For scores s_i they solve

    max_w  sum_i w_i s_i - KL(w || q)    subject to  sum_i w_i = 1,  (1 - rho) q <= w <= (1 + rho) q

whose exact solution is w_i = clip(q_i exp(s_i - nu), (1 - rho) q_i, (1 + rho) q_i) with nu set so the
weights sum to 1 (`tilt_in_box`). Scores: loss -> s_i = calib_loss_i / T;  random -> s_i = sigma z_i,
z_i ~ N(0, 1) drawn from a generator seeded by (run seed, round), so a resumed run redraws the same
weights without stored state. The random spread sigma is matched to ClinGuard's weights in Phase 6.
"""
import hashlib
import math

import torch

KINDS = ("uniform", "fixed_q", "external", "loss", "random")


def uniform(k):
    return [1.0 / k] * k


def fixed_q(train_counts):
    """q_i proportional to site i's eligible training-study count."""
    if any(c <= 0 for c in train_counts):
        raise ValueError(f"every site needs a positive training count, got {list(train_counts)}")
    total = float(sum(train_counts))
    return [c / total for c in train_counts]


def tilt_in_box(q, scores, rho, iters=200):
    """w_i = clip(q_i exp(s_i - nu), (1-rho) q_i, (1+rho) q_i), with nu found by bisection so sum w = 1.

    The sum is non-increasing in nu, equals 1 + rho when every weight sits at its upper bound and
    1 - rho when every weight sits at its lower bound, so a root always exists for 0 <= rho < 1.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho must lie in [0, 1), got {rho}")
    if len(q) != len(scores):
        raise ValueError(f"{len(q)} baseline weights but {len(scores)} scores")
    if any(not math.isfinite(s) for s in scores):
        raise ValueError(f"scores must be finite, got {list(scores)}")
    if rho == 0.0:
        return list(q)
    lo_b = [(1 - rho) * qi for qi in q]
    hi_b = [(1 + rho) * qi for qi in q]

    def weights(nu):
        return [min(max(qi * math.exp(si - nu), l), h) for qi, si, l, h in zip(q, scores, lo_b, hi_b)]

    nu_lo = min(scores) - math.log(1 + rho) - 1.0       # every weight at its upper bound: sum = 1 + rho
    nu_hi = max(scores) - math.log(1 - rho) + 1.0       # every weight at its lower bound: sum = 1 - rho
    for _ in range(iters):
        mid = 0.5 * (nu_lo + nu_hi)
        if sum(weights(mid)) > 1.0:
            nu_lo = mid
        else:
            nu_hi = mid
    return weights(0.5 * (nu_lo + nu_hi))


def round_seed(run_seed, round_index, purpose="random_weights"):
    """A per-round seed that depends only on (run seed, round, purpose)."""
    digest = hashlib.sha256(f"{run_seed}|{round_index}|{purpose}".encode()).hexdigest()
    return int(digest[:15], 16)


class Weighting:
    """Callable returning this round's weights. `ctx` holds what the coordinator knows that round:
    `round` (int) and, for loss weighting, `calib_loss` (one float per site, measured at phi_t)."""

    def __init__(self, kind, train_counts, external=None, rho=0.5, temperature=1.0, sigma=0.1, seed=0):
        if kind not in KINDS:
            raise ValueError(f"unknown weighting '{kind}'")
        self.kind = kind
        self.q = fixed_q(train_counts)
        self.k = len(train_counts)
        self.rho, self.temperature, self.sigma, self.seed = rho, temperature, sigma, seed
        if kind == "external":
            if external is None or len(external) != self.k:
                raise ValueError("external weighting needs one weight per site")
            if min(external) < 0 or abs(sum(external) - 1.0) > 1e-9:
                raise ValueError(f"external weights must lie on the simplex, got {list(external)}")
        if kind in ("loss", "random") and not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must lie in [0, 1), got {rho}")
        if kind == "loss" and temperature <= 0:
            raise ValueError(f"loss weighting needs a positive temperature, got {temperature}")
        if kind == "random" and sigma < 0:
            raise ValueError(f"random weighting needs sigma >= 0, got {sigma}")
        self.external = list(external) if external is not None else None

    @property
    def needs_calibration_loss(self):
        return self.kind == "loss"

    def __call__(self, ctx=None):
        ctx = ctx or {}
        if self.kind == "uniform":
            return uniform(self.k)
        if self.kind == "fixed_q":
            return list(self.q)
        if self.kind == "external":
            return list(self.external)
        if self.kind == "loss":
            losses = ctx.get("calib_loss")
            if losses is None or len(losses) != self.k:
                raise ValueError("loss weighting needs one calibration loss per site in ctx['calib_loss']")
            return tilt_in_box(self.q, [l / self.temperature for l in losses], self.rho)
        g = torch.Generator().manual_seed(round_seed(self.seed, ctx["round"]))
        z = torch.randn(self.k, generator=g, dtype=torch.float64).tolist()
        return tilt_in_box(self.q, [self.sigma * zi for zi in z], self.rho)
