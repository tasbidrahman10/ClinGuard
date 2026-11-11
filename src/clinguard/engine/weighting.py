"""Site weights for the outer aggregation.

Implemented now: uniform, fixed-q (from eligible training-study counts) and external weights.
Random matched-variance weights, loss weighting and the ClinGuard controller arrive in Phase 6;
they plug in through the same `Weighting` call signature.
"""


def uniform(k):
    return [1.0 / k] * k


def fixed_q(train_counts):
    """q_i proportional to site i's eligible training-study count."""
    if any(c <= 0 for c in train_counts):
        raise ValueError(f"every site needs a positive training count, got {list(train_counts)}")
    total = float(sum(train_counts))
    return [c / total for c in train_counts]


class Weighting:
    """Callable returning this round's weights. `ctx` is whatever the coordinator knows that round."""

    def __init__(self, kind, train_counts, external=None):
        if kind not in ("uniform", "fixed_q", "external"):
            raise ValueError(f"unknown weighting '{kind}'")
        self.kind = kind
        self.q = fixed_q(train_counts)
        self.k = len(train_counts)
        if kind == "external":
            if external is None or len(external) != self.k:
                raise ValueError("external weighting needs one weight per site")
            if min(external) < 0 or abs(sum(external) - 1.0) > 1e-9:
                raise ValueError(f"external weights must lie on the simplex, got {list(external)}")
        self.external = list(external) if external is not None else None

    def __call__(self, ctx=None):
        if self.kind == "uniform":
            return uniform(self.k)
        if self.kind == "fixed_q":
            return list(self.q)
        return list(self.external)
