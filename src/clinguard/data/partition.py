"""Site partitions and the four data roles (plan sections 5.4 and 5.5).

Every random choice is a hash of (seed, purpose, uid), not a draw from a running generator, so a
study's site never depends on which other studies are eligible.

skew_v1 (primary)  site sampled from a per-group probability table; then, inside each
                   site x group, studies are ordered by hash and cut 70 / 10 / 10 / 10.
iid_v1 (control)   every study keeps its skew_v1 role; inside each role x group, studies are
                   dealt evenly to the sites in hash order. A study is therefore a test study in
                   both partitions or in neither.
"""
import hashlib
from collections import defaultdict

ROLES = ["train", "calibration", "development", "test"]


def unit_hash(seed, purpose, uid):
    """Deterministic number in [0, 1) for one study."""
    h = hashlib.sha256(f"{seed}|{purpose}|{uid}".encode()).hexdigest()
    return int(h[:16], 16) / 16 ** 16


def _check_table(cfg, groups):
    for g in groups:
        p = cfg["site_probabilities"].get(g)
        if p is None or len(p) != len(cfg["sites"]) or abs(sum(p) - 1.0) > 1e-9 or min(p) < 0:
            raise ValueError(f"site_probabilities[{g}] must give one probability per site, summing to 1")
    if abs(sum(cfg["roles"][r] for r in ROLES) - 1.0) > 1e-9:
        raise ValueError("role fractions must sum to 1")


def skew_partition(studies, cfg):
    """studies: iterable of (uid, group). Returns {uid: (site, role)}."""
    studies = list(studies)
    _check_table(cfg, {g for _, g in studies})
    seed, sites = cfg["split_seed"], cfg["sites"]
    strata = defaultdict(list)
    for uid, group in studies:
        u, acc, site = unit_hash(seed, "site", uid), 0.0, sites[-1]
        for s, p in zip(sites, cfg["site_probabilities"][group]):
            acc += p
            if u < acc:
                site = s
                break
        strata[(site, group)].append(uid)
    fractions = [cfg["roles"][r] for r in ROLES]
    out = {}
    for (site, _), uids in strata.items():
        uids.sort(key=lambda x: unit_hash(seed, "role", x))
        n, cum, start = len(uids), 0.0, 0
        for role, f in zip(ROLES, fractions):
            cum += f
            end = n if role == ROLES[-1] else int(cum * n + 1e-9)
            for uid in uids[start:end]:
                out[uid] = (site, role)
            start = end
    return out


def iid_partition(studies, skew_assignment, cfg):
    """Same roles as skew_assignment; sites dealt evenly within each role x group."""
    seed, sites = cfg["split_seed"], cfg["sites"]
    strata = defaultdict(list)
    for uid, group in studies:
        strata[(skew_assignment[uid][1], group)].append(uid)
    out = {}
    for k, (key, uids) in enumerate(sorted(strata.items())):
        uids.sort(key=lambda x: unit_hash(seed, "iid", x))
        for j, uid in enumerate(uids):
            out[uid] = (sites[(j + k) % len(sites)], key[0])     # rotate who gets the remainder
    return out
