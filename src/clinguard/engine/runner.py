"""One simulated run: K sites trained one after another, one coordinator, T rounds.

    for each round t:
        for each site i:  load phi_t -> [calibration loss at phi_t, loss weighting only]
                          -> train H optimizer steps -> d_i = phi_t - phi_i
        w      = weighting(...)                       (uniform, fixed q, external, loss, random)
        d_bar  = sum_i w_i d_i
        m, phi = Nesterov outer step                  (engine/outer.py)
        log the round, write a checkpoint

Baselines are this same loop with a different `weighting` or outer setting, so comparisons are
controlled by construction. Rerunning the same call after an interruption continues from the last
complete round.
"""
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass

import torch

from clinguard.engine import outer, state
from clinguard.engine.accounting import Accountant
from clinguard.engine.weighting import Weighting
from clinguard.utils.hashing import hash_config


@dataclass
class RunConfig:
    H: int = 100                      # inner optimizer steps per round
    total_steps_per_site: int = 1000  # fixed data budget; rounds = total_steps_per_site / H
    eta_out: float = 0.7
    mu: float = 0.9
    weighting: str = "fixed_q"        # uniform | fixed_q | external | loss | random
    external_weights: tuple = ()
    rho: float = 0.5                  # box (1 - rho) q <= w <= (1 + rho) q for loss / random (and ClinGuard)
    loss_temperature: float = 1.0     # loss weighting: scores = calibration loss / T
    random_sigma: float = 0.1         # random weighting: scores = sigma * N(0, 1); matched to ClinGuard in Phase 6
    reset_inner_state: bool = False   # True only for the FedAvg-with-reset variant
    seed: int = 11
    method: str = "fixed_q_diloco"    # label used in the run id and logs

    @property
    def rounds(self):
        if self.total_steps_per_site % self.H:
            raise ValueError(f"total_steps_per_site={self.total_steps_per_site} is not a multiple of H={self.H}")
        return self.total_steps_per_site // self.H


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return None


def run(cfg, sites, phi0, out_dir, manifest_hash=None, stop_after_round=None, extra_config=None, on_round_end=None):
    """Runs (or resumes) the experiment. Returns (phi, m) after the last completed round.

    stop_after_round: stop once this round index is complete, leaving the checkpoint in place.
        Used by the resume tests to imitate an interrupted session.
    extra_config: settings outside the engine (model, data, LoRA file, evaluation). They enter the
        config hash, so a run refuses to resume when any of them changed.
    on_round_end: called as on_round_end(t, out_dir) after round t's checkpoint is written
        (the job runner uploads the run folder and renews its lease here).
    """
    os.makedirs(out_dir, exist_ok=True)
    cfg_dict = asdict(cfg) if extra_config is None else dict(engine=asdict(cfg), extra=extra_config)
    config_hash = hash_config(cfg_dict)
    rounds_path = os.path.join(out_dir, "rounds.jsonl")
    weighting = Weighting(cfg.weighting, [s.n_train for s in sites],
                          external=list(cfg.external_weights) or None, rho=cfg.rho,
                          temperature=cfg.loss_temperature, sigma=cfg.random_sigma, seed=cfg.seed)
    acct = Accountant([s.name for s in sites], phi0.numel())

    resumed = state.load_checkpoint(out_dir, sites=sites, accountant=acct,
                                    config_hash=config_hash, manifest_hash=manifest_hash)
    if resumed is None:
        start, phi, m = 0, phi0.detach().clone(), torch.zeros_like(phi0)     # m_{-1} = 0
        open(rounds_path, "w").close()
        with open(os.path.join(out_dir, "run.json"), "w") as f:
            json.dump(dict(config=cfg_dict, config_hash=config_hash, manifest_hash=manifest_hash,
                           code_commit=_git_commit(), sites=[s.name for s in sites],
                           train_counts=[s.n_train for s in sites], n_params=phi0.numel(),
                           q=weighting.q, python=sys.version.split()[0], torch=torch.__version__,
                           platform=platform.platform()), f, indent=1)
    else:
        start, phi, m = resumed
        # keep only the logged rounds that the checkpoint covers
        kept = [l for l in open(rounds_path) if l.strip() and json.loads(l)["round"] < start]
        with open(rounds_path, "w") as f:
            f.writelines(kept)

    for t in range(start, cfg.rounds):
        acct.start_round()
        deltas, losses, calib = [], {}, []
        for s in sites:
            s.set_params(phi)
            acct.broadcast(s.name)
            if weighting.needs_calibration_loss:              # measured at phi_t, before local training
                with acct.timed(s.name, "calib_sec"):
                    calib.append(float(s.calibration_loss()))
            if cfg.reset_inner_state:
                s.reset_inner_state()
            with acct.timed(s.name):
                stats = s.train(cfg.H)
            acct.record_train(s.name, stats)
            deltas.append(outer.pseudo_gradient(phi, s.get_params()))
            acct.upload_update(s.name)                        # the calibration loss rides in the summary bytes
            losses[s.name] = stats["loss"]

        w = weighting(dict(round=t, calib_loss=calib or None))
        d_bar = outer.aggregate(deltas, w)
        phi, m = outer.nesterov_step(phi, m, d_bar, cfg.eta_out, cfg.mu)

        rec = dict(round=t, method=cfg.method, weights=dict(zip([s.name for s in sites], w)),
                   train_loss=losses,
                   calib_loss=dict(zip([s.name for s in sites], calib)) if calib else None,
                   delta_norm={s.name: d.norm().item() for s, d in zip(sites, deltas)},
                   d_bar_norm=d_bar.norm().item(), m_norm=m.norm().item(), phi_norm=phi.norm().item(),
                   cost=acct.end_round())
        with open(rounds_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        state.save_checkpoint(out_dir, next_round=t + 1, phi=phi, m=m, sites=sites, accountant=acct,
                              config_hash=config_hash, manifest_hash=manifest_hash)
        if on_round_end is not None:
            on_round_end(t, out_dir)
        if stop_after_round is not None and t >= stop_after_round:
            break
    return phi, m
