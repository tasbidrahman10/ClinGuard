"""A simulated hospital that trains the real report-generation model (implements engine.task.SiteTask).

All sites share ONE loaded model (plan section 3): the runner loads phi_t into the LoRA B matrices, the
site restores its own AdamW and sampler state, trains H optimizer steps, and the runner reads phi back.
Each site owns its AdamW optimizer over the same B parameters, so the optimizer states stay separate.

Inner optimization (plan section 6.3): AdamW (0.9, 0.999), weight decay 0, gradient clipping 1.0,
effective batch = micro_batch x grad_accum, linear warm-up over the site's first `warmup_steps`
optimizer steps (round 0 only at H >= warmup_steps), constant afterwards. One optimizer step is one
AdamW update, not one micro-batch. On a CUDA device with fp16 autocast, a per-site GradScaler is used
and checkpointed.
"""
import contextlib

import torch

from clinguard.data.dataset import IGNORE_INDEX, batch_to, build_batch
from clinguard.data.sampler import EpochSampler
from clinguard.model import params


class VLMSite:
    def __init__(self, name, model, processor, train_examples, calib_examples=(), *, micro_batch=1, grad_accum=4,
                 lr=1e-4, warmup_steps=50, clip=1.0, seed=0, amp_dtype=None, calib_batch=4):
        if not train_examples:
            raise ValueError(f"site {name} has no training examples")
        self.name = name
        self.model, self.processor = model, processor
        self.train_examples, self.calib_examples = list(train_examples), list(calib_examples)
        self.n_train = len(self.train_examples)
        self.micro_batch, self.grad_accum, self.calib_batch = micro_batch, grad_accum, calib_batch
        self.lr, self.warmup_steps, self.clip = lr, warmup_steps, clip
        self.params = params.phi_params(model)
        self.device = self.params[0].device
        self.amp_dtype = amp_dtype
        self.sampler = EpochSampler(self.n_train, micro_batch, seed)
        self.opt_steps = 0
        use_scaler = amp_dtype == torch.float16 and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if use_scaler else None
        self.reset_inner_state()

    # --- SiteTask interface -------------------------------------------------------------------------
    def get_params(self):
        return params.get_phi(self.params)

    def set_params(self, phi):
        params.set_phi(self.params, phi)

    def train(self, steps):
        self.model.train()
        self.sampler.reset_seen()
        total = 0.0
        for _ in range(steps):
            lr = self.lr * min(1.0, (self.opt_steps + 1) / self.warmup_steps) if self.warmup_steps else self.lr
            for group in self.opt.param_groups:
                group["lr"] = lr
            self.opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(self.grad_accum):
                batch = [self.train_examples[i] for i in self.sampler.next()]
                loss = self._loss(batch) / self.grad_accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"site {self.name}: non-finite training loss at optimizer step {self.opt_steps}")
                (self.scaler.scale(loss) if self.scaler else loss).backward()
                step_loss += loss.item()
            if self.scaler:
                self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)
            if self.scaler:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()
            self.opt_steps += 1
            total += step_loss
        return dict(loss=total / steps, examples=self.sampler.drawn, unique=len(self.sampler.seen))

    def calibration_loss(self):
        """Token-weighted mean loss on the calibration split at the current parameters. Eval mode, no
        gradients, no random numbers: optimizer, sampler and random state are untouched."""
        if not self.calib_examples:
            raise RuntimeError(f"site {self.name} has no calibration examples")
        was_training = self.model.training
        self.model.eval()
        loss_sum, tokens = 0.0, 0
        try:
            with torch.no_grad():
                for i in range(0, len(self.calib_examples), self.calib_batch):
                    enc = batch_to(build_batch(self.processor, self.calib_examples[i:i + self.calib_batch]), self.device)
                    n = int((enc["labels"][:, 1:] != IGNORE_INDEX).sum())       # tokens the loss averages over
                    with self._autocast():
                        loss = self.model(**enc).loss
                    loss_sum += loss.item() * n
                    tokens += n
        finally:
            self.model.train(was_training)
        return loss_sum / tokens

    def reset_inner_state(self):
        self.opt = torch.optim.AdamW(self.params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)

    def state_dict(self):
        return dict(opt=self.opt.state_dict(), sampler=self.sampler.state(), opt_steps=self.opt_steps,
                    scaler=self.scaler.state_dict() if self.scaler else None)

    def load_state_dict(self, state):
        self.opt.load_state_dict(state["opt"])
        self.sampler.load(state["sampler"])
        self.opt_steps = state["opt_steps"]
        if self.scaler and state["scaler"] is not None:
            self.scaler.load_state_dict(state["scaler"])

    # --- internals ---------------------------------------------------------------------------------
    def _autocast(self):
        if self.amp_dtype is None or self.device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _loss(self, batch):
        enc = batch_to(build_batch(self.processor, batch), self.device)
        with self._autocast():
            return self.model(**enc).loss

