"""The interface between the engine and a site (hospital).

The engine and controller only ever see flat parameter vectors and this interface, so they can
be tested on toy problems on CPU. The real VLM site (Phase 3) implements the same methods.
"""
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class SiteTask(Protocol):
    name: str
    n_train: int                       # eligible training studies at this site (defines q)

    def get_params(self) -> torch.Tensor:
        """A copy of the site's current trainable vector phi (flat, CPU)."""

    def set_params(self, phi: torch.Tensor) -> None:
        """Load the broadcast phi_t. Must not touch the inner optimizer state."""

    def train(self, steps: int) -> dict:
        """Run `steps` optimizer steps on local training data. Returns at least
        {"loss": mean training loss, "examples": items drawn, "unique": distinct items drawn}."""

    def reset_inner_state(self) -> None:
        """Forget the inner optimizer state (FedAvg-with-reset variant only)."""

    def state_dict(self) -> dict:
        """Everything needed to resume exactly: inner optimizer state and sampler state."""

    def load_state_dict(self, state: dict) -> None:
        ...
