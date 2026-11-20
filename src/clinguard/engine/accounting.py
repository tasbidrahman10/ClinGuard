"""Cost accounting (RQ4). Bytes are LOGICAL: what the protocol would send between a site and the
coordinator in this single-machine simulation, not measured network traffic.
"""
import time
from contextlib import contextmanager

FP32_BYTES = 4
SUMMARY_BYTES = 16          # one loss (8 bytes) and one count (8 bytes) per site per round


class Accountant:
    def __init__(self, site_names, n_params):
        self.sites = list(site_names)
        self.p = int(n_params)
        self.total = {s: dict(upload_bytes=0, download_bytes=0, train_sec=0.0, calib_sec=0.0, examples=0)
                      for s in self.sites}
        self._round = None

    def start_round(self):
        self._round = {s: dict(upload_bytes=0, download_bytes=0, train_sec=0.0, calib_sec=0.0, examples=0,
                               unique=0) for s in self.sites}

    def broadcast(self, site):
        """Coordinator -> site: the shared trainable vector in fp32."""
        self._add(site, "download_bytes", self.p * FP32_BYTES)

    def upload_update(self, site, extra_floats=0):
        """Site -> coordinator: its pseudo-gradient in fp32, a small summary, and any sketches."""
        self._add(site, "upload_bytes", (self.p + extra_floats) * FP32_BYTES + SUMMARY_BYTES)

    @contextmanager
    def timed(self, site, key="train_sec"):
        t0 = time.perf_counter()
        yield
        self._add(site, key, time.perf_counter() - t0)

    def record_train(self, site, stats):
        self._add(site, "examples", stats["examples"])
        self._round[site]["unique"] = stats["unique"]

    def _add(self, site, key, value):
        self._round[site][key] += value
        self.total[site][key] += value

    def end_round(self):
        r, self._round = self._round, None
        return r

    def state(self):
        return dict(total=self.total)

    def load(self, s):
        self.total = s["total"]
