"""Per-site batch sampler: epoch-wise shuffled permutation, no replacement within an epoch.

Each site owns one sampler with its own generator. Its full state is a plain dict, so it goes
into the round checkpoint and a resumed run draws exactly the batches an uninterrupted run would.
"""
import numpy as np


class EpochSampler:
    def __init__(self, n_items, batch_size, seed):
        if n_items <= 0 or batch_size <= 0:
            raise ValueError("n_items and batch_size must be positive")
        self.n, self.bs = int(n_items), int(batch_size)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.perm = self.rng.permutation(self.n)
        self.pos = 0
        self.epoch = 0
        self.seen = set()      # distinct items drawn since the last reset_seen()
        self.drawn = 0         # items drawn since the last reset_seen()

    def next(self):
        """Next batch of item positions (0 .. n_items-1). A batch may span an epoch boundary."""
        out = []
        while len(out) < self.bs:
            if self.pos == self.n:
                self.perm = self.rng.permutation(self.n)
                self.pos = 0
                self.epoch += 1
            take = min(self.bs - len(out), self.n - self.pos)
            out.extend(self.perm[self.pos:self.pos + take].tolist())
            self.pos += take
        self.seen.update(out)
        self.drawn += len(out)
        return out

    def reset_seen(self):
        self.seen, self.drawn = set(), 0

    def state(self):
        return dict(n=self.n, bs=self.bs, rng=self.rng.bit_generator.state, perm=self.perm.tolist(),
                    pos=self.pos, epoch=self.epoch)

    def load(self, s):
        if s["n"] != self.n or s["bs"] != self.bs:
            raise ValueError("sampler state belongs to a different site or batch size")
        self.rng.bit_generator.state = s["rng"]
        self.perm = np.asarray(s["perm"], dtype=np.int64)
        self.pos, self.epoch = s["pos"], s["epoch"]
        self.reset_seen()
