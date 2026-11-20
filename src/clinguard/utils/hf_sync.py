"""The shared run store (plan section 4.1): one private Hugging Face repo used by three Kaggle accounts.

Layout:  runs/<run_id>/{checkpoint.pt, run.json, rounds.jsonl, phi_final.safetensors, LEASE.json,
                        generations/dev_<site>.jsonl}
         shared/lora_A_r16_seed2026.safetensors

Two backends with one interface: HFStore (the real repo) and LocalStore (a folder standing in for it, used
by the tests). Lease writes are compare-and-swap: HFStore commits with `parent_commit`, so a write fails if
anyone else committed in between, and the lease is then re-read and re-checked.
"""
import json
import os
import shutil
import socket
import time
import uuid
from datetime import datetime, timedelta, timezone

LEASE_HOURS = 13.0


class Conflict(Exception):
    """The store changed between reading its head and writing."""


class LeaseHeld(RuntimeError):
    pass


class RunFinished(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------------------------- stores

class LocalStore:
    """A folder that behaves like the HF repo, including compare-and-swap on a head counter."""

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _p(self, path):
        return os.path.join(self.root, *path.split("/"))

    def head(self):
        p = self._p(".head")
        return open(p).read() if os.path.exists(p) else "0"

    def _bump(self, parent):
        current = self.head()
        if parent is not None and parent != current:
            raise Conflict(f"head moved from {parent}")
        with open(self._p(".head"), "w") as f:
            f.write(str(int(current) + 1))

    def exists(self, path):
        return os.path.exists(self._p(path))

    def list(self, prefix):
        base = self._p(prefix)
        out = []
        for d, _, files in os.walk(base):
            out += [os.path.relpath(os.path.join(d, f), self.root).replace(os.sep, "/") for f in files]
        return sorted(out)

    def read_bytes(self, path):
        with open(self._p(path), "rb") as f:
            return f.read()

    def commit(self, adds, message, parent=None):
        """adds: {path_in_store: bytes or local file path}, written as one commit."""
        self._bump(parent)
        for dst, src in adds.items():
            os.makedirs(os.path.dirname(self._p(dst)), exist_ok=True)
            if isinstance(src, bytes):
                with open(self._p(dst), "wb") as f:
                    f.write(src)
            else:
                shutil.copyfile(src, self._p(dst))

    def write_bytes(self, path, data, message, parent=None):
        self.commit({path: data}, message, parent)

    def upload_files(self, files, message):
        """files: {path_in_store: local_path}."""
        self.commit(files, message)

    def download(self, path, local_path):
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        shutil.copyfile(self._p(path), local_path)
        return local_path


class HFStore:
    """The private Hugging Face model repo. The token comes from the HF_TOKEN environment variable."""

    def __init__(self, repo_id):
        from huggingface_hub import HfApi
        self.repo_id = repo_id
        self.api = HfApi()

    def head(self):
        return self.api.repo_info(self.repo_id).sha

    def exists(self, path):
        return self.api.file_exists(self.repo_id, path)

    def list(self, prefix):
        return sorted(f for f in self.api.list_repo_files(self.repo_id) if f.startswith(prefix))

    def read_bytes(self, path):
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(self.repo_id, path, force_download=True)
        with open(local, "rb") as f:
            return f.read()

    def commit(self, adds, message, parent=None):
        """adds: {path_in_repo: bytes or local file path}, one commit. With `parent`, the commit fails
        (Conflict) if the repo head moved since it was read."""
        from huggingface_hub import CommitOperationAdd
        from huggingface_hub.errors import HfHubHTTPError
        ops = [CommitOperationAdd(dst, src) for dst, src in adds.items()]
        try:
            self.api.create_commit(self.repo_id, ops, commit_message=message, parent_commit=parent)
        except HfHubHTTPError as e:
            if parent is not None and getattr(e.response, "status_code", None) in (409, 412):
                raise Conflict(str(e)) from e
            raise

    def write_bytes(self, path, data, message, parent=None):
        self.commit({path: data}, message, parent)

    def upload_files(self, files, message):
        self.commit(files, message)

    def download(self, path, local_path):
        from huggingface_hub import hf_hub_download
        tmp = hf_hub_download(self.repo_id, path, force_download=True)
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        shutil.copyfile(tmp, local_path)
        return local_path


def read_json(store, path):
    return json.loads(store.read_bytes(path)) if store.exists(path) else None


# ---------------------------------------------------------------------------------------------- lease

def holder_identity(account):
    """Who is training: the account, plus a per-process id so two sessions of one account differ."""
    return dict(account=account, session=f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}")


class Lease:
    """LEASE.json in a run folder: who trains the run now, until when, and whether the run is finished."""

    def __init__(self, store, run_id, holder, commit, hours=LEASE_HOURS):
        self.store, self.run_id, self.holder, self.commit, self.hours = store, run_id, holder, commit, hours
        self.path = f"runs/{run_id}/LEASE.json"

    def _write(self, status, check, message, attempts=6, expire_now=False, extra_files=None):
        for _ in range(attempts):
            head = self.store.head()
            current = read_json(self.store, self.path)
            check(current)
            now = _now()
            expires = now if expire_now else now + timedelta(hours=self.hours)
            record = dict(run_id=self.run_id, status=status, holder=self.holder, code_commit=self.commit,
                          acquired=(current or {}).get("acquired") if self._mine(current) else now.isoformat(),
                          updated=now.isoformat(), expires=expires.isoformat(),
                          previous_holder=None if self._mine(current) else (current or {}).get("holder"))
            try:
                adds = {**(extra_files or {}), self.path: json.dumps(record, indent=1).encode()}
                self.store.commit(adds, message, parent=head)
                return record
            except Conflict:
                time.sleep(1.0)
        raise RuntimeError(f"could not write {self.path} after {attempts} attempts (store busy)")

    def _mine(self, current):
        return bool(current) and current.get("holder") == self.holder

    @staticmethod
    def expired(current):
        return datetime.fromisoformat(current["expires"]) <= _now()

    def acquire(self, takeover=False):
        """Takes the lease. Refuses while another holder's lease is unexpired (unless takeover=True, for a
        session that is known to be dead) and when the run is already finished."""
        def check(current):
            if not current:
                return
            if current["status"] == "finished":
                raise RunFinished(f"{self.run_id} is already finished")
            if not self._mine(current) and not self.expired(current) and not takeover:
                raise LeaseHeld(f"{self.run_id} is held by {current['holder']} until {current['expires']}")
        return self._write("running", check, f"lease {self.run_id}: acquired by {self.holder['account']}")

    def renew(self, files=None, message=None):
        """Renews the lease; `files` ({path: local path or bytes}) are uploaded in the same commit."""
        def check(current):
            if not self._mine(current):
                raise LeaseHeld(f"lost the lease on {self.run_id}: now held by {(current or {}).get('holder')}")
        return self._write("running", check, message or f"lease {self.run_id}: renewed", extra_files=files)

    def pause(self, files=None):
        """Hand the lease back without finishing (a deliberate stop): anyone may resume at once."""
        def check(current):
            if not self._mine(current):
                raise LeaseHeld(f"lost the lease on {self.run_id} before pausing")
        return self._write("paused", check, f"lease {self.run_id}: paused", expire_now=True, extra_files=files)

    def finish(self, files=None):
        def check(current):
            if not self._mine(current):
                raise LeaseHeld(f"lost the lease on {self.run_id} before finishing")
        return self._write("finished", check, f"lease {self.run_id}: finished", extra_files=files)
