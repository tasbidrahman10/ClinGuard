"""CheXbert labels of the reference reports of one or more data roles (evaluation environment, CPU).

  .venv-eval/Scripts/python.exe scripts/label_reports.py --roles train calibration --hf-repo tasbid66/clinguard-runs

Writes runs_work/labels/chexbert_<role>.json ({uid: 14 labels, CheXbert classification mode}) and, with
--hf-repo, uploads it to labels/chexbert_<role>.json in the PRIVATE store (derived from report text, so it
never goes into the public repository). Roles are the same in both partitions (deviation P1), so the files
do not depend on the partition. Used by the finding-sentence training objective (I30) and later by the
silver probes (plan 7.2). The test role is refused.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from clinguard.data import dataset  # noqa: E402
from clinguard.eval import metrics  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--roles", nargs="+", required=True, choices=("train", "calibration", "development"))
    p.add_argument("--partition", default="skew_v1")
    p.add_argument("--hf-repo")
    p.add_argument("--out-dir", default="runs_work/labels")
    p.add_argument("--raw-dir", default="data/raw/iu_xray")
    p.add_argument("--images-dir", default="data/raw/iu_xray/images/images_normalized")
    p.add_argument("--manifest-dir", default="manifests")
    a = p.parse_args(argv)

    m = dataset.verify_source(a.raw_dir, a.manifest_dir, a.partition)
    chexbert_label, _ = metrics.load_scorers()
    os.makedirs(a.out_dir, exist_ok=True)
    store = None
    if a.hf_repo:
        from clinguard.utils.hf_sync import HFStore
        store = HFStore(a.hf_repo)
    for role in a.roles:
        examples = [e for s in m["sites"] for e in dataset.load_examples(a.raw_dir, a.images_dir, a.manifest_dir,
                                                                          a.partition, s, role)]
        labels = {e.uid: chexbert_label(e.findings) for e in examples}
        path = os.path.join(a.out_dir, f"chexbert_{role}.json")
        with open(path, "w") as f:
            json.dump(labels, f, sort_keys=True)
        print(f"{role}: {len(labels)} reports labelled -> {path}", flush=True)
        if store:
            store.upload_files({f"labels/chexbert_{role}.json": path}, f"CheXbert labels of the {role} reports")
            print(f"   uploaded labels/chexbert_{role}.json", flush=True)


if __name__ == "__main__":
    main()
