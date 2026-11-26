"""Score a run's generated development reports (plan sections 10.2-10.3). Run in the evaluation environment.

  .venv-eval/Scripts/python.exe scripts/evaluate.py --hf-repo tasbid66/clinguard-runs --run-id <run_id> [--upload]
  .venv-eval/Scripts/python.exe scripts/evaluate.py --generations runs_eval/<run_id>/generations --out results.json

Downloads runs/<run_id>/generations/*.jsonl when --hf-repo is given, scores every report (CheXbert labels,
proxy_v1 error, RadGraph-XL, BLEU-4, ROUGE-L) and writes into runs_eval/<run_id>/:
  dev_metrics.json      per site, pooled and worst site, for the subsets all / no_prior / novel_reference
  dev_per_report.jsonl  per-report labels and scores (no report text; keep private all the same)
--subdir picks another generation folder of the run, e.g. generations_oracle (diagnosis prompt with the true
findings, I32); its files are then named dev_metrics_<subdir>.json and dev_per_report_<subdir>.jsonl.
--upload puts dev_metrics.json into the run folder of the HF repo. Report text never leaves the private
store and this machine.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from clinguard.data import dataset  # noqa: E402
from clinguard.eval import metrics  # noqa: E402


def training_texts(raw_dir, images_dir, manifest_dir, partition):
    """Cleaned reference texts of every training study of the partition (for the novel-reference subset)."""
    m = dataset.verify_source(raw_dir, manifest_dir, partition)
    return {e.findings for site in m["sites"]
            for e in dataset.load_examples(raw_dir, images_dir, manifest_dir, partition, site, "train")}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--hf-repo")
    p.add_argument("--run-id")
    p.add_argument("--generations", help="local folder with dev_<site>.jsonl files")
    p.add_argument("--out-dir", default="runs_eval")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--subdir", default="generations", help="generation folder inside the run (with --hf-repo)")
    p.add_argument("--raw-dir", default="data/raw/iu_xray")
    p.add_argument("--images-dir", default="data/raw/iu_xray/images/images_normalized")
    p.add_argument("--manifest-dir", default="manifests")
    a = p.parse_args(argv)

    store = None
    if a.hf_repo:
        if not a.run_id:
            p.error("--hf-repo needs --run-id")
        from clinguard.utils.hf_sync import HFStore
        store = HFStore(a.hf_repo)
        gen_dir = os.path.join(a.out_dir, a.run_id, a.subdir)
        for path in store.list(f"runs/{a.run_id}/{a.subdir}/"):
            store.download(path, os.path.join(gen_dir, os.path.basename(path)))
    elif a.generations:
        gen_dir = a.generations
    else:
        p.error("give --hf-repo with --run-id, or --generations")

    rows = []
    for path in sorted(glob.glob(os.path.join(gen_dir, "dev_*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    if not rows:
        sys.exit(f"no generations found in {gen_dir}")
    run_id = a.run_id or rows[0]["run_id"]
    partition = rows[0]["partition"]

    chexbert_label, radgraph_rewards = metrics.load_scorers()
    scored = metrics.score_reports(rows, chexbert_label, radgraph_rewards)
    summary = dict(run_id=run_id, partition=partition, n_reports=len(scored), generations=a.subdir,
                   **metrics.summarize(scored, training_texts(a.raw_dir, a.images_dir, a.manifest_dir, partition)))

    out = os.path.join(a.out_dir, run_id)
    os.makedirs(out, exist_ok=True)
    tag = "" if a.subdir == "generations" else f"_{a.subdir}"
    with open(os.path.join(out, f"dev_metrics{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(out, f"dev_per_report{tag}.jsonl"), "w") as f:
        for s in scored:
            f.write(json.dumps({k: s[k] for k in ("site", "uid", "group", "needs_prior", "ref_labels", "hyp_labels",
                                                  "proxy_error", "proxy_details", "radgraph", "rouge_l")}) + "\n")
    if a.upload and store:
        store.upload_files({f"runs/{run_id}/eval/dev_metrics{tag}.json": os.path.join(out, f"dev_metrics{tag}.json")},
                           f"{run_id}: development metrics ({metrics.PROXY_VERSION})")

    print_summary(summary)
    return summary


def print_summary(summary):
    s = summary["subsets"]["all"]
    print(f"{summary['run_id']} ({summary.get('generations', 'generations')}): {summary['n_reports']} reports "
          f"(proxy error: automatic proxy, not reader-assessed)")
    for site, m in list(s["sites"].items()) + [("pooled", s["pooled"])]:
        recall = "-" if m["finding_recall"] is None else f"{m['finding_recall']:.2f}"
        print(f"  {site:6s} n={m['n']:3d} proxy error {m['proxy_error_rate']:.3f} (always-normal "
              f"{m['always_normal_error_rate']:.3f}, constant report {m['constant_report_error_rate']:.3f}) | CheXbert F1-14 micro {m['chexbert_f1_14']['micro']:.3f} | "
              f"RadGraph partial {m['radgraph']['partial']:.3f} | BLEU-4 {m['bleu4']:.3f} | ROUGE-L {m['rouge_l']:.3f}")
        print(f"  {'':6s} asserts a finding {m['share_asserting_a_finding']:.0%} | finding recall {recall} | "
              f"distinct texts {m['distinct_text_share']:.0%} | most common text {m['top_text_share']:.0%} | "
              f"words {m['mean_words_generated']:.0f}")
    w, v = s["worst_site"], s["vs_always_normal"]
    print(f"  worst site: {w['site']} proxy error {w['proxy_error_rate']:.3f} (always-normal worst {w['always_normal_worst']:.3f})")
    print(f"  vs always-normal, pooled, paired: model-only errors {v['model_only_errors']}, always-normal-only errors "
          f"{v['baseline_only_errors']}, one-sided exact McNemar p = {v['p_one_sided']:.3g}")
    k = s["vs_constant_report"]
    print(f"  vs constant report ({', '.join(summary['constant_report'])}), pooled, paired: model-only errors "
          f"{k['model_only_errors']}, constant-only errors {k['baseline_only_errors']}, p = {k['p_one_sided']:.3g}")


if __name__ == "__main__":
    main()
