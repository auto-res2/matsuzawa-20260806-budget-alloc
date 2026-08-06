"""Aggregate the three arms into the paired comparison the paper reports.

The arms run as separate jobs, so this is where they are finally put side by
side. Two things happen here that cannot happen inside a single run:

  * the comparison is PAIRED -- every difference is taken within a system, so
    the enormous variation in system difficulty cancels instead of drowning
    the effect;
  * the arms are checked for having actually received identical MSAs, rather
    than that being assumed. If they did not, the comparison is confounded and
    the reader needs to know.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)

PRIMARY_METRIC = "oracle_lddt_pli"
SUPPORTING_METRICS = ["success_rate", "top_ranked_lddt_pli", "ligand_rmsd",
                      "unique_pose_modes"]

# Frozen before any data was collected. A difference below this is not the
# effect this experiment was designed to detect, whatever its p-value.
GO_NO_GO_MARGIN = 0.03
N_BOOTSTRAP = 1000

ARM_COLOURS = {"baseline": "#4C72B0", "proposed": "#C44E52", "lineage": "#55A868"}
BIN_ORDER = ["20-30", "30-40", "40-50", "50-60"]


def _load(results_dir: str, run_id: str) -> dict | None:
    path = os.path.join(results_dir, run_id, "metrics.json")
    if not os.path.exists(path):
        log.warning("no metrics.json for %s", run_id)
        return None
    return json.load(open(path))


def _load_records(results_dir: str, run_id: str) -> dict | None:
    path = os.path.join(results_dir, run_id, "records.json")
    return json.load(open(path)) if os.path.exists(path) else None


def _paired(a: dict, b: dict, key: str) -> tuple[list[str], list[float]]:
    """Systems scored in BOTH arms, and their within-system differences."""
    ma = {s["system_id"]: s for s in a["per_system"] if s.get("scored")}
    mb = {s["system_id"]: s for s in b["per_system"] if s.get("scored")}
    ids, diffs = [], []
    for sid in sorted(set(ma) & set(mb)):
        va, vb = ma[sid].get(key), mb[sid].get(key)
        if va is None or vb is None:
            continue
        if not (math.isfinite(va) and math.isfinite(vb)):
            continue
        ids.append(sid)
        diffs.append(float(va) - float(vb))
    return ids, diffs


def _bootstrap_ci(values: list[float], n: int = N_BOOTSTRAP,
                  seed: int = 0) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    k = len(values)
    for _ in range(n):
        means.append(sum(values[rng.randrange(k)] for _ in range(k)) / k)
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[min(n - 1, int(0.975 * n))]
    return sum(values) / k, lo, hi


def _by_bin(metrics: dict, key: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for s in metrics["per_system"]:
        if not s.get("scored"):
            continue
        v = s.get(key)
        if v is None or not math.isfinite(v):
            continue
        out.setdefault(s["bin"], []).append(float(v))
    return out


def check_msa_consistency(results_dir: str, run_ids: list[str]) -> dict:
    """Verify every arm saw the same MSA for the same system.

    MSA-search returned byte-identical alignments for repeated queries when
    tested, but 'verified once' is not 'true for every system in this run', and
    a silent drift here would invalidate the central comparison.
    """
    digests: dict[str, dict[str, dict]] = {}
    for rid in run_ids:
        recs = _load_records(results_dir, rid)
        if not recs:
            continue
        for r in recs["records"]:
            digests.setdefault(r["system_id"], {})[rid] = r.get("msa_digests", {})

    checked = mismatched = 0
    offenders = []
    for sid, per_run in digests.items():
        if len(per_run) < 2:
            continue
        checked += 1
        values = [json.dumps(d, sort_keys=True) for d in per_run.values()]
        if len(set(values)) > 1:
            mismatched += 1
            offenders.append(sid)
    return {
        "systems_checked": checked,
        "systems_mismatched": mismatched,
        "identical_fraction": (checked - mismatched) / checked if checked else None,
        "mismatched_systems": offenders[:20],
    }


def _fig_bins(by_run: dict[str, dict], out: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    width = 0.26
    for k, (rid, m) in enumerate(sorted(by_run.items())):
        binned = _by_bin(m, PRIMARY_METRIC)
        xs, ys, es = [], [], []
        for i, b in enumerate(BIN_ORDER):
            vals = binned.get(b, [])
            if not vals:
                continue
            mean, lo, hi = _bootstrap_ci(vals, seed=k)
            xs.append(i + (k - 1) * width)
            ys.append(mean)
            es.append([mean - lo, hi - mean])
        if not xs:
            continue
        err = list(zip(*es)) if es else None
        ax.bar(xs, ys, width=width, label=f"{m['arm']} (n={m['n_systems_scored']})",
               color=ARM_COLOURS.get(m["arm"], "#888"),
               yerr=err, capsize=2, edgecolor="white", linewidth=0.6)
    ax.set_xticks(range(len(BIN_ORDER)))
    ax.set_xticklabels(BIN_ORDER)
    ax.set_xlabel("Training-set similarity bin (SuCOS-pocket, 2023 cutoff)")
    ax.set_ylabel("oracle@25 lDDT-PLI")
    ax.set_title("Oracle lDDT-PLI by allocation and similarity")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _fig_paired(diffs: dict[str, list[float]], out: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, vals in diffs.items():
        if not vals:
            continue
        ax.hist(vals, bins=25, alpha=0.6, label=f"{name} (n={len(vals)})")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(GO_NO_GO_MARGIN, color="crimson", lw=1, ls="--",
               label=f"pre-registered margin +{GO_NO_GO_MARGIN}")
    ax.set_xlabel("Within-system difference in oracle@25 lDDT-PLI (arm - baseline)")
    ax.set_ylabel("systems")
    ax.set_title("Paired differences against the baseline allocation")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _fig_oracle_vs_top(by_run: dict[str, dict], out: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    labels, oracle, top = [], [], []
    for rid, m in sorted(by_run.items()):
        labels.append(m["arm"])
        oracle.append(m.get(PRIMARY_METRIC) or 0.0)
        top.append(m.get("top_ranked_lddt_pli") or 0.0)
    x = range(len(labels))
    ax.bar([i - 0.18 for i in x], oracle, width=0.36, label="oracle@25",
           color="#4C72B0", edgecolor="white")
    ax.bar([i + 0.18 for i in x], top, width=0.36, label="top-ranked",
           color="#DD8452", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("lDDT-PLI")
    ax.set_title("What ranking leaves on the table, per allocation")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _fig_modes(by_run: dict[str, dict], out: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    data, labels = [], []
    for rid, m in sorted(by_run.items()):
        vals = [s["unique_pose_modes"] for s in m["per_system"]
                if s.get("scored") and s.get("unique_pose_modes") is not None]
        if vals:
            data.append(vals)
            labels.append(m["arm"])
    if data:
        parts = ax.violinplot(data, showmeans=True)
        for pc, lab in zip(parts["bodies"], labels):
            pc.set_facecolor(ARM_COLOURS.get(lab, "#888"))
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
    ax.set_ylabel("distinct pose modes among 25 (2 A, complete linkage)")
    ax.set_title("Did the allocation actually change what was sampled?")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def evaluate(results_dir: str, run_ids: list[str]) -> dict:
    by_run = {}
    for rid in run_ids:
        m = _load(results_dir, rid)
        if m:
            by_run[rid] = m
    if not by_run:
        raise SystemExit(f"no metrics found under {results_dir} for {run_ids}")

    baseline_id = next((r for r, m in by_run.items() if m["arm"] == "baseline"), None)
    comparisons = {}
    if baseline_id:
        for rid, m in by_run.items():
            if rid == baseline_id:
                continue
            ids, diffs = _paired(m, by_run[baseline_id], PRIMARY_METRIC)
            mean, lo, hi = _bootstrap_ci(diffs)
            comparisons[m["arm"]] = {
                "vs": by_run[baseline_id]["arm"],
                "n_paired": len(ids),
                "mean_paired_difference": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "crosses_zero": bool(lo <= 0.0 <= hi),
                "exceeds_margin": bool(lo > GO_NO_GO_MARGIN),
                "meets_go_no_go": bool(lo > 0.0 and mean >= GO_NO_GO_MARGIN),
                "n_improved": sum(1 for d in diffs if d > 0),
                "n_worsened": sum(1 for d in diffs if d < 0),
            }
            for key in SUPPORTING_METRICS:
                if key == "success_rate":
                    continue
                _, sd = _paired(m, by_run[baseline_id], key)
                sm, slo, shi = _bootstrap_ci(sd, seed=7)
                comparisons[m["arm"]][f"paired_{key}"] = {
                    "mean": sm, "ci95_low": slo, "ci95_high": shi, "n": len(sd)}

    proposed = [m for m in by_run.values() if m["arm"] == "proposed"]
    baselines = [m for m in by_run.values() if m["arm"] in ("baseline", "lineage")]
    best_proposed = max((m[PRIMARY_METRIC] for m in proposed
                         if m.get(PRIMARY_METRIC) is not None), default=None)
    best_baseline = max((m[PRIMARY_METRIC] for m in baselines
                         if m.get(PRIMARY_METRIC) is not None), default=None)

    out_dir = os.path.join(results_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    figures = []
    try:
        _fig_bins(by_run, os.path.join(out_dir, "oracle_by_bin.pdf"))
        figures.append("comparison/oracle_by_bin.pdf")
        if baseline_id:
            diffs = {}
            for rid, m in by_run.items():
                if rid == baseline_id:
                    continue
                _, d = _paired(m, by_run[baseline_id], PRIMARY_METRIC)
                diffs[m["arm"]] = d
            _fig_paired(diffs, os.path.join(out_dir, "paired_differences.pdf"))
            figures.append("comparison/paired_differences.pdf")
        _fig_oracle_vs_top(by_run, os.path.join(out_dir, "oracle_vs_topranked.pdf"))
        figures.append("comparison/oracle_vs_topranked.pdf")
        _fig_modes(by_run, os.path.join(out_dir, "pose_modes.pdf"))
        figures.append("comparison/pose_modes.pdf")
    except Exception as exc:  # noqa: BLE001 - a broken figure must not lose the numbers
        log.error("figure generation failed: %s", exc)

    aggregated = {
        "primary_metric": PRIMARY_METRIC,
        "supporting_metrics": SUPPORTING_METRICS,
        "go_no_go_margin": GO_NO_GO_MARGIN,
        "metrics_by_run_id": {
            rid: {k: m.get(k) for k in
                  [PRIMARY_METRIC, *SUPPORTING_METRICS, "arm", "mode",
                   "n_systems", "n_systems_scored", "scoring_coverage",
                   "system_coverage", "mean_gen_seconds", "n_failures"]}
            for rid, m in by_run.items()
        },
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": (best_proposed - best_baseline
                if best_proposed is not None and best_baseline is not None else None),
        "paired_comparisons": comparisons,
        "msa_consistency": check_msa_consistency(results_dir, list(by_run)),
        "unscored_systems_by_run": {rid: m.get("unscored_systems", [])
                                    for rid, m in by_run.items()},
        "figures": figures,
    }
    with open(os.path.join(out_dir, "aggregated_metrics.json"), "w") as fh:
        json.dump(aggregated, fh, indent=2)
    return aggregated


def _parse_args(argv: list[str]) -> tuple[str, list[str]]:
    """Accept the Hydra-style key=value form the contract specifies."""
    results_dir, run_ids = ".research/results", []
    for arg in argv:
        if arg.startswith("results_dir="):
            results_dir = arg.split("=", 1)[1]
        elif arg.startswith("run_ids="):
            raw = arg.split("=", 1)[1]
            try:
                run_ids = json.loads(raw)
            except ValueError:
                run_ids = [r.strip() for r in raw.strip("[]").split(",") if r.strip()]
    return results_dir, run_ids


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results_dir, run_ids = _parse_args(sys.argv[1:])
    if not run_ids:
        run_ids = ["proposed-boltz2-rnp", "comparative-1-boltz2-rnp",
                   "comparative-2-boltz2-rnp"]
    agg = evaluate(results_dir, run_ids)
    print(json.dumps({k: v for k, v in agg.items() if k != "metrics_by_run_id"},
                     indent=2))


if __name__ == "__main__":
    main()
