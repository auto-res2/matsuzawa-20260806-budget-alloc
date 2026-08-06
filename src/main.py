"""Orchestrator for one run_id: resolve mode, launch the arm, report metrics.

Per AGENTS.md this file holds no inference logic. It applies the mode scaling,
runs `src.inference` as a subprocess, then turns the raw per-pose records into
the metrics the paper reports.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from .preprocess import select_systems

log = logging.getLogger(__name__)

# Success as Runs N' Poses defines it: the conjunction, not lDDT-PLI alone.
# lDDT-PLI on its own accepts a pose whose contacts are broadly right but whose
# conformation is wrong, which is why the authors pair it with an RMSD cut.
SUCCESS_LDDT_PLI = 0.8
SUCCESS_RMSD = 2.0

# Two poses within this distance count as the same mode. Same cut as the
# success criterion, so "a distinct mode" and "a distinguishable pose" agree.
MODE_RMSD = 2.0


def _finite(values):
    return [v for v in values
            if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]


def _rank_key(pose: dict) -> tuple:
    """Order poses by the arm's own confidence.

    Runs N' Poses shows confidence scores are not comparable across methods
    (optimal thresholds ranged 0.75-0.99), so in the mixed-lineage arm raw
    values from Boltz-2 and DiffDock must not be compared directly. They are
    converted to within-generator percentile ranks before merging, and poses
    with no confidence at all sort last rather than being read as zero.
    """
    conf = pose.get("_conf_pct")
    return (conf is not None, conf if conf is not None else -1.0)


def _assign_percentiles(poses: list[dict]) -> None:
    by_source: dict[str, list[dict]] = {}
    for p in poses:
        by_source.setdefault(p.get("source", "?"), []).append(p)
    for group in by_source.values():
        scored = [p for p in group if isinstance(p.get("confidence"), (int, float))]
        scored.sort(key=lambda p: p["confidence"])
        n = len(scored)
        for i, p in enumerate(scored):
            p["_conf_pct"] = (i + 0.5) / n if n else None
        for p in group:
            p.setdefault("_conf_pct", None)


def _matrix_nan_fraction(matrix) -> float | None:
    """How much of the pose-pose matrix failed to score.

    A NaN pair can never merge, so it necessarily raises the mode count. If
    one arm produces more unscoreable pairs than another -- plausible for the
    mixed-lineage arm, whose poses come from two different file formats --
    the diagnostic would read as extra diversity that is really extra
    failure. Reported so that reading is available rather than hidden.
    """
    if not matrix:
        return None
    n = len(matrix)
    total = bad = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            v = matrix[i][j]
            if v is None or not math.isfinite(v):
                bad += 1
    return (bad / total) if total else None


def _unique_modes(matrix, n_poses: int) -> int | None:
    """Count pose modes by complete-linkage clustering at MODE_RMSD.

    Complete linkage (rather than single) so that a cluster only forms when
    every member is within the cut of every other; single linkage would chain
    distinct modes together through intermediate poses.
    """
    if not matrix:
        return None
    clusters = [[i] for i in range(n_poses)]
    while True:
        best, pair = None, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                worst = 0.0
                ok = True
                for i in clusters[a]:
                    for j in clusters[b]:
                        v = matrix[i][j]
                        if v is None or not math.isfinite(v):
                            ok = False
                            break
                        worst = max(worst, v)
                    if not ok:
                        break
                if ok and worst <= MODE_RMSD and (best is None or worst < best):
                    best, pair = worst, (a, b)
        if pair is None:
            break
        a, b = pair
        clusters[a] = clusters[a] + clusters[b]
        clusters.pop(b)
    return len(clusters)


def compute_metrics(payload: dict) -> dict:
    """Per-system metrics plus their aggregates."""
    per_system, n_scored_poses, n_total_poses = [], 0, 0

    for rec in payload["records"]:
        poses = rec["poses"]
        n_total_poses += len(poses)
        lddt = _finite([p.get("lddt_pli") for p in poses])
        n_scored_poses += len(lddt)
        if not lddt:
            per_system.append({
                "system_id": rec["system_id"], "bin": rec["bin"],
                "similarity": rec["similarity"], "scored": False,
            })
            continue

        _assign_percentiles(poses)
        ranked = sorted(poses, key=_rank_key, reverse=True)
        top = next((p for p in ranked if p.get("lddt_pli") is not None), None)

        success = any(
            p.get("lddt_pli") is not None and p.get("scrmsd") is not None
            and p["lddt_pli"] > SUCCESS_LDDT_PLI and p["scrmsd"] < SUCCESS_RMSD
            for p in poses
        )
        rmsds = _finite([p.get("scrmsd") for p in poses])
        per_system.append({
            "system_id": rec["system_id"],
            "bin": rec["bin"],
            "similarity": rec["similarity"],
            "scored": True,
            "oracle_lddt_pli": max(lddt),
            "top_ranked_lddt_pli": top["lddt_pli"] if top else None,
            "ligand_rmsd": min(rmsds) if rmsds else None,
            "success": bool(success),
            "unique_pose_modes": _unique_modes(rec.get("pose_matrix"), len(poses)),
            "pose_matrix_nan_fraction": _matrix_nan_fraction(rec.get("pose_matrix")),
            "n_scored": len(lddt),
            "n_poses": len(poses),
            "gen_seconds": rec.get("gen_seconds"),
            "n_api_calls": rec.get("n_api_calls"),
        })

    scored = [s for s in per_system if s.get("scored")]

    def mean(key):
        vals = _finite([s.get(key) for s in scored])
        return sum(vals) / len(vals) if vals else None

    metrics = {
        "run_id": payload["run_id"],
        "arm": payload["arm"],
        "mode": payload["mode"],
        "n_systems": len(per_system),
        "n_systems_scored": len(scored),
        "oracle_lddt_pli": mean("oracle_lddt_pli"),
        "top_ranked_lddt_pli": mean("top_ranked_lddt_pli"),
        "ligand_rmsd": mean("ligand_rmsd"),
        "success_rate": (sum(1 for s in scored if s["success"]) / len(scored)
                         if scored else None),
        "unique_pose_modes": mean("unique_pose_modes"),
        "pose_matrix_nan_fraction": mean("pose_matrix_nan_fraction"),
        # Reported, never silently applied: if scoring drops systems, the
        # reader has to be able to see how many and which.
        "scoring_coverage": (n_scored_poses / n_total_poses) if n_total_poses else None,
        "system_coverage": (len(scored) / len(per_system)) if per_system else None,
        "unscored_systems": [s["system_id"] for s in per_system if not s.get("scored")],
        "n_failures": len(payload.get("failures", [])),
        "mean_gen_seconds": mean("gen_seconds"),
        "per_system": per_system,
    }
    return metrics


def _validate(metrics: dict, mode: str) -> tuple[bool, str, dict]:
    """The machine-parsed verdict. Inference task, so it checks samples."""
    minimum = {"sanity": 3, "pilot": 15}.get(mode, 1)
    scored = metrics["n_systems_scored"]
    summary = {
        "systems": metrics["n_systems"],
        "systems_scored": scored,
        "oracle_lddt_pli": metrics["oracle_lddt_pli"],
        "success_rate": metrics["success_rate"],
        "scoring_coverage": metrics["scoring_coverage"],
    }
    if scored < minimum:
        return False, f"too_few_scored_systems_{scored}_of_{minimum}", summary
    if metrics["oracle_lddt_pli"] is None:
        return False, "missing_metrics", summary
    if not math.isfinite(metrics["oracle_lddt_pli"]):
        return False, "non_finite_primary_metric", summary

    # Identical values across every pose would mean the generator is not
    # actually varying, which would make the whole comparison vacuous.
    distinct = {round(s["oracle_lddt_pli"], 6) for s in metrics["per_system"]
                if s.get("scored")}
    if scored > 1 and len(distinct) == 1:
        return False, "identical_outputs_across_systems", summary
    return True, "", summary


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    mode = cfg.mode
    n_systems = cfg.run.n_systems.get(mode, None)
    if n_systems is not None:
        n_systems = int(n_systems)
    results_dir = os.path.abspath(cfg.results_dir)
    cache_dir = os.path.abspath(cfg.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    run_dir = os.path.join(results_dir, cfg.run.run_id)
    os.makedirs(run_dir, exist_ok=True)

    log.info("run_id=%s arm=%s mode=%s n_systems=%s",
             cfg.run.run_id, cfg.run.arm, mode, n_systems)

    systems = select_systems(cache_dir, n_systems)
    if not systems:
        print("SANITY_VALIDATION: FAIL reason=no_systems_selected", flush=True)
        sys.exit(1)

    spec_dir = os.path.join(cache_dir, "spec", cfg.run.run_id)
    os.makedirs(spec_dir, exist_ok=True)
    systems_path = os.path.join(spec_dir, "systems.json")
    config_path = os.path.join(spec_dir, "config.yaml")
    spec_path = os.path.join(spec_dir, "spec.json")
    json.dump(systems, open(systems_path, "w"))
    resolved = OmegaConf.to_container(cfg, resolve=True)
    resolved["cache_dir"] = cache_dir
    OmegaConf.save(OmegaConf.create(resolved), config_path)
    json.dump({"config_path": config_path, "systems_path": systems_path,
               "results_dir": results_dir}, open(spec_path, "w"))

    proc = subprocess.run(
        [sys.executable, "-u", "-m", "src.inference", "--spec", spec_path],
        cwd=hydra.utils.get_original_cwd(),
    )
    if proc.returncode != 0:
        verdict = "SANITY_VALIDATION" if mode == "sanity" else "PILOT_VALIDATION"
        print(f"{verdict}: FAIL reason=inference_subprocess_rc_{proc.returncode}",
              flush=True)
        sys.exit(proc.returncode)

    payload = json.load(open(os.path.join(run_dir, "records.json")))
    metrics = compute_metrics(payload)
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    _log_wandb(cfg, metrics, mode)

    ok, reason, summary = _validate(metrics, mode)
    if mode in ("sanity", "pilot"):
        tag = "SANITY_VALIDATION" if mode == "sanity" else "PILOT_VALIDATION"
        print(f"{tag}: {'PASS' if ok else 'FAIL reason=' + reason}", flush=True)
        print(f"{tag}_SUMMARY: {json.dumps(summary)}", flush=True)
        if not ok:
            sys.exit(1)
    else:
        print(f"FULL_SUMMARY: {json.dumps(summary)}", flush=True)


def _log_wandb(cfg: DictConfig, metrics: dict, mode: str) -> None:
    """W&B must never be the reason an experiment dies."""
    try:
        import wandb

        project = cfg.wandb.project
        if mode in ("sanity", "pilot"):
            project = f"{project}-{mode}"
        run = wandb.init(
            entity=cfg.wandb.entity, project=project, name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )
        scalar = {k: v for k, v in metrics.items()
                  if isinstance(v, (int, float)) or v is None}
        wandb.log(scalar)
        run.summary.update(scalar)
        print(f"WANDB_RUN_URL: {run.url}", flush=True)
        wandb.finish()
    except Exception as exc:  # noqa: BLE001
        log.warning("W&B logging skipped: %s", exc)


if __name__ == "__main__":
    main()
