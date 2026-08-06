"""Spend a fixed 25-structure budget three different ways, then score it.

Every arm emits exactly `budget` structures. That is the whole design: prior
work reports that diversifying MSAs helps, but generates more models when it
does, so the benefit of diversity cannot be told apart from the benefit of
simply computing more. Holding the structure count equal separates them.

The three models are hosted NVIDIA NIM endpoints, so nothing here loads
weights or touches a GPU; the local work is HTTPS calls and structure parsing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .model import OpenStructureScorer

log = logging.getLogger(__name__)

NIM_BASE = "https://health.api.nvidia.com/v1/biology"
BOLTZ2_URL = f"{NIM_BASE}/mit/boltz2/predict"
DIFFDOCK_URL = f"{NIM_BASE}/mit/diffdock"
MSA_URL = f"{NIM_BASE}/colabfold/msa-search/predict"

# The MSA-search response is keyed by database. colabfold is the deepest and
# is what the Boltz ecosystem conventionally uses; the others are fallbacks.
MSA_DB_PREFERENCE = ["colabfold", "Uniref30_2302", "colabfold_envdb_202108"]


class NimError(RuntimeError):
    pass


def _api_key() -> str:
    for var in ("NVIDIA_API_KEY", "NGC_API_KEY"):
        val = os.environ.get(var)
        if val:
            return val.strip()
    raise NimError(
        "NVIDIA_API_KEY is not set. Every arm calls the hosted NIM endpoints, "
        "so the run cannot proceed without it."
    )


def nim_post(url: str, payload: dict, timeout: int = 1800,
             retries: int = 4) -> dict:
    """POST with backoff on the failures that are worth retrying.

    5xx and 429 are transient. 4xx other than 429 means the request itself is
    wrong and retrying just burns time, so those raise immediately.
    """
    key = _api_key()
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=body,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json",
                     "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:500].decode("utf-8", "replace")
            last = f"HTTP {exc.code}: {detail}"
            if exc.code < 500 and exc.code != 429:
                raise NimError(f"{url} rejected the request -- {last}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        sleep = min(60, 2 ** attempt * 5) + random.uniform(0, 3)
        log.warning("NIM call failed (%s), retry %d/%d in %.1fs",
                    last, attempt + 1, retries, sleep)
        time.sleep(sleep)
    raise NimError(f"{url} failed after {retries} attempts -- {last}")


# --------------------------------------------------------------------------
# MSA


def fetch_msa(sequence: str, cache_dir: str, max_sequences: int = 4096) -> str:
    """One MSA per unique sequence, cached on disk.

    Cached because the three arms run as three separate jobs and must receive
    byte-identical MSAs -- otherwise the arms differ by more than allocation.
    MSA-search was verified to return byte-identical alignments for repeated
    identical queries, so the cache is an optimisation and a guard, not a
    correctness crutch; inference records the digest either way.
    """
    os.makedirs(cache_dir, exist_ok=True)
    digest = hashlib.sha256(f"{sequence}|{max_sequences}".encode()).hexdigest()[:24]
    path = os.path.join(cache_dir, f"{digest}.a3m")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return open(path).read()

    resp = nim_post(MSA_URL, {
        "sequence": sequence,
        "databases": ["all"],
        "max_msa_sequences": max_sequences,
        "output_alignment_formats": ["a3m"],
    })
    alignments = resp.get("alignments", {})
    a3m = None
    for db in MSA_DB_PREFERENCE:
        entry = alignments.get(db)
        if isinstance(entry, dict) and "a3m" in entry:
            a3m = entry["a3m"]["alignment"]
            break
    if a3m is None:
        for entry in alignments.values():
            if isinstance(entry, dict) and "a3m" in entry:
                a3m = entry["a3m"]["alignment"]
                break
    if not a3m:
        raise NimError(f"MSA-search returned no a3m; keys={list(alignments)}")

    with open(path, "w") as fh:
        fh.write(a3m)
    return a3m


def parse_a3m(a3m: str) -> tuple[str, list[tuple[str, str]]]:
    """Split an a3m into its query record and the remaining homologs."""
    records, header, buf = [], None, []
    for line in a3m.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(buf)))
            header, buf = line, []
        elif header is not None:
            buf.append(line)
    if header is not None:
        records.append((header, "".join(buf)))
    if not records:
        raise ValueError("empty a3m")
    return records[0], records[1:]


def subsample_msa(a3m: str, n_variants: int, seed: int = 0) -> list[str]:
    """Depth-varied MSA variants, deterministically.

    Shallower alignments weaken the co-evolutionary signal for the dominant
    conformation, which is the mechanism reported to let AlphaFold-family
    models sample alternative states. Variant 0 is the full alignment, so the
    proposed arm always contains the baseline's input as one of its five.

    The query record is always kept: an alignment without it is not an
    alignment of this protein.
    """
    query, homologs = parse_a3m(a3m)
    fractions = [1.0, 0.5, 0.25, 0.125, 0.0625][:n_variants]
    while len(fractions) < n_variants:  # if someone asks for more than five
        fractions.append(fractions[-1] / 2)

    variants = []
    for i, frac in enumerate(fractions):
        if frac >= 1.0 or not homologs:
            chosen = homologs
        else:
            rng = random.Random(f"{seed}:{i}:{len(homologs)}")
            k = max(1, int(round(len(homologs) * frac)))
            chosen = rng.sample(homologs, k)
            # Keep input order; sampling should change depth, not ordering.
            order = {id(r): n for n, r in enumerate(homologs)}
            chosen.sort(key=lambda r: order[id(r)])
        lines = [query[0], query[1]]
        for hdr, seq in chosen:
            lines += [hdr, seq]
        variants.append("\n".join(lines) + "\n")
    return variants


# --------------------------------------------------------------------------
# Generators


def _polymers(sequences: dict[str, str], msa_by_seq: dict[str, str]) -> list[dict]:
    """Map the benchmark's chain labels onto Boltz-2's chain-id rules.

    Runs N' Poses names chains like "1.A", which Boltz-2 rejects (single letter
    or four alphanumerics). Identical sequences share one MSA, which is both
    correct and what keeps a homodimer from costing two MSA searches.
    """
    out = []
    for i, (_, seq) in enumerate(sorted(sequences.items())):
        out.append({
            "id": chr(ord("A") + i),
            "molecule_type": "protein",
            "sequence": seq,
            "msa": {"msa_search": {"a3m": {
                "alignment": msa_by_seq[seq], "format": "a3m", "rank": 0}}},
        })
    return out


def boltz2(sequences: dict[str, str], msa_by_seq: dict[str, str], smiles: str,
           n_samples: int, cfg) -> list[dict]:
    resp = nim_post(BOLTZ2_URL, {
        "polymers": _polymers(sequences, msa_by_seq),
        "ligands": [{"id": "L1", "smiles": smiles}],
        "recycling_steps": int(cfg.boltz2.recycling_steps),
        "sampling_steps": int(cfg.boltz2.sampling_steps),
        "diffusion_samples": int(n_samples),
    })
    structures = resp.get("structures") or []
    confidences = resp.get("confidence_scores") or []
    poses = []
    for i, struct in enumerate(structures):
        poses.append({
            "source": "boltz2",
            "cif": struct["structure"],
            "confidence": float(confidences[i]) if i < len(confidences) else None,
        })
    return poses


def diffdock(receptor_pdb_path: str, smiles: str, n_poses: int, cfg) -> list[dict]:
    """Dock into an already-predicted receptor.

    DiffDock takes ATOM records only, so the receptor written by the OST worker
    is already stripped of the co-folded ligand.
    """
    atom_only = "\n".join(
        line for line in open(receptor_pdb_path).read().splitlines()
        if line.startswith("ATOM")
    )
    resp = nim_post(DIFFDOCK_URL, {
        "protein": atom_only,
        "ligand": smiles,
        "ligand_file_type": "txt",
        "num_poses": int(n_poses),
        "time_divisions": int(cfg.diffdock.time_divisions),
        "steps": int(cfg.diffdock.steps),
    })
    sdfs = resp.get("ligand_positions") or []
    confs = resp.get("position_confidence") or []
    poses = []
    for i, sdf in enumerate(sdfs):
        raw = confs[i] if i < len(confs) else None
        poses.append({
            "source": "diffdock",
            "sdf": sdf,
            # DiffDock returns null confidences in some responses; the ranking
            # code must not treat that as a score.
            "confidence": float(raw) if isinstance(raw, (int, float)) else None,
        })
    return poses


# --------------------------------------------------------------------------
# Arms


def run_arm(system: dict, cfg, work_dir: str, scorer: OpenStructureScorer,
            msa_cache: str) -> dict:
    """Generate this arm's `budget` structures for one system and score them."""
    arm = cfg.run.arm
    budget = int(cfg.run.budget)
    sysdir = os.path.join(work_dir, system["system_id"])
    os.makedirs(sysdir, exist_ok=True)

    # One MSA per unique sequence, shared by every arm.
    msa_by_seq, msa_digests = {}, {}
    for seq in sorted(set(system["sequences"].values())):
        a3m = fetch_msa(seq, msa_cache, int(cfg.msa.max_sequences))
        msa_by_seq[seq] = a3m
        msa_digests[hashlib.sha256(seq.encode()).hexdigest()[:12]] = \
            hashlib.sha256(a3m.encode()).hexdigest()[:16]

    t0 = time.time()
    n_calls = 0

    if arm == "baseline":
        poses = boltz2(system["sequences"], msa_by_seq, system["smiles"],
                       budget, cfg)
        n_calls = 1

    elif arm == "proposed":
        n_variants = int(cfg.run.n_msa_variants)
        per = budget // n_variants
        poses = []
        for vi in range(n_variants):
            variant_by_seq = {
                seq: subsample_msa(a3m, n_variants, int(cfg.msa.seed))[vi]
                for seq, a3m in msa_by_seq.items()
            }
            got = boltz2(system["sequences"], variant_by_seq,
                         system["smiles"], per, cfg)
            for p in got:
                p["msa_variant"] = vi
            poses += got
            n_calls += 1

    elif arm == "lineage":
        n_dock = int(cfg.run.n_dock_poses)
        poses = boltz2(system["sequences"], msa_by_seq, system["smiles"],
                       budget - n_dock, cfg)
        n_calls = 1
        # Dock into the most confident co-folded receptor.
        best = max(range(len(poses)),
                   key=lambda i: (poses[i]["confidence"] is not None,
                                  poses[i]["confidence"] or 0.0))
        best_cif = os.path.join(sysdir, "receptor_src.cif")
        open(best_cif, "w").write(poses[best]["cif"])
        rec_pdb = os.path.join(sysdir, "receptor.pdb")
        info = scorer.receptor_pdb(best_cif, rec_pdb)
        if not info.get("ok"):
            raise RuntimeError(f"receptor extraction failed: {info.get('error')}")
        docked = diffdock(rec_pdb, system["smiles"], n_dock, cfg)
        for p in docked:
            p["receptor_cif"] = best_cif
        poses += docked
        n_calls += 1
    else:
        raise ValueError(f"unknown arm {arm!r}")

    gen_seconds = time.time() - t0

    # Materialise the poses, then score all of them the same way.
    jobs, pose_specs = [], []
    for i, pose in enumerate(poses):
        if pose["source"] == "boltz2":
            cif_path = os.path.join(sysdir, f"{arm}_{i:02d}.cif")
            open(cif_path, "w").write(pose["cif"])
            spec = {"model_cif": cif_path}
        else:
            sdf_path = os.path.join(sysdir, f"{arm}_{i:02d}.sdf")
            open(sdf_path, "w").write(pose["sdf"])
            spec = {"model_cif": pose["receptor_cif"], "model_ligand_sdf": sdf_path}
        pose_specs.append(spec)
        jobs.append({"id": f"{arm}_{i:02d}", **spec,
                     "target_cif": system["target_cif"],
                     "target_sdf": system["target_sdf"]})

    results = scorer.score(jobs)
    by_id = {r["id"]: r for r in results}
    for i, pose in enumerate(poses):
        r = by_id.get(f"{arm}_{i:02d}", {})
        pose["lddt_pli"] = r.get("lddt_pli")
        pose["scrmsd"] = r.get("scrmsd")
        pose["scoring_error"] = r.get("error")
        pose.pop("cif", None)
        pose.pop("sdf", None)

    record = {
        "system_id": system["system_id"],
        "similarity": system["similarity"],
        "bin": system["bin"],
        "ligand_ccd": system["ligand_ccd"],
        "arm": arm,
        "n_poses": len(poses),
        "n_api_calls": n_calls,
        "gen_seconds": round(gen_seconds, 1),
        "msa_digests": msa_digests,
        "poses": poses,
    }

    if bool(cfg.run.pose_matrix):
        t1 = time.time()
        pm = scorer.pose_matrix(pose_specs)
        record["pose_matrix"] = pm.get("matrix")
        record["pose_matrix_error"] = pm.get("error")
        record["pose_matrix_seconds"] = round(time.time() - t1, 1)

    return record


def run(cfg, systems: list[dict], results_dir: str) -> dict:
    """Execute this arm over every system. Returns the raw record set."""
    run_id = cfg.run.run_id
    out_dir = os.path.join(results_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.join(cfg.cache_dir, "work", run_id)
    msa_cache = os.path.join(cfg.cache_dir, "msa")
    scorer = OpenStructureScorer(os.path.join(cfg.cache_dir, "ost"))
    if not scorer.available():
        raise RuntimeError(
            f"OpenStructure interpreter not found at {scorer.ost_python}. It is "
            "built into the image by the Dockerfile; set OST_PYTHON to point at "
            "a local conda prefix when running outside the container."
        )

    records, failures = [], []

    def one(system: dict) -> dict | None:
        try:
            rec = run_arm(system, cfg, work_dir, scorer, msa_cache)
            log.info("%s %s: %d poses, oracle=%s, %.0fs",
                     run_id, system["system_id"], rec["n_poses"],
                     _fmt(_oracle(rec)), rec["gen_seconds"])
            return rec
        except Exception as exc:  # noqa: BLE001 - one system must not kill the run
            log.error("%s %s FAILED: %s", run_id, system["system_id"], exc)
            failures.append({"system_id": system["system_id"], "error": str(exc)})
            return None

    workers = max(1, int(cfg.run.workers))
    if workers == 1:
        for system in systems:
            rec = one(system)
            if rec:
                records.append(rec)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for rec in pool.map(one, systems):
                if rec:
                    records.append(rec)

    records.sort(key=lambda r: r["system_id"])
    payload = {"run_id": run_id, "arm": cfg.run.arm, "mode": cfg.mode,
               "budget": int(cfg.run.budget), "records": records,
               "failures": failures}
    with open(os.path.join(out_dir, "records.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def _oracle(record: dict) -> float | None:
    vals = [p["lddt_pli"] for p in record["poses"] if p.get("lddt_pli") is not None]
    return max(vals) if vals else None


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _main() -> None:
    """Subprocess entry point. main.py resolves config, this executes the arm."""
    import argparse

    from omegaconf import OmegaConf

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()

    spec = json.load(open(args.spec))
    cfg = OmegaConf.load(spec["config_path"])
    systems = json.load(open(spec["systems_path"]))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(cfg, systems, spec["results_dir"])


if __name__ == "__main__":
    _main()
