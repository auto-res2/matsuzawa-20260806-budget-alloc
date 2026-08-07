"""Fetch Runs N' Poses and select the systems the experiment runs on.

Only three files are downloaded (about 470 MB total). The two large archives
in the same Zenodo record are deliberately untouched: `prediction_files.tar.gz`
(39.5 GB) holds other methods' coordinates, which this experiment does not
use, and `msa_files.tar.gz` (46 GB) holds the authors' MSAs, which would defeat
the point -- the MSA has to come from the same MSA-search call for every arm so
that the arms differ only in how the 25-structure budget is spent.

That matters operationally too: the group directory on the target cluster is
mounted read-only, so nothing survives between runs and every run re-fetches.
470 MB is a re-fetch you can afford; 39.5 GB is not.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import tarfile
import time
import urllib.request

import pandas as pd

log = logging.getLogger(__name__)

ZENODO_RECORD = "18366081"
ZENODO_FILE = "https://zenodo.org/records/{rec}/files/{key}?download=1"

# Bin edges from the authors' own plotting.py, so the stratification in this
# paper lines up with the stratification in theirs.
SIMILARITY_BINS = [0, 20, 30, 40, 50, 60, 70, 80, 100]

# Boltz-2 was trained to a 2023-06-01 cutoff, two years later than the four
# methods the benchmark was built around, so its similarity has to be measured
# against its own cutoff. Using the default column would understate how much of
# this population Boltz-2 has already seen.
SIM_COLUMN = "sucos_shape_pocket_qcov_2023"

# Below 20 every method fails almost uniformly and no allocation can separate;
# at 60 and above memorisation has already solved the system and the ceiling
# hides any difference.
SIM_LOW, SIM_HIGH = 20.0, 60.0

# MSA-search rejects any sequence containing a non-standard residue:
#   HTTP 422 {"error": "Value error, Invalid protein sequence: ...X..."}
# That is a real rejection, not a transient fault, so it is excluded here
# rather than left to fail mid-run. Filtering at selection keeps the
# population identical across arms by construction; discovering it at run
# time would drop the system from whichever arms happened to reach it.
# Exactly one system of 221 is affected.
STANDARD_RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")


def _download(key: str, dest_dir: str, retries: int = 6) -> str:
    """Fetch one Zenodo file, resuming rather than restarting on failure.

    The cluster's group directory is read-only, so nothing survives between
    runs and every run re-fetches this record -- including a 414 MB archive.
    Zenodo throttles that: a bare urlretrieve met HTTP 504 and killed a run
    before a single system had been processed. Retrying with a Range header
    resumes the partial file instead of paying for the whole transfer again.
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, key)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        log.info("cached %s (%.1f MB)", key, os.path.getsize(dest) / 1e6)
        return dest

    url = ZENODO_FILE.format(rec=ZENODO_RECORD, key=key)
    tmp = dest + ".part"
    last = None
    for attempt in range(retries):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        req = urllib.request.Request(url)
        if have:
            # Zenodo answers Range on GET (it reports 200 to HEAD, which is
            # why this looks unsupported if you probe it the obvious way).
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=600) as resp, \
                    open(tmp, "ab" if have and resp.status == 206 else "wb") as fh:
                if have and resp.status != 206:
                    fh.truncate(0)
                shutil.copyfileobj(resp, fh, 1024 * 1024)
            os.replace(tmp, dest)
            log.info("got %s (%.1f MB)", key, os.path.getsize(dest) / 1e6)
            return dest
        except Exception as exc:  # noqa: BLE001 - transport faults are expected here
            last = f"{type(exc).__name__}: {exc}"
            got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            sleep = min(120, 2 ** attempt * 6) + random.uniform(0, 5)
            log.warning("download of %s failed (%s) at %.1f MB, retry %d/%d "
                        "in %.0fs", key, last, got / 1e6, attempt + 1,
                        retries, sleep)
            time.sleep(sleep)
    raise RuntimeError(f"could not download {key} after {retries} attempts -- {last}")


def _extract_ground_truth(tgz: str, system_ids: list[str], out_dir: str) -> str:
    """Unpack only the selected systems, not all 2,600."""
    root = os.path.join(out_dir, "ground_truth")
    wanted = set(system_ids)
    missing = [s for s in wanted if not os.path.isdir(os.path.join(root, s))]
    if not missing:
        return root
    log.info("extracting %d/%d ground-truth systems", len(missing), len(wanted))
    prefixes = tuple(f"ground_truth/{s}/" for s in missing)
    with tarfile.open(tgz) as tf:
        members = [m for m in tf if m.name.startswith(prefixes)]
        tf.extractall(out_dir, members=members)
    return root


def _bin_of(sim: float) -> str:
    for lo, hi in zip(SIMILARITY_BINS, SIMILARITY_BINS[1:]):
        if lo <= sim < hi:
            return f"{lo}-{hi}"
    return f"{SIMILARITY_BINS[-2]}-{SIMILARITY_BINS[-1]}"


def select_systems(cache_dir: str, n_systems: int | None,
                   shard_index: int = 0, shard_count: int = 1) -> list[dict]:
    """Return the study population, deterministically ordered.

    `n_systems` truncates for the cheaper modes. The subset is taken evenly
    across the similarity range rather than off the top of an alphabetical
    list, so a sanity or pilot run spans easy and hard systems instead of
    landing entirely in one bin and reporting a number that cannot generalise.

    Determinism matters beyond reproducibility here: the three arms run as
    three separate jobs, and a paired comparison is only valid if all three
    select the identical system list.
    """
    ann_path = _download("annotations.csv", cache_dir)
    inputs_path = _download("inputs.json", cache_dir)

    ann = pd.read_csv(ann_path, low_memory=False)
    inputs = json.load(open(inputs_path))

    keep = (
        (ann["ligand_is_proper"] == True)  # noqa: E712 - pandas mask, not identity
        & (ann["num_proper_ligand_chains"] == 1)
        & ann[SIM_COLUMN].notna()
        & ann[SIM_COLUMN].between(SIM_LOW, SIM_HIGH, inclusive="left")
        & ann["ligand_smiles"].notna()
        & ann["system_id"].isin(inputs)
    )
    sel = ann[keep].sort_values(["system_id"]).reset_index(drop=True)
    log.info("population: %d systems in the %g-%g bin",
             len(sel), SIM_LOW, SIM_HIGH)

    if n_systems is not None and n_systems < len(sel):
        ordered = sel.sort_values([SIM_COLUMN, "system_id"]).reset_index(drop=True)
        step = len(ordered) / n_systems
        idx = [int(i * step) for i in range(n_systems)]
        sel = ordered.iloc[idx].sort_values("system_id").reset_index(drop=True)
        log.info("subsampled to %d systems spanning similarity %.1f-%.1f",
                 len(sel), sel[SIM_COLUMN].min(), sel[SIM_COLUMN].max())

    # Shard AFTER the deterministic ordering, interleaved rather than sliced
    # into blocks, so every shard spans the whole similarity range. Long runs
    # on this cluster get killed by the orchestrator (three times: at 1h55m,
    # and at 8h28m with 186 systems lost, because outputs on BYO Slurm are
    # only collected after a clean exit -- a killed run's checkpoints are
    # never retrieved). Short shards that finish are the only way to keep
    # work. Interleaving also means a lost shard costs coverage evenly
    # instead of removing one end of the difficulty range.
    if shard_count > 1:
        sel = sel.iloc[shard_index::shard_count].reset_index(drop=True)
        log.info("shard %d/%d: %d systems", shard_index, shard_count, len(sel))

    gt_tgz = _download("ground_truth.tar.gz", cache_dir)
    gt_root = _extract_ground_truth(gt_tgz, sel["system_id"].tolist(), cache_dir)

    systems = []
    for _, row in sel.iterrows():
        sid = row["system_id"]
        gt_dir = os.path.join(gt_root, sid)
        sdf = os.path.join(gt_dir, "ligand_files",
                           f"{row['ligand_instance_chain']}.sdf")
        cif = os.path.join(gt_dir, "system.cif")
        if not (os.path.exists(sdf) and os.path.exists(cif)):
            log.warning("skipping %s: ground truth incomplete", sid)
            continue
        seqs = inputs[sid]["sequences"]
        odd = {c for seq in seqs.values() for c in set(seq) - STANDARD_RESIDUES}
        if odd:
            log.warning("skipping %s: non-standard residues %s rejected by "
                        "MSA-search", sid, sorted(odd))
            continue
        systems.append({
            "system_id": sid,
            "ligand_chain": row["ligand_instance_chain"],
            "ligand_ccd": row["ligand_ccd_code"],
            "smiles": row["ligand_smiles"],
            "similarity": float(row[SIM_COLUMN]),
            "bin": _bin_of(float(row[SIM_COLUMN])),
            "sequences": seqs,
            "target_cif": cif,
            "target_sdf": sdf,
        })
    log.info("prepared %d systems", len(systems))
    return systems
