"""The lDDT-PLI scorer, and the bridge that reaches it.

OpenStructure is the reference implementation of lDDT-PLI and is what
PLINDER's own evaluation uses, but it cannot live in this process: it is
conda-only, and its aarch64 build of 2.11.1 requires Python 3.12 while the
repository's CLI contract fixes this side at 3.11. So it sits in its own
prefix (`/opt/ost`, built in the Dockerfile) and is driven as a subprocess
exchanging JSON.

The worker source lives here as a string and is materialised at run time.
That keeps the file list within what AGENTS.md permits while keeping the
scorer's definition next to the code that calls it.

Writing a numpy lDDT-PLI instead was tried and rejected: without
`add_mdl_contacts=True` a naive implementation measures only recall of the
reference's contacts and never counts the false contacts a model invents, so
it scores degenerate predictions (a ligand collapsed into the pocket centre)
up to 0.31 too generously. OST has that defence on by default. Call OST.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess

log = logging.getLogger(__name__)

# Where the Dockerfile puts the conda prefix. Overridable so the same code
# runs against a locally built prefix during development.
OST_PYTHON = os.environ.get("OST_PYTHON", "/opt/ost/bin/python")

WORKER_SRC = r'''
"""Score co-folded protein-ligand predictions with OpenStructure."""
from __future__ import annotations

import json
import sys
import traceback

from ost import conop, io, mol
from ost.mol.alg.ligand_scoring_lddtpli import LDDTPLIScorer
from ost.mol.alg.ligand_scoring_scrmsd import SCRMSDScorer

# plinder.eval overrides exactly one OST default: substructure_match (False
# upstream). lddt_pli_radius and the four thresholds are already OST defaults;
# naming them keeps the settings auditable rather than implied.
PLINDER_KWARGS = dict(
    substructure_match=True,
    resnum_alignments=False,
    coverage_delta=0.2,
)
LDDT_KWARGS = dict(
    add_mdl_contacts=True,
    lddt_pli_radius=6.0,
    lddt_pli_thresholds=[0.5, 1.0, 2.0, 4.0],
)


def _load_model(path):
    """Load a predicted complex and give its ligand real bonds.

    Boltz-2 writes the ligand as residue LIG1 in a non-poly chain. That name is
    absent from the compound library, so it arrives with no connectivity and
    OST declines to score it as a disconnected graph -- returning no number and
    raising nothing, which is the failure mode that silently eats systems.
    HeuristicProcessor assigns bonds from interatomic distances.
    """
    ent = io.LoadMMCIF(path, fault_tolerant=True)
    conop.HeuristicProcessor().Process(ent)
    return ent


def _first_score(scorer):
    """OST returns {model_chain: {resnum: value}}; take the single ligand."""
    for _, per_res in scorer.score.items():
        for _, value in per_res.items():
            return float(value)
    return None


def score_one(job):
    out = {"id": job["id"], "lddt_pli": None, "scrmsd": None,
           "assigned": False, "error": None}
    try:
        model = _load_model(job["model_cif"])
        model_rec = mol.CreateEntityFromView(model.Select("peptide=true"), True)

        if job.get("model_ligand_sdf"):
            # A docked pose: ligand from SDF, receptor from the co-folded cif.
            model_lig = io.LoadSDF(job["model_ligand_sdf"])
        else:
            sel = model.Select("peptide=false and ele!=H and water=false")
            model_lig = mol.CreateEntityFromView(sel, True)

        target = io.LoadMMCIF(job["target_cif"], fault_tolerant=True)
        target_rec = mol.CreateEntityFromView(target.Select("peptide=true"), True)
        target_lig = io.LoadSDF(job["target_sdf"])

        lddt = LDDTPLIScorer(model_rec, target_rec, [model_lig], [target_lig],
                             **PLINDER_KWARGS, **LDDT_KWARGS)
        out["lddt_pli"] = _first_score(lddt)
        out["assigned"] = out["lddt_pli"] is not None

        rmsd = SCRMSDScorer(model_rec, target_rec, [model_lig], [target_lig],
                            **PLINDER_KWARGS)
        out["scrmsd"] = _first_score(rmsd)
    except Exception:  # noqa: BLE001 - one bad system must not kill the batch
        out["error"] = traceback.format_exc(limit=4)
    return out


def receptor_pdb(job):
    """Write the model receptor as PDB. DiffDock takes ATOM records only."""
    out = {"id": job["id"], "ok": False, "error": None}
    try:
        ent = _load_model(job["model_cif"])
        rec = mol.CreateEntityFromView(ent.Select("peptide=true"), True)
        io.SavePDB(rec, job["out_pdb"])
        out["ok"] = True
        out["n_atoms"] = rec.atom_count
    except Exception:  # noqa: BLE001
        out["error"] = traceback.format_exc(limit=4)
    return out


def pose_matrix(job):
    """Pairwise symmetry-corrected RMSD among one system's poses.

    Feeds unique_pose_modes. Symmetry correction matters: a ring flip relates
    two chemically identical poses, and counting them as distant would inflate
    apparent diversity -- exactly the quantity this metric exists to measure.

    Each pose is scored against pose 0 as the reference frame, which is what
    SCRMSDScorer's binding-site superposition gives us for free.
    """
    out = {"id": job["id"], "matrix": None, "error": None}
    try:
        loaded = []
        for spec in job["poses"]:
            ent = _load_model(spec["model_cif"])
            rec = mol.CreateEntityFromView(ent.Select("peptide=true"), True)
            if spec.get("model_ligand_sdf"):
                lig = io.LoadSDF(spec["model_ligand_sdf"])
            else:
                sel = ent.Select("peptide=false and ele!=H and water=false")
                lig = mol.CreateEntityFromView(sel, True)
            loaded.append((rec, lig))

        n = len(loaded)
        m = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                ri, li = loaded[i]
                rj, lj = loaded[j]
                try:
                    s = SCRMSDScorer(ri, rj, [li], [lj], **PLINDER_KWARGS)
                    v = _first_score(s)
                except Exception:  # noqa: BLE001
                    v = None
                v = float("nan") if v is None else float(v)
                m[i][j] = m[j][i] = v
        out["matrix"] = m
    except Exception:  # noqa: BLE001
        out["error"] = traceback.format_exc(limit=4)
    return out


def main():
    spec = json.load(sys.stdin)
    op = spec.get("op", "score")
    fn = {"score": score_one, "receptor_pdb": receptor_pdb,
          "pose_matrix": pose_matrix}[op]
    json.dump({"results": [fn(j) for j in spec["jobs"]]}, sys.stdout)


if __name__ == "__main__":
    main()
'''


class OpenStructureScorer:
    """Drives the OpenStructure worker in its own conda prefix."""

    def __init__(self, work_dir: str, ost_python: str | None = None,
                 timeout: int = 7200):
        self.ost_python = ost_python or OST_PYTHON
        self.timeout = timeout
        os.makedirs(work_dir, exist_ok=True)
        self.worker_path = os.path.join(work_dir, "_ost_worker.py")
        with open(self.worker_path, "w") as fh:
            fh.write(WORKER_SRC)

    def available(self) -> bool:
        return bool(shutil.which(self.ost_python) or os.path.exists(self.ost_python))

    def _call(self, op: str, jobs: list[dict]) -> list[dict]:
        if not jobs:
            return []
        payload = json.dumps({"op": op, "jobs": jobs})
        proc = subprocess.run(
            [self.ost_python, self.worker_path],
            input=payload, capture_output=True, text=True, timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"OpenStructure worker failed (op={op}, rc={proc.returncode}): "
                f"{proc.stderr[-2000:]}"
            )
        try:
            return json.loads(proc.stdout)["results"]
        except (ValueError, KeyError) as exc:
            raise RuntimeError(
                f"OpenStructure worker returned unparseable output (op={op}): "
                f"{proc.stdout[:500]!r} / stderr {proc.stderr[-1000:]}"
            ) from exc

    def score(self, jobs: list[dict]) -> list[dict]:
        return self._call("score", jobs)

    def receptor_pdb(self, model_cif: str, out_pdb: str) -> dict:
        res = self._call("receptor_pdb", [
            {"id": "r", "model_cif": model_cif, "out_pdb": out_pdb}])
        return res[0]

    def pose_matrix(self, poses: list[dict]) -> dict:
        return self._call("pose_matrix", [{"id": "m", "poses": poses}])[0]
