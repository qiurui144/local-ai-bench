# Reproducibility

Every benchmark run in this repository is recorded with sufficient
information to recreate it later. The contract:

> Given a `reproducibility.json` produced by
> `benchmark.rigor.reproducibility.ReproducibilitySnapshot.capture()`,
> a future maintainer on different hardware can rebuild the same
> dependency set, fetch the same data files, and run the same code
> path. Multi-seed noise is the only allowed source of divergence.

## What gets captured

`ReproducibilitySnapshot.capture()` records:

1. **Code state** — `git rev-parse HEAD`, branch, porcelain `git
   status`, and a truncated `git diff --stat` if the tree is dirty.
2. **Python environment** — `pip freeze` (full) plus `pip list
   --not-required` (top-level only, easier to read).
3. **Hardware / OS** — CPU model, logical/physical cores, total RAM,
   GPU summary via `nvidia-smi` / `rocm-smi`, kernel, distribution.
4. **Data inputs** — for every dataset path passed in, file size and
   SHA256. Directories are recursed.
5. **Timestamp** — Unix time.

## How to use it

```python
from pathlib import Path
from benchmark.rigor.reproducibility import ReproducibilitySnapshot

snap = ReproducibilitySnapshot.capture(
    data_paths=[Path("golden/expectations.json")],
    repo_root=Path("."),
)
snap.write(Path("reports/runs/2026-05-26-001/reproducibility.json"))
```

The snapshot must be written alongside any `manifest.json` produced
by `benchmark.rigor.multi_seed_runner.write_manifest`. That pair is
what we archive.

## Pinning policy

- `requirements.txt` pins to specific versions of all runtime deps.
  Use `pip-compile` to refresh; PR a fresh pin alongside any model /
  framework change.
- `requirements-dev.txt` is separate and looser; dev tooling versions
  do not affect benchmark numbers.
- Embedding library versions (`sentence-transformers`, `transformers`,
  `numpy`) are pinned to exact patch versions because dimensions and
  output ranges have changed across minor versions.

## Hardware classes

If a benchmark requires hardware (GPU), the hardware class is recorded
in the snapshot. Re-running on a different class is allowed; the
expected drift is documented in [`baselines.md`](baselines.md).

## Seed policy

- All multi-seed runs default to seeds `(0, 1, 2)`. Choose a larger
  set only when a smaller difference must be detected.
- The seed is passed into `multi_seed_runner.pin_seeds` which sets
  `random`, `numpy`, and `PYTHONHASHSEED`. Frameworks with their own
  RNGs (PyTorch, TensorFlow) must be seeded inside the user-supplied
  `run_fn`.

## Container reproducibility (optional)

For maximum reproducibility, run inside a container with the exact
runtime image. The `requirements.txt` is sufficient for a CPU-only
container; GPU runs additionally need a base image with the
CUDA/ROCm driver matching `nvidia-smi`'s driver_version field in the
snapshot.

A minimal Dockerfile sketch:

```dockerfile
FROM python:3.12-slim
WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "benchmark.rag.regression_ci"]
```

## When reproducibility fails

If two runs of the same configuration produce metric distributions
that fail an alignment check (`AlignmentChecker.compare`), the first
thing to inspect is the diff between the two snapshots:

```bash
diff <(jq -S . run1/reproducibility.json) \
     <(jq -S . run2/reproducibility.json)
```

Common culprits:

- A transitively-updated package in `pip freeze`.
- A different GPU driver `driver_version`.
- A different `cpu_model` (some kernels have CPU-architecture
  branches).

## References

- Pineau, J. et al. (2021). Improving Reproducibility in Machine
  Learning Research. JMLR.
- ACL Reproducibility Checklist (2020).
- Sculley, D. et al. (2018). Winner's Curse? On Pace, Progress, and
  Empirical Rigor.
