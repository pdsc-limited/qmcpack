#!/usr/bin/env python3
"""Run an application-level DeepQMC vmc_batch walkers-per-crowd benchmark.

This script generates small He all-electron ``vmc_batch`` inputs with
``crowds=1`` and varying ``total_walkers``.  For a one-thread QMCPACK run this
means the effective batched crowd size is ``walkers_per_crowd=total_walkers``.
It then runs QMCPACK for selected JAX platforms and summarizes BlockCPU timing.

Example:
  PYTHONPATH=/path/to/deepqmc/site-packages \
  LD_LIBRARY_PATH=/path/to/python/lib:/path/to/qmcpack/view/lib:${LD_LIBRARY_PATH:-} \
  python utils/deepqmc/benchmark_vmc_batch_crowdsize.py \
    --qmcpack ./build-deepqmc-gpu/bin/qmcpack \
    --checkpoint /path/to/deepqmc/chkpt.pt \
    --platforms cpu cuda \
    --cuda-visible-devices 0
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
from typing import Iterable


DEFAULT_WALKERS = "1,2,4,8,16,32,64,128,256,512,1024,2048"


def parse_int_list(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    return values


def write_input(path: Path, project_id: str, checkpoint: Path, walkers: int, blocks: int, steps: int, substeps: int) -> None:
    path.write_text(
        f'''<?xml version="1.0"?>
<simulation>
  <project id="{project_id}" series="0"/>
  <random seed="11"/>

  <particleset name="ion0" size="1">
    <group name="He"><parameter name="charge">2</parameter></group>
    <attrib name="position" datatype="posArray">0.0 0.0 0.0</attrib>
  </particleset>

  <particleset name="e" random="yes" randomsrc="ion0">
    <group name="u" size="1"><parameter name="charge">-1</parameter></group>
    <group name="d" size="1"><parameter name="charge">-1</parameter></group>
  </particleset>

  <wavefunction name="psi0" target="e">
    <deepqmc name="DNN" source="ion0" model="{checkpoint}" mol_idx="0"/>
  </wavefunction>

  <hamiltonian name="h0" type="generic" target="e">
    <pairpot name="ElecElec" type="coulomb" source="e" target="e"/>
    <pairpot name="Coulomb" type="coulomb" source="ion0" target="e"/>
  </hamiltonian>

  <qmc method="vmc_batch" move="pbyp">
    <estimators><estimator name="LocalEnergy" hdf5="no"/></estimators>
    <parameter name="total_walkers">{walkers}</parameter>
    <parameter name="crowds">1</parameter>
    <parameter name="blocks">{blocks}</parameter>
    <parameter name="steps">{steps}</parameter>
    <parameter name="subSteps">{substeps}</parameter>
    <parameter name="warmupSteps">0</parameter>
    <parameter name="timestep">0.1</parameter>
  </qmc>
</simulation>
'''
    )


def scalar_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            rows.append(
                {
                    "index": int(parts[0]),
                    "local_energy": float(parts[1]),
                    "block_weight": float(parts[7]),
                    "block_cpu": float(parts[8]),
                    "accept_ratio": float(parts[9]),
                }
            )
    return rows


def summarize_run(run_dir: Path, platform: str, walkers: int, elapsed_sec: float, steps: int, substeps: int, nelec: int) -> dict[str, float | int | str]:
    scalar_files = sorted(run_dir.glob("*.scalar.dat"))
    if not scalar_files:
        raise RuntimeError(f"No scalar.dat file found in {run_dir}")
    rows = scalar_rows(scalar_files[0])
    if not rows:
        raise RuntimeError(f"No scalar rows found in {scalar_files[0]}")

    block_cpu = [row["block_cpu"] for row in rows]
    steady_block_cpu = block_cpu[1:] if len(block_cpu) > 1 else block_cpu
    mean_block_cpu = statistics.mean(steady_block_cpu)
    block_count = len(rows)
    walker_steps_per_block = walkers * steps * substeps
    electron_moves_per_block = walker_steps_per_block * nelec

    return {
        "platform": platform,
        "walkers_per_crowd": walkers,
        "blocks": block_count,
        "elapsed_sec": elapsed_sec,
        "block0_sec": block_cpu[0],
        "mean_block_sec_excl0": mean_block_cpu,
        "ms_per_step_per_walker": mean_block_cpu / walker_steps_per_block * 1.0e3,
        "us_per_electron_move": mean_block_cpu / electron_moves_per_block * 1.0e6,
        "accept_ratio_mean": statistics.mean(row["accept_ratio"] for row in rows),
        "local_energy_mean": statistics.mean(row["local_energy"] for row in rows),
    }


def write_csv(path: Path, rows: Iterable[dict[str, float | int | str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    by_walkers: dict[int, dict[str, dict[str, float | int | str]]] = {}
    for row in rows:
        by_walkers.setdefault(int(row["walkers_per_crowd"]), {})[str(row["platform"])] = row

    lines = [
        "# DeepQMC vmc_batch walkers-per-crowd benchmark",
        "",
        "Mean block time excludes block 0. Inputs use `crowds=1`, so `walkers_per_crowd=total_walkers` for a one-thread QMCPACK run.",
        "",
        "| walkers/crowd | CPU block s | CPU ms/step/walker | GPU block s | GPU ms/step/walker | GPU speedup |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for walkers in sorted(by_walkers):
        cpu = by_walkers[walkers].get("cpu")
        gpu = by_walkers[walkers].get("cuda")
        cpu_block = float(cpu["mean_block_sec_excl0"]) if cpu else float("nan")
        gpu_block = float(gpu["mean_block_sec_excl0"]) if gpu else float("nan")
        speedup = cpu_block / gpu_block if cpu and gpu else float("nan")

        cpu_cols = f"{cpu_block:.6f} | {float(cpu['ms_per_step_per_walker']):.3f}" if cpu else "nan | nan"
        gpu_cols = f"{gpu_block:.6f} | {float(gpu['ms_per_step_per_walker']):.3f}" if gpu else "nan | nan"
        speedup_col = f"{speedup:.2f}x" if cpu and gpu else "nan"
        lines.append(f"| {walkers} | {cpu_cols} | {gpu_cols} | {speedup_col} |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qmcpack", required=True, type=Path, help="Path to a DeepQMC-enabled qmcpack executable")
    parser.add_argument("--checkpoint", required=True, type=Path, help="DeepQMC checkpoint .pt file")
    parser.add_argument("--output", type=Path, default=Path("deepqmc_vmc_batch_crowdsize"), help="Output directory")
    parser.add_argument("--walkers", type=parse_int_list, default=parse_int_list(DEFAULT_WALKERS), help="Comma-separated total_walkers values")
    parser.add_argument("--platforms", nargs="+", default=["cpu", "cuda"], choices=["cpu", "cuda"], help="JAX platforms to run")
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--omp-num-threads", default="1", help="OMP_NUM_THREADS value for each run")
    parser.add_argument("--cuda-visible-devices", default="0", help="CUDA_VISIBLE_DEVICES value for cuda runs")
    parser.add_argument("--skip-existing", action="store_true", help="Parse existing completed run directories instead of rerunning them")
    args = parser.parse_args()

    if not args.qmcpack.exists():
        raise SystemExit(f"qmcpack executable does not exist: {args.qmcpack}")
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")

    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for platform in args.platforms:
        for walkers in args.walkers:
            run_dir = args.output / platform / f"walkers_{walkers}"
            run_dir.mkdir(parents=True, exist_ok=True)
            input_path = run_dir / "deepqmc_he_vmc_batch.xml"
            project_id = f"deepqmc_he_vmc_batch_{platform}_walkers_{walkers}"
            write_input(input_path, project_id, args.checkpoint.resolve(), walkers, args.blocks, args.steps, args.substeps)

            log_path = run_dir / "run.log"
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(args.omp_num_threads)
            env["JAX_PLATFORMS"] = platform
            if platform == "cuda":
                env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
            else:
                env.pop("CUDA_VISIBLE_DEVICES", None)

            elapsed_sec = None
            if args.skip_existing and log_path.exists():
                match = re.search(r"elapsed_sec\s+([0-9.]+)", log_path.read_text(errors="replace"))
                if match:
                    elapsed_sec = float(match.group(1))
            if elapsed_sec is None:
                for old_file in run_dir.glob(f"{project_id}.s000.*"):
                    old_file.unlink()
                start = time.monotonic()
                with log_path.open("w") as log:
                    print(f"=== {platform} walkers_per_crowd={walkers} ===", file=log)
                    log.flush()
                    subprocess.run([str(args.qmcpack.resolve()), input_path.name], cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    elapsed_sec = time.monotonic() - start
                    print(f"elapsed_sec {elapsed_sec:.6f}", file=log)

            rows.append(summarize_run(run_dir, platform, walkers, elapsed_sec, args.steps, args.substeps, nelec=2))
            write_csv(args.output / "summary.csv", rows)
            write_markdown(args.output / "summary.md", rows)
            print(f"{platform:4s} walkers={walkers:5d} elapsed={elapsed_sec:.3f}s")

    print(f"Wrote {args.output / 'summary.csv'}")
    print(f"Wrote {args.output / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
