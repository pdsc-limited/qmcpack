#!/usr/bin/env python3
"""Train a small DeepQMC He checkpoint for QMCPACK inference prototyping.

This script intentionally mirrors the miniapp proof-of-concept setup and writes a
DeepQMC CheckpointStore checkpoint usable by QMCPACK's experimental
<deepqmc model="..."> wavefunction component.
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

# DeepQMC must be imported before jax. Keep this order.
import deepqmc  # noqa: F401
import jax
import jax.tree_util

# DeepQMC versions used in this prototype still refer to jax.tree_map.
jax.tree_map = jax.tree_util.tree_map

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from deepqmc.app import instantiate_ansatz
from deepqmc.hamil import MolecularHamiltonian
from deepqmc.log import CheckpointStore
from deepqmc.molecule import Molecule
from deepqmc.observable import SpinMonitor
from deepqmc.sampling import DecorrSampler, MetropolisSampler, combine_samplers, initialize_sampling
from deepqmc.train import train


def build_default_he_problem():
    mol = Molecule(coords=[[0.0, 0.0, 0.0]], charges=[2], charge=0, spin=0, unit="bohr")
    hamiltonian = MolecularHamiltonian(mol=mol)

    deepqmc_dir = os.path.dirname(deepqmc.__file__)
    config_dir = os.path.join(deepqmc_dir, "conf/ansatz")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="default")
    ansatz_cfg = instantiate(cfg, _recursive_=True, _convert_="all")
    ansatz = instantiate_ansatz(hamiltonian, ansatz_cfg)

    opt_config_dir = os.path.join(deepqmc_dir, "conf/task/opt")
    with initialize_config_dir(version_base=None, config_dir=opt_config_dir):
        opt_cfg = compose(config_name="kfac")
    optimizer = instantiate(opt_cfg, _recursive_=True, _convert_="all")

    elec_sampler = partial(combine_samplers, samplers=[DecorrSampler(length=20), MetropolisSampler])
    sampler_factory = partial(initialize_sampling, elec_sampler=elec_sampler)
    return hamiltonian, ansatz, optimizer, sampler_factory


def latest_checkpoint(training_dir: Path) -> Path:
    checkpoints = sorted(
        training_dir.glob("chkpt-*.pt"),
        key=lambda path: CheckpointStore.extract_step_from_filename(path.name),
    )
    if not checkpoints:
        raise RuntimeError(f"No DeepQMC checkpoints were written in {training_dir}")
    return checkpoints[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="/workspace/qmcpack/deepqmc_runs/he_proto")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--electron-batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-spin-monitor", action="store_true")
    args = parser.parse_args()

    H, ansatz, optimizer, sampler_factory = build_default_he_problem()
    observable_monitors = [] if args.no_spin_monitor else [SpinMonitor(period=1, save_samples=True)]

    train(
        H,
        ansatz,
        optimizer,
        sampler_factory,
        steps=args.steps,
        electron_batch_size=args.electron_batch_size,
        seed=args.seed,
        workdir=args.workdir,
        observable_monitors=observable_monitors,
    )

    checkpoint = latest_checkpoint(Path(args.workdir) / "training")
    print(f"DeepQMC He checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
