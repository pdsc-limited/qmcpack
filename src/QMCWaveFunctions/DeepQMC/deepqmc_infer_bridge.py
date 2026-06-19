# This file is distributed under the University of Illinois/NCSA Open Source License.
# See LICENSE file in top directory for details.

"""Python side of QMCPACK's experimental DeepQMC inference bridge.

The C++ WaveFunctionComponent calls DeepQMCInferBridge.compute_log_gl with a
batch of walker electron coordinates.  The returned quantities are in the
QMCPACK WaveFunctionComponent convention: log(psi), grad log(psi), and per
particle laplacian log(psi).

Import order is intentional.  The standalone miniapp found that importing
DeepQMC before JAX avoids initialization problems.
"""

import deepqmc  # noqa: F401  keep before jax
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import os
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from deepqmc.app import instantiate_ansatz
from deepqmc.molecule import Molecule
from deepqmc.hamil import MolecularHamiltonian
from deepqmc.log import CheckpointStore
from deepqmc.types import PhysicalConfiguration


class DeepQMCInferBridge:
    def __init__(self, model_path):
        # Prototype default mirrors the miniapp: a neutral He atom in bohr.
        self.mol = Molecule(coords=[[0.0, 0.0, 0.0]], charges=[2], charge=0, spin=0, unit='bohr')
        self.H = MolecularHamiltonian(mol=self.mol)

        deepqmc_dir = os.path.dirname(deepqmc.__file__)
        config_dir = os.path.join(deepqmc_dir, 'conf/ansatz')
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='default')

        _ansatz = instantiate(cfg, _recursive_=True, _convert_='all')
        self.ansatz = instantiate_ansatz(self.H, _ansatz)

        step, train_state = CheckpointStore.load(Path(model_path))
        params = train_state.params

        def drop_first_two_dims(x):
            if hasattr(x, 'ndim') and x.ndim >= 2:
                return x[0, 0]
            return x

        self.params = jax.tree_util.tree_map(drop_first_two_dims, params)

        ansatz = self.ansatz
        params = self.params

        def single_log_grad_lap(r, R, mol_idx):
            phys_conf = PhysicalConfiguration(R=R, r=r, mol_idx=jnp.array([mol_idx]))

            def log_psi_fn(r_flat):
                return ansatz.apply(params, jdc.replace(phys_conf, r=r_flat.reshape(-1, 3))).log

            r_flat = r.flatten()
            log_val = log_psi_fn(r_flat)
            grad_flat = jax.grad(log_psi_fn)(r_flat)
            hess = jax.hessian(log_psi_fn)(r_flat)
            diag = jnp.diag(hess).reshape(-1, 3)
            lap_per_electron = jnp.sum(diag, axis=1)
            return log_val, grad_flat.reshape(-1, 3), lap_per_electron

        @jax.jit
        def _compute_log_gl_batch(r_batch, R, mol_idx):
            vals, grads, laps = jax.vmap(lambda r: single_log_grad_lap(r, R, mol_idx))(r_batch)
            return vals, grads, laps

        self._compute_log_gl_batch = _compute_log_gl_batch

    def compute_log_gl(self, nuclear_coords, electron_coords, mol_idx, batch_size, n_elec):
        R = jnp.array(nuclear_coords).reshape(-1, 3)
        r_batch = jnp.array(electron_coords).reshape(batch_size, n_elec, 3)
        vals, grads, laps = self._compute_log_gl_batch(r_batch, R, mol_idx)
        jax.block_until_ready((vals, grads, laps))
        return vals.tolist(), grads.reshape(-1).tolist(), laps.reshape(-1).tolist()
