# DeepQMC Inference Prototype Status

Branch: `nn_wf_inference_proto`

## Goal

Prototype use of pretrained DeepQMC wavefunctions from QMCPACK via a new batch-first `WaveFunctionComponent`, gated behind `ENABLE_DEEPQMC_INFERENCE`.

## Key Design Decisions

- The feature is optional and must be compiled only when `ENABLE_DEEPQMC_INFERENCE=ON`.
- The implementation must be batch/multi-walker first.
- `WaveFunctionComponent::mw_evaluateLog(...)` is the primary execution path.
- Single-walker APIs should be compatibility wrappers around the multi-walker implementation, not separate implementations.
- Particle-by-particle legacy APIs are not a performance target for the prototype; they should either full-recompute through the batch path or fail clearly until explicitly supported.
- The DeepQMC bridge should return QMCPACK-native quantities directly:
  - `log(psi)` per walker
  - `grad log(psi)` per electron per walker
  - `laplacian log(psi)` per electron per walker
- Use the existing miniapp at `../deepqmc_infer_miniapp` as proof-of-concept source material.

## Initial Reference Points

Miniapp files:

- `../deepqmc_infer_miniapp/src/DeepQMCWaveFunction.hpp`
- `../deepqmc_infer_miniapp/src/DeepQMCWaveFunction.cpp`
- `../deepqmc_infer_miniapp/src/deepqmc_infer_bridge.py`
- `../deepqmc_infer_miniapp/src/main.cpp`

QMCPACK integration examples:

- `src/QMCWaveFunctions/WaveFunctionComponent.h`
- `src/QMCWaveFunctions/WaveFunctionFactory.cpp`
- `src/QMCWaveFunctions/ExampleHeComponent.*`
- `src/QMCWaveFunctions/ExampleHeBuilder.*`

## Proposed Files

Likely new files:

- `src/QMCWaveFunctions/DeepQMC/DeepQMCWaveFunctionComponent.h`
- `src/QMCWaveFunctions/DeepQMC/DeepQMCWaveFunctionComponent.cpp`
- `src/QMCWaveFunctions/DeepQMC/DeepQMCWaveFunctionBuilder.h`
- `src/QMCWaveFunctions/DeepQMC/DeepQMCWaveFunctionBuilder.cpp`
- `src/QMCWaveFunctions/DeepQMC/DeepQMCBridge.h`
- `src/QMCWaveFunctions/DeepQMC/DeepQMCBridge.cpp`
- `src/QMCWaveFunctions/DeepQMC/deepqmc_infer_bridge.py`

Likely modified files:

- top-level or relevant CMake option definition for `ENABLE_DEEPQMC_INFERENCE`
- `src/QMCWaveFunctions/CMakeLists.txt`
- `src/QMCWaveFunctions/WaveFunctionFactory.cpp`
- `src/QMCWaveFunctions/tests/CMakeLists.txt`

## XML Sketch

```xml
<wavefunction name="psi0" target="e">
  <deepqmc name="DNN"
           source="ion0"
           model="runs/02/training/chkpt-10000.pt"
           mol_idx="0"
           python_module_path="...optional..." />
</wavefunction>
```

## Current Work Items

1. Explore `WaveFunctionComponent` and `TrialWaveFunction` batch APIs and test patterns.
2. Explore CMake optional feature gating patterns.
3. Explore miniapp bridge details and required API changes for log-derivative output.
4. Implement a mockable batch bridge interface. **Initial version done.**
5. Implement `DeepQMCWaveFunctionComponent::mw_evaluateLog(...)` against the bridge. **Initial version done.**
6. Add single-walker compatibility wrapper for `evaluateLog(...)`. **Initial version done.**
7. Register XML builder behind `ENABLE_DEEPQMC_INFERENCE`. **Initial version done; real Python bridge construction still placeholder.**
8. Add unit tests that do not require DeepQMC by using a mock bridge. **Initial direct component test added.**
9. Add optional real DeepQMC integration test later.
10. Replace placeholder unavailable bridge in `DeepQMCWaveFunctionBuilder` with the real Python/JAX bridge.
11. Verify compile/tests in a configured QMCPACK build with required dependencies.

## Progress Log

### 2026-06-17

- Created branch `nn_wf_inference_proto` from `develop`.
- Established batch-first design constraints.
- Created this running status document.
- Launched read-only exploration subagents for:
  - WaveFunctionComponent batch API details
  - CMake gating patterns
  - DeepQMC miniapp bridge behavior
- Subagents using the current harness/agent overrides again appeared to hang after 1-2 tool calls; they were steered to stop. Do not rely on them for current findings.
- Added initial gated prototype files:
  - `src/QMCWaveFunctions/DeepQMC/DeepQMCBridge.h/.cpp`
  - `src/QMCWaveFunctions/DeepQMC/DeepQMCWaveFunctionComponent.h/.cpp`
  - `src/QMCWaveFunctions/DeepQMC/DeepQMCWaveFunctionBuilder.h/.cpp`
  - `src/QMCWaveFunctions/tests/test_deepqmc_wfc.cpp`
- Added `ENABLE_DEEPQMC_INFERENCE` CMake option and conditional source/test inclusion.
- Registered `<deepqmc>` in `WaveFunctionFactory`; with the option disabled it aborts clearly, with the option enabled it constructs the prototype component.
- Implemented batch-first `DeepQMCWaveFunctionComponent::mw_evaluateLog(...)` against a mockable `DeepQMCBridge` interface.
- Implemented `evaluateLog(...)` as a one-walker wrapper around `mw_evaluateLog(...)`.
- Prototype particle-by-particle APIs (`ratio`, `ratioGrad`, `evalGrad`) currently throw clear unsupported errors.

## Open Questions

- Exact CMake location and naming convention for `ENABLE_DEEPQMC_INFERENCE`.
- Whether Python bridge file should be installed/copied beside the executable, embedded, or loaded from XML path.
- Whether first prototype should support only fixed ion coordinates from QMCPACK input or allow model-provided molecule metadata.
- How to manage Python/JAX initialization and GPU platform selection in MPI runs.
- Whether `ratio`/`ratioGrad` should initially full-recompute through batch size 1 or throw unsupported for non-compatible drivers.

## Validation Status

Implementation started and the initial scaffold compiles in the updated Spack environment at `/workspace/spack_env/qmcpack`.

Successful configure command used explicit MPI wrapper compilers because the active environment had stale `CC/CXX/FC` values pointing at missing view `clang` symlinks:

```bash
CC=$(which mpicc) CXX=$(which mpicxx) FC=$(which mpifort) \
  cmake -S . -B build-deepqmc -G Ninja \
  -DENABLE_DEEPQMC_INFERENCE=ON \
  -DBUILD_UNIT_TESTS=ON \
  -DBUILD_AFQMC=OFF \
  -DBUILD_MICRO_BENCHMARKS=OFF
```

Build passed:

```bash
cmake --build build-deepqmc --target test_wavefunction_trialwf -j 8
```

Full `ctest -R deterministic-unit_test_wavefunction_trialwf` initially failed in `TrialWaveFunction_diamondC_1x1x1` with a SIGSEGV. ASAN/debug exposed the root cause as a LAPACK ABI issue:

```text
** On entry to DGETRI parameter number 6 had an illegal value
Xgetri failed with error -6
```

The Spack env had `openblas+ilp64`, while QMCPACK's generic BLAS/LAPACK path expected LP64 integers. Fixed the env by changing `/workspace/spack_env/qmcpack/spack.yaml` to `openblas~ilp64`, then running:

```bash
cd /workspace/spack_env/qmcpack
spack concretize -f
spack install
```

After reconfiguring/rebuilding, full trialwf unit test passed:

```bash
ctest --test-dir build-deepqmc -R deterministic-unit_test_wavefunction_trialwf --output-on-failure
```

Result: 100% tests passed, 1/1.

Targeted DeepQMC tests also passed:

```bash
cd build-deepqmc/src/QMCWaveFunctions/tests
./test_wavefunction_trialwf "DeepQMCWaveFunctionComponent*" --success
```

Result: all targeted DeepQMC tests passed, 46 assertions in 2 test cases.
