//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#include "catch.hpp"

#include "Configuration.h"
#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/DeepQMC/DeepQMCBridge.h"
#include "QMCWaveFunctions/DeepQMC/DeepQMCWaveFunctionComponent.h"

namespace qmcplusplus
{
using RealType = DeepQMCBridge::RealType;

namespace
{
class RecordingDeepQMCBridge : public DeepQMCBridge
{
public:
  BatchResult evaluateLogBatch(const std::vector<RealType>& ion_coords,
                               const std::vector<RealType>& electron_coords,
                               int mol_idx,
                               int batch_size,
                               int n_elec) const override
  {
    call_count++;
    last_ion_coords      = ion_coords;
    last_electron_coords = electron_coords;
    last_mol_idx         = mol_idx;
    last_batch_size      = batch_size;
    last_n_elec          = n_elec;

    BatchResult result;
    result.log_values.resize(batch_size);
    result.grad_log_values.resize(static_cast<std::size_t>(batch_size) * n_elec * OHMMS_DIM);
    result.lap_log_values.resize(static_cast<std::size_t>(batch_size) * n_elec);

    for (int iw = 0; iw < batch_size; ++iw)
    {
      result.log_values[iw] = 10.0 + iw;
      for (int iat = 0; iat < n_elec; ++iat)
      {
        const std::size_t particle_offset = static_cast<std::size_t>(iw) * n_elec + iat;
        for (int d = 0; d < OHMMS_DIM; ++d)
          result.grad_log_values[particle_offset * OHMMS_DIM + d] = 100.0 * iw + 10.0 * iat + d;
        result.lap_log_values[particle_offset] = 1000.0 * iw + iat;
      }
    }
    return result;
  }

  mutable int call_count = 0;
  mutable std::vector<RealType> last_ion_coords;
  mutable std::vector<RealType> last_electron_coords;
  mutable int last_mol_idx    = -1;
  mutable int last_batch_size = -1;
  mutable int last_n_elec     = -1;
};

ParticleSet makeElectrons(const SimulationCell& simulation_cell,
                          std::initializer_list<ParticleSet::SingleParticlePos> positions)
{
  ParticleSet electrons(simulation_cell);
  electrons.setName("e");
  electrons.create({static_cast<int>(positions.size())});
  int iat = 0;
  for (const auto& pos : positions)
    electrons.R[iat++] = pos;
  electrons.update();
  return electrons;
}

ParticleSet makeIons(const SimulationCell& simulation_cell)
{
  ParticleSet ions(simulation_cell);
  ions.setName("ion0");
  ions.create({1});
  ions.R[0] = {1.0, 2.0, 3.0};
  ions.update();
  return ions;
}
} // namespace

TEST_CASE("DeepQMCWaveFunctionComponent batched evaluateLog", "[wavefunction][deepqmc]")
{
  const SimulationCell simulation_cell;
  ParticleSet ions = makeIons(simulation_cell);
  ParticleSet elec0 = makeElectrons(simulation_cell, {{0.0, 0.1, 0.2}, {1.0, 1.1, 1.2}});
  ParticleSet elec1 = makeElectrons(simulation_cell, {{2.0, 2.1, 2.2}, {3.0, 3.1, 3.2}});

  auto bridge = std::make_shared<RecordingDeepQMCBridge>();
  DeepQMCWaveFunctionComponent comp0("deep", ions, bridge, 7);
  DeepQMCWaveFunctionComponent comp1("deep", ions, bridge, 7);

  ParticleSet::ParticleGradient G0, G1;
  ParticleSet::ParticleLaplacian L0, L1;
  G0.resize(elec0.getTotalNum());
  G1.resize(elec1.getTotalNum());
  L0.resize(elec0.getTotalNum());
  L1.resize(elec1.getTotalNum());
  G0 = 0.0;
  G1 = 0.0;
  L0 = 0.0;
  L1 = 0.0;

  RefVectorWithLeader<WaveFunctionComponent> wfc_list(comp0);
  wfc_list.push_back(comp0);
  wfc_list.push_back(comp1);
  RefVectorWithLeader<ParticleSet> p_list(elec0);
  p_list.push_back(elec0);
  p_list.push_back(elec1);
  RefVector<ParticleSet::ParticleGradient> G_list;
  G_list.push_back(G0);
  G_list.push_back(G1);
  RefVector<ParticleSet::ParticleLaplacian> L_list;
  L_list.push_back(L0);
  L_list.push_back(L1);

  comp0.mw_evaluateLog(wfc_list, p_list, G_list, L_list);

  CHECK(bridge->call_count == 1);
  CHECK(bridge->last_mol_idx == 7);
  CHECK(bridge->last_batch_size == 2);
  CHECK(bridge->last_n_elec == 2);
  REQUIRE(bridge->last_ion_coords.size() == 3);
  CHECK(bridge->last_ion_coords[0] == Approx(1.0));
  CHECK(bridge->last_ion_coords[1] == Approx(2.0));
  CHECK(bridge->last_ion_coords[2] == Approx(3.0));

  const std::vector<RealType> expected_electron_coords{0.0, 0.1, 0.2, 1.0, 1.1, 1.2,
                                                       2.0, 2.1, 2.2, 3.0, 3.1, 3.2};
  REQUIRE(bridge->last_electron_coords.size() == expected_electron_coords.size());
  for (int i = 0; i < expected_electron_coords.size(); ++i)
    CHECK(bridge->last_electron_coords[i] == Approx(expected_electron_coords[i]));

  CHECK(std::real(comp0.get_log_value()) == Approx(10.0));
  CHECK(std::real(comp1.get_log_value()) == Approx(11.0));

  CHECK(G0[0][0] == Approx(0.0));
  CHECK(G0[0][1] == Approx(1.0));
  CHECK(G0[0][2] == Approx(2.0));
  CHECK(G0[1][0] == Approx(10.0));
  CHECK(G0[1][1] == Approx(11.0));
  CHECK(G0[1][2] == Approx(12.0));
  CHECK(L0[0] == Approx(0.0));
  CHECK(L0[1] == Approx(1.0));

  CHECK(G1[0][0] == Approx(100.0));
  CHECK(G1[0][1] == Approx(101.0));
  CHECK(G1[0][2] == Approx(102.0));
  CHECK(G1[1][0] == Approx(110.0));
  CHECK(G1[1][1] == Approx(111.0));
  CHECK(G1[1][2] == Approx(112.0));
  CHECK(L1[0] == Approx(1000.0));
  CHECK(L1[1] == Approx(1001.0));
}

TEST_CASE("DeepQMCWaveFunctionComponent single walker delegates to batched evaluateLog", "[wavefunction][deepqmc]")
{
  const SimulationCell simulation_cell;
  ParticleSet ions = makeIons(simulation_cell);
  ParticleSet elec = makeElectrons(simulation_cell, {{0.0, 0.1, 0.2}, {1.0, 1.1, 1.2}});

  auto bridge = std::make_shared<RecordingDeepQMCBridge>();
  DeepQMCWaveFunctionComponent comp("deep", ions, bridge, 3);

  ParticleSet::ParticleGradient G;
  ParticleSet::ParticleLaplacian L;
  G.resize(elec.getTotalNum());
  L.resize(elec.getTotalNum());
  G = 0.0;
  L = 0.0;

  auto log_value = comp.evaluateLog(elec, G, L);

  CHECK(bridge->call_count == 1);
  CHECK(bridge->last_batch_size == 1);
  CHECK(bridge->last_n_elec == 2);
  CHECK(bridge->last_mol_idx == 3);
  CHECK(std::real(log_value) == Approx(10.0));
  CHECK(G[1][2] == Approx(12.0));
  CHECK(L[1] == Approx(1.0));
}

} // namespace qmcplusplus
