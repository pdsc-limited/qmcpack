//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#include "QMCWaveFunctions/DeepQMC/DeepQMCWaveFunctionComponent.h"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace qmcplusplus
{
namespace
{
void validateResultShape(const DeepQMCBridge::BatchResult& result, int batch_size, int n_elec)
{
  const std::size_t batch     = static_cast<std::size_t>(batch_size);
  const std::size_t particles = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(n_elec);
  if (result.log_values.size() != batch)
    throw std::runtime_error("DeepQMC bridge returned wrong number of log values");
  if (result.grad_log_values.size() != particles * OHMMS_DIM)
    throw std::runtime_error("DeepQMC bridge returned wrong number of gradient values");
  if (result.lap_log_values.size() != particles)
    throw std::runtime_error("DeepQMC bridge returned wrong number of laplacian values");
}

[[noreturn]] void throwPbypUnsupported(const char* method_name)
{
  std::ostringstream msg;
  msg << "DeepQMCWaveFunctionComponent::" << method_name
      << " is not supported in the prototype. DeepQMC inference is batch/multi-walker first; "
         "use walker-level/Crowd evaluation paths.";
  throw std::runtime_error(msg.str());
}
} // namespace

DeepQMCWaveFunctionComponent::DeepQMCWaveFunctionComponent(std::string name,
                                                           const ParticleSet& ions,
                                                           std::shared_ptr<const DeepQMCBridge> bridge,
                                                           int mol_idx)
    : WaveFunctionComponent(std::move(name)), ions_(ions), bridge_(std::move(bridge)), mol_idx_(mol_idx)
{
  if (!bridge_)
    throw std::runtime_error("DeepQMCWaveFunctionComponent requires a non-null DeepQMCBridge");
}

std::vector<DeepQMCWaveFunctionComponent::RealType> DeepQMCWaveFunctionComponent::flattenIonCoords(
    const ParticleSet& ions)
{
  std::vector<RealType> ion_coords;
  ion_coords.reserve(ions.getTotalNum() * OHMMS_DIM);
  for (int iat = 0; iat < ions.getTotalNum(); ++iat)
    for (int d = 0; d < OHMMS_DIM; ++d)
      ion_coords.push_back(ions.R[iat][d]);
  return ion_coords;
}

void DeepQMCWaveFunctionComponent::appendElectronCoords(const ParticleSet& electrons,
                                                        std::vector<RealType>& electron_coords)
{
  for (int iat = 0; iat < electrons.getTotalNum(); ++iat)
    for (int d = 0; d < OHMMS_DIM; ++d)
      electron_coords.push_back(electrons.R[iat][d]);
}

DeepQMCWaveFunctionComponent::LogValue DeepQMCWaveFunctionComponent::evaluateLog(
    const ParticleSet& P,
    ParticleSet::ParticleGradient& G,
    ParticleSet::ParticleLaplacian& L)
{
  auto& mutable_p = const_cast<ParticleSet&>(P);
  RefVectorWithLeader<WaveFunctionComponent> wfc_list(*this);
  wfc_list.push_back(*this);
  RefVectorWithLeader<ParticleSet> p_list(mutable_p);
  p_list.push_back(mutable_p);
  RefVector<ParticleSet::ParticleGradient> G_list;
  G_list.push_back(G);
  RefVector<ParticleSet::ParticleLaplacian> L_list;
  L_list.push_back(L);

  mw_evaluateLog(wfc_list, p_list, G_list, L_list);
  return log_value_;
}

void DeepQMCWaveFunctionComponent::mw_evaluateLog(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const RefVector<ParticleSet::ParticleGradient>& G_list,
    const RefVector<ParticleSet::ParticleLaplacian>& L_list) const
{
  if (wfc_list.size() != p_list.size() || wfc_list.size() != G_list.size() || wfc_list.size() != L_list.size())
    throw std::runtime_error("DeepQMCWaveFunctionComponent::mw_evaluateLog list size mismatch");
  if (wfc_list.empty())
    return;

  const auto& leader = wfc_list.getCastedLeader<DeepQMCWaveFunctionComponent>();
  const int batch_size = static_cast<int>(wfc_list.size());
  const int n_elec     = static_cast<int>(p_list[0].getTotalNum());

  std::vector<RealType> electron_coords;
  electron_coords.reserve(static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(n_elec) * OHMMS_DIM);
  for (int iw = 0; iw < batch_size; ++iw)
  {
    if (p_list[iw].getTotalNum() != n_elec)
      throw std::runtime_error("DeepQMCWaveFunctionComponent requires all walkers in a batch to have the same electron count");
    appendElectronCoords(p_list[iw], electron_coords);
  }

  const std::vector<RealType> ion_coords = flattenIonCoords(leader.ions_);
  DeepQMCBridge::BatchResult result =
      leader.bridge_->evaluateLogBatch(ion_coords, electron_coords, leader.mol_idx_, batch_size, n_elec);
  validateResultShape(result, batch_size, n_elec);

  for (int iw = 0; iw < batch_size; ++iw)
  {
    auto& component = wfc_list.getCastedElement<DeepQMCWaveFunctionComponent>(iw);
    component.log_value_ = LogValue(result.log_values[iw]);

    auto& G = G_list[iw].get();
    auto& L = L_list[iw].get();
    if (G.size() < n_elec)
      G.resize(n_elec);
    if (L.size() < n_elec)
      L.resize(n_elec);

    for (int iat = 0; iat < n_elec; ++iat)
    {
      const std::size_t particle_offset = (static_cast<std::size_t>(iw) * n_elec + iat);
      GradType grad;
      for (int d = 0; d < OHMMS_DIM; ++d)
        grad[d] = result.grad_log_values[particle_offset * OHMMS_DIM + d];
      G[iat] += grad;
      L[iat] += result.lap_log_values[particle_offset];
    }
  }
}

DeepQMCWaveFunctionComponent::PsiValue DeepQMCWaveFunctionComponent::ratio(ParticleSet& P, int iat)
{
  throwPbypUnsupported("ratio");
}

DeepQMCWaveFunctionComponent::GradType DeepQMCWaveFunctionComponent::evalGrad(ParticleSet& P, int iat)
{
  throwPbypUnsupported("evalGrad");
}

DeepQMCWaveFunctionComponent::PsiValue DeepQMCWaveFunctionComponent::ratioGrad(ParticleSet& P,
                                                                                int iat,
                                                                                GradType& grad_iat)
{
  throwPbypUnsupported("ratioGrad");
}

DeepQMCWaveFunctionComponent::LogValue DeepQMCWaveFunctionComponent::updateBuffer(ParticleSet& P,
                                                                                  WFBufferType& buf,
                                                                                  bool fromscratch)
{
  return evaluateLog(P, P.G, P.L);
}

void DeepQMCWaveFunctionComponent::evaluateDerivatives(ParticleSet& P,
                                                       const OptVariables& optvars,
                                                       Vector<ValueType>& dlogpsi,
                                                       Vector<ValueType>& dhpsioverpsi)
{
  // Pretrained DeepQMC inference parameters are fixed in this prototype.
}

std::unique_ptr<WaveFunctionComponent> DeepQMCWaveFunctionComponent::makeClone(ParticleSet& tpq) const
{
  return std::make_unique<DeepQMCWaveFunctionComponent>(my_name_, ions_, bridge_, mol_idx_);
}

} // namespace qmcplusplus
