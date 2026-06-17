//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_DEEPQMCWAVEFUNCTIONCOMPONENT_H
#define QMCPLUSPLUS_DEEPQMCWAVEFUNCTIONCOMPONENT_H

#include <memory>
#include <string>
#include <vector>

#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "QMCWaveFunctions/DeepQMC/DeepQMCBridge.h"

namespace qmcplusplus
{

class DeepQMCWaveFunctionComponent : public WaveFunctionComponent
{
public:
  DeepQMCWaveFunctionComponent(std::string name,
                               const ParticleSet& ions,
                               std::shared_ptr<const DeepQMCBridge> bridge,
                               int mol_idx);

  std::string getClassName() const override { return "DeepQMCWaveFunctionComponent"; }

  LogValue evaluateLog(const ParticleSet& P,
                       ParticleSet::ParticleGradient& G,
                       ParticleSet::ParticleLaplacian& L) override;

  void mw_evaluateLog(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                      const RefVectorWithLeader<ParticleSet>& p_list,
                      const RefVector<ParticleSet::ParticleGradient>& G_list,
                      const RefVector<ParticleSet::ParticleLaplacian>& L_list) const override;

  void acceptMove(ParticleSet& P, int iat, bool safe_to_delay = false) override {}
  void restore(int iat) override {}
  PsiValue ratio(ParticleSet& P, int iat) override;
  GradType evalGrad(ParticleSet& P, int iat) override;
  PsiValue ratioGrad(ParticleSet& P, int iat, GradType& grad_iat) override;

  void registerData(ParticleSet& P, WFBufferType& buf) override {}
  LogValue updateBuffer(ParticleSet& P, WFBufferType& buf, bool fromscratch = false) override;
  void copyFromBuffer(ParticleSet& P, WFBufferType& buf) override {}

  void evaluateDerivatives(ParticleSet& P,
                           const OptVariables& optvars,
                           Vector<ValueType>& dlogpsi,
                           Vector<ValueType>& dhpsioverpsi) override;

  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet& tpq) const override;

  const ParticleSet& getIons() const { return ions_; }
  const DeepQMCBridge& getBridge() const { return *bridge_; }
  int getMolIdx() const { return mol_idx_; }

private:
  static std::vector<RealType> flattenIonCoords(const ParticleSet& ions);
  static void appendElectronCoords(const ParticleSet& electrons, std::vector<RealType>& electron_coords);

  const ParticleSet& ions_;
  std::shared_ptr<const DeepQMCBridge> bridge_;
  int mol_idx_;
};

} // namespace qmcplusplus

#endif
