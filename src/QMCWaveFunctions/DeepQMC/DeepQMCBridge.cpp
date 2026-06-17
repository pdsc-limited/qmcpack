//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#include "QMCWaveFunctions/DeepQMC/DeepQMCBridge.h"

#include <stdexcept>
#include <utility>

namespace qmcplusplus
{
namespace
{
class UnavailableDeepQMCBridge : public DeepQMCBridge
{
public:
  explicit UnavailableDeepQMCBridge(std::string reason) : reason_(std::move(reason)) {}

  BatchResult evaluateLogBatch(const std::vector<RealType>&,
                               const std::vector<RealType>&,
                               int,
                               int,
                               int) const override
  {
    throw std::runtime_error("DeepQMC inference bridge is not available: " + reason_);
  }

private:
  std::string reason_;
};
} // namespace

std::shared_ptr<const DeepQMCBridge> makeUnavailableDeepQMCBridge(std::string reason)
{
  return std::make_shared<UnavailableDeepQMCBridge>(std::move(reason));
}

} // namespace qmcplusplus
