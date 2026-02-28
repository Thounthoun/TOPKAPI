// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "utils/topopt.hpp"

namespace palace
{
using namespace Catch;

TEST_CASE("TopOpt Heaviside projection", "[topopt][projection][Serial]")
{
  constexpr double eta = 0.5;

  SECTION("T1.5 - Heaviside forward")
  {
    for (double beta : {1.0, 8.0, 64.0})
    {
      const double denom = std::tanh(beta * eta) + std::tanh(beta * (1.0 - eta));
      for (double rho_tilde : {0.0, 0.25, eta, 0.75, 1.0})
      {
        const double expected =
            (std::tanh(beta * eta) + std::tanh(beta * (rho_tilde - eta))) / denom;
        REQUIRE(topopt::HeavisideProjection(rho_tilde, beta, eta) ==
                Approx(expected).margin(1.0e-12));
      }
      REQUIRE(topopt::HeavisideProjection(0.0, beta, eta) ==
              Approx(0.0).margin(1.0e-12));
      REQUIRE(topopt::HeavisideProjection(1.0, beta, eta) ==
              Approx(1.0).margin(1.0e-12));
      REQUIRE(topopt::HeavisideProjection(eta, beta, eta) ==
              Approx(eta).margin(1.0e-12));
    }
  }

  SECTION("T1.6 - Heaviside derivative")
  {
    for (double beta : {1.0, 8.0, 64.0})
    {
      const double denom = std::tanh(beta * eta) + std::tanh(beta * (1.0 - eta));
      double max_deriv = -1.0;
      for (double rho_tilde : {0.0, 0.25, eta, 0.75, 1.0})
      {
        const double t = std::tanh(beta * (rho_tilde - eta));
        const double expected = beta * (1.0 - t * t) / denom;
        const double deriv = topopt::HeavisideProjectionDerivative(rho_tilde, beta, eta);
        REQUIRE(deriv == Approx(expected).margin(1.0e-12));
        max_deriv = std::max(max_deriv, deriv);
      }
      REQUIRE(topopt::HeavisideProjectionDerivative(eta, beta, eta) ==
              Approx(max_deriv).margin(1.0e-12));
    }
  }

  SECTION("T1.7 - Stability at large beta")
  {
    constexpr double beta = 256.0;
    for (double rho_tilde : {0.0, 1.0e-8, eta, 1.0 - 1.0e-8, 1.0})
    {
      const double value = topopt::HeavisideProjection(rho_tilde, beta, eta);
      const double deriv = topopt::HeavisideProjectionDerivative(rho_tilde, beta, eta);
      REQUIRE(std::isfinite(value));
      REQUIRE(std::isfinite(deriv));
      REQUIRE(value >= 0.0);
      REQUIRE(value <= 1.0);
    }
  }
}

}  // namespace palace
