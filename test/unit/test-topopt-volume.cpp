// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "utils/topopt.hpp"

namespace palace
{
using namespace Catch;

TEST_CASE("TopOpt volume metrics", "[topopt][volume][Serial]")
{
  SECTION("T1.8 - Volume constraint")
  {
    mfem::Vector rho(3), element_volume(3);
    rho = 0.5;
    element_volume(0) = 1.0;
    element_volume(1) = 2.0;
    element_volume(2) = 3.0;

    REQUIRE(topopt::VolumeFraction(rho, element_volume) ==
            Approx(0.5).margin(1.0e-14));

    const mfem::Vector grad = topopt::VolumeFractionGradient(element_volume);
    REQUIRE(grad(0) == Approx(1.0 / 6.0).margin(1.0e-14));
    REQUIRE(grad(1) == Approx(2.0 / 6.0).margin(1.0e-14));
    REQUIRE(grad(2) == Approx(3.0 / 6.0).margin(1.0e-14));
  }

  SECTION("T1.9 - Binarization measure")
  {
    mfem::Vector rho_hat(4), element_volume(4);
    element_volume = 1.0;

    rho_hat = 0.5;
    REQUIRE(topopt::BinarizationMeasure(rho_hat, element_volume) ==
            Approx(1.0).margin(1.0e-14));

    rho_hat(0) = 0.0;
    rho_hat(1) = 1.0;
    rho_hat(2) = 0.0;
    rho_hat(3) = 1.0;
    REQUIRE(topopt::BinarizationMeasure(rho_hat, element_volume) ==
            Approx(0.0).margin(1.0e-14));
  }
}

}  // namespace palace
