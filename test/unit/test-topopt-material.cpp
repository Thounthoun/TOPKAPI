// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <memory>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "models/materialoperator.hpp"
#include "test-helpers.hpp"
#include "utils/communication.hpp"
#include "utils/iodata.hpp"
#include "utils/topopt.hpp"
#include "utils/units.hpp"

namespace palace
{
using namespace Catch;

namespace
{

struct MaterialOperatorFixture
{
  Units units;
  IoData iodata;
  std::unique_ptr<mfem::Mesh> serial_mesh;
  std::unique_ptr<mfem::ParMesh> par_mesh;
  Mesh palace_mesh;
  MaterialOperator mat_op;

  MaterialOperatorFixture(double eps = 2.0, double sigma = 0.0, double tandelta = 0.0)
      : units(1.0, 1.0),
        iodata(units),
        serial_mesh(std::make_unique<mfem::Mesh>(SingleTetMesh())),
        par_mesh(std::make_unique<mfem::ParMesh>(Mpi::World(), *serial_mesh)),
        palace_mesh(std::move(par_mesh)),
        mat_op([&]() -> MaterialOperator {
          auto &material = iodata.domains.materials.emplace_back();
          material.attributes = {1};
          material.epsilon_r.s = {eps, eps, eps};
          material.tandelta.s = {tandelta, tandelta, tandelta};
          material.sigma.s = {sigma, sigma, sigma};
          return MaterialOperator(iodata, palace_mesh);
        }())
  {
  }
};

struct TwoAttributeMaterialOperatorFixture
{
  Units units;
  IoData iodata;
  std::unique_ptr<mfem::Mesh> serial_mesh;
  Mesh palace_mesh;
  MaterialOperator mat_op;

  TwoAttributeMaterialOperatorFixture()
      : units(1.0, 1.0),
        iodata(units),
        serial_mesh(std::make_unique<mfem::Mesh>(
            mfem::Mesh::MakeCartesian3D(2, 1, 1, mfem::Element::HEXAHEDRON))),
        palace_mesh([&]() -> std::unique_ptr<mfem::ParMesh> {
          serial_mesh->SetAttribute(0, 1);
          serial_mesh->SetAttribute(1, 2);
          serial_mesh->SetAttributes();
          return std::make_unique<mfem::ParMesh>(Mpi::World(), *serial_mesh);
        }()),
        mat_op([&]() -> MaterialOperator {
          auto &material1 = iodata.domains.materials.emplace_back();
          material1.attributes = {1};
          material1.epsilon_r.s = {2.0, 2.0, 2.0};
          material1.sigma.s = {0.0, 0.0, 0.0};

          auto &material2 = iodata.domains.materials.emplace_back();
          material2.attributes = {2};
          material2.epsilon_r.s = {2.0, 2.0, 2.0};
          material2.sigma.s = {0.0, 0.0, 0.0};

          return MaterialOperator(iodata, palace_mesh);
        }())
  {
  }
};

void CheckDiagonalMatrix(const mfem::DenseMatrix &M, double value)
{
  for (int i = 0; i < M.Height(); i++)
  {
    for (int j = 0; j < M.Width(); j++)
    {
      const double expected = (i == j) ? value : 0.0;
      REQUIRE(M(i, j) == Approx(expected).margin(1.0e-14));
    }
  }
}

}  // namespace

TEST_CASE("TopOpt material mutators", "[topopt][materialoperator][Serial]")
{
  SECTION("T1.1 - UpdatePermittivityReal")
  {
    MaterialOperatorFixture fixture(2.0, 0.25);
    fixture.mat_op.UpdatePermittivityReal(1, 4.5);

    CheckDiagonalMatrix(fixture.mat_op.GetPermittivityReal(1), 4.5);
    CheckDiagonalMatrix(fixture.mat_op.GetConductivityReal(1), 0.25);
    REQUIRE(fixture.mat_op.GetLightSpeedMax(1) ==
            Approx(1.0 / std::sqrt(4.5)).margin(1.0e-12));
  }

  SECTION("T1.1b - UpdatePermittivityReal preserves loss tangent")
  {
    constexpr double eps_old = 2.0;
    constexpr double eps_new = 5.0;
    constexpr double tandelta = 0.02;

    MaterialOperatorFixture fixture(eps_old, 0.0, tandelta);
    fixture.mat_op.UpdatePermittivityReal(1, eps_new);

    CheckDiagonalMatrix(fixture.mat_op.GetPermittivityReal(1), eps_new);
    CheckDiagonalMatrix(fixture.mat_op.GetPermittivityImag(1), -eps_new * tandelta);
  }

  SECTION("T1.2 - UpdateConductivityReal")
  {
    MaterialOperatorFixture fixture(2.0, 0.1);
    fixture.mat_op.UpdateConductivityReal(1, 3.25);

    CheckDiagonalMatrix(fixture.mat_op.GetConductivityReal(1), 3.25);
    CheckDiagonalMatrix(fixture.mat_op.GetPermittivityReal(1), 2.0);
  }
}

TEST_CASE("TopOpt interpolation formulas", "[topopt][materialoperator][Serial]")
{
  SECTION("T1.3 - n-squared interpolation (Chen)")
  {
    constexpr double n_low = 1.0;
    constexpr double n_high = 3.48;
    for (double rho : {0.0, 0.25, 0.5, 0.75, 1.0})
    {
      const double n = n_low + rho * (n_high - n_low);
      const double expected_eps = n * n;
      const double expected_deriv = 2.0 * n * (n_high - n_low);

      REQUIRE(topopt::InterpolatePermittivityNSquared(rho, n_low, n_high) ==
              Approx(expected_eps).margin(1.0e-14));
      REQUIRE(topopt::InterpolatePermittivityNSquaredDerivative(rho, n_low, n_high) ==
              Approx(expected_deriv).margin(1.0e-14));
    }
  }

  SECTION("T1.4 - log-linear conductivity interpolation (Aage)")
  {
    constexpr double sigma_d = 1.0e-4;
    constexpr double sigma_m = 1.0e6;
    constexpr double sigma_0 = sigma_m;
    for (double rho : {0.0, 0.5, 1.0})
    {
      const double log_sigma = std::log10(sigma_d / sigma_0) +
                               rho * (std::log10(sigma_m / sigma_0) -
                                      std::log10(sigma_d / sigma_0));
      const double expected_sigma = sigma_0 * std::pow(10.0, log_sigma);
      const double expected_deriv = expected_sigma * std::log(10.0) *
                                    (std::log10(sigma_m / sigma_0) -
                                     std::log10(sigma_d / sigma_0));

      REQUIRE(topopt::InterpolateConductivityLogLinear(rho, sigma_d, sigma_m, sigma_0) ==
              Approx(expected_sigma).margin(1.0e-14));
      REQUIRE(topopt::InterpolateConductivityLogLinearDerivative(rho, sigma_d, sigma_m,
                                                                 sigma_0) ==
              Approx(expected_deriv).epsilon(1.0e-14));
    }
  }
}

TEST_CASE("TopOpt material batch updates", "[topopt][materialoperator][Serial]")
{
  SECTION("Chen n-squared batch update")
  {
    TwoAttributeMaterialOperatorFixture fixture;
    mfem::Array<int> attrs(2);
    attrs[0] = 1;
    attrs[1] = 2;
    mfem::Vector rho_hat(2);
    rho_hat(0) = 0.0;
    rho_hat(1) = 1.0;

    fixture.mat_op.UpdatePermittivityNSquared(attrs, rho_hat, 1.0, 3.48);

    CheckDiagonalMatrix(fixture.mat_op.GetPermittivityReal(1), 1.0);
    CheckDiagonalMatrix(fixture.mat_op.GetPermittivityReal(2), 3.48 * 3.48);
    REQUIRE(fixture.mat_op.GetLightSpeedMax(1) == Approx(1.0).margin(1.0e-12));
    REQUIRE(fixture.mat_op.GetLightSpeedMax(2) ==
            Approx(1.0 / 3.48).margin(1.0e-12));
  }

  SECTION("Aage log-linear conductivity batch update")
  {
    TwoAttributeMaterialOperatorFixture fixture;
    mfem::Array<int> attrs(2);
    attrs[0] = 1;
    attrs[1] = 2;
    mfem::Vector rho_hat(2);
    rho_hat(0) = 0.0;
    rho_hat(1) = 1.0;

    constexpr double sigma_d = 1.0e-4;
    constexpr double sigma_m = 1.0e6;
    constexpr double sigma_0 = sigma_m;
    fixture.mat_op.UpdateConductivityLogLinear(attrs, rho_hat, sigma_d, sigma_m, sigma_0);

    CheckDiagonalMatrix(fixture.mat_op.GetConductivityReal(1), sigma_d);
    CheckDiagonalMatrix(fixture.mat_op.GetConductivityReal(2), sigma_m);
  }
}

}  // namespace palace
