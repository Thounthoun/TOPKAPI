// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <sstream>
#include <vector>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fem/mesh.hpp"
#include "linalg/vector.hpp"
#include "models/spaceoperator.hpp"
#include "utils/communication.hpp"
#include "utils/iodata.hpp"
#include "utils/topopt.hpp"
#include "utils/topoptcontext.hpp"
#include "utils/units.hpp"

namespace palace
{
using namespace Catch;

namespace
{

struct GradientFixture
{
  Units units;
  IoData iodata;
  std::unique_ptr<mfem::Mesh> serial_mesh;
  std::vector<std::unique_ptr<Mesh>> mesh;
  std::unique_ptr<SpaceOperator> space_op;
  std::unique_ptr<topopt::HelmholtzFilter> filter;
  std::unique_ptr<topopt::TopOptContext> topopt_ctx;
  topopt::DesignLayout design_layout;

  GradientFixture()
    : units(1.0, 1.0), iodata(units),
      serial_mesh(std::make_unique<mfem::Mesh>(
          mfem::Mesh::MakeCartesian3D(5, 1, 1, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0)))
  {
    mfem::Array<int> design_attrs(5);
    mfem::Array<int> design_elems(5);
    for (int i = 0; i < 5; i++)
    {
      serial_mesh->SetAttribute(i, i + 1);
      design_attrs[i] = i + 1;
      design_elems[i] = i;

      auto &material = iodata.domains.materials.emplace_back();
      material.attributes = {i + 1};
      material.epsilon_r.s = {1.0, 1.0, 1.0};
      material.mu_r.s = {1.0, 1.0, 1.0};
    }
    serial_mesh->SetAttributes();
    design_layout.Set(design_elems, design_attrs);

    iodata.problem.type = ProblemType::EIGENMODE;
    iodata.solver.order = 1;
    iodata.CheckConfiguration();

    auto par_mesh = std::make_unique<mfem::ParMesh>(Mpi::World(), *serial_mesh);
    iodata.NondimensionalizeInputs(*par_mesh);
    mesh.push_back(std::make_unique<Mesh>(std::move(par_mesh)));

    space_op = std::make_unique<SpaceOperator>(iodata, mesh);
    filter = std::make_unique<topopt::HelmholtzFilter>(mesh.back()->Get(), 1, 2.0 / 5.0);
    topopt_ctx = std::make_unique<topopt::TopOptContext>(iodata, *space_op, design_layout,
                                                         *filter);
  }
};

double EvaluateObjective(GradientFixture &fixture, const mfem::Vector &rho, double beta,
                         double eta, double omega)
{
  fixture.topopt_ctx->SetBackgroundDensity(0.0);
  fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);
  return fixture.topopt_ctx->EvaluateResponseNorm(omega);
}

double EvaluateProjectedObjective(GradientFixture &fixture, const mfem::Vector &rho,
                                  const mfem::Vector &weights, double beta, double eta,
                                  double background_density = 0.0)
{
  fixture.topopt_ctx->SetBackgroundDensity(background_density);
  fixture.topopt_ctx->UpdateState(rho, beta, eta);
  return linalg::Dot(Mpi::World(), fixture.topopt_ctx->GetState().GetProjectedDensity(),
                     weights);
}

double EvaluateResponseFromProjectedDensity(GradientFixture &fixture, const mfem::Vector &rho_hat,
                                            double omega)
{
  return fixture.topopt_ctx->EvaluateResponseNormFromProjectedDensity(rho_hat, omega, 1.0,
                                                                      3.0);
}

double EvaluateSolvedObjective(GradientFixture &fixture, const mfem::Vector &rho, double beta,
                               double eta, double omega, const mfem::Vector &rhs,
                               const mfem::Vector &weights)
{
  fixture.topopt_ctx->SetBackgroundDensity(0.0);
  fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);
  return fixture.topopt_ctx->EvaluateLinearSolvedObjective(omega, rhs, weights);
}

double EvaluateStoredEnergyObjective(GradientFixture &fixture, const mfem::Vector &rho,
                                     double beta, double eta, double omega,
                                     const mfem::Vector &rhs)
{
  fixture.topopt_ctx->SetBackgroundDensity(0.0);
  fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);
  return fixture.topopt_ctx->EvaluateStoredEnergyObjective(omega, rhs);
}

double EvaluateProjectedVolumeFraction(GradientFixture &fixture, const mfem::Vector &rho,
                                       const mfem::Vector &element_volume, double beta,
                                       double eta, double background_density = 0.0)
{
  fixture.topopt_ctx->SetBackgroundDensity(background_density);
  fixture.topopt_ctx->UpdateState(rho, beta, eta);
  return fixture.topopt_ctx->EvaluateProjectedVolumeFraction(element_volume);
}

double CentralDifference(GradientFixture &fixture, const mfem::Vector &rho, int idx,
                         double delta, double beta, double eta, double omega)
{
  mfem::Vector rho_plus(rho), rho_minus(rho);
  rho_plus(idx) += delta;
  rho_minus(idx) -= delta;
  return (EvaluateObjective(fixture, rho_plus, beta, eta, omega) -
          EvaluateObjective(fixture, rho_minus, beta, eta, omega)) /
         (2.0 * delta);
}

}  // namespace

TEST_CASE("TopOpt design state pipeline", "[topopt][gradient][Serial]")
{
  GradientFixture fixture;

  SECTION("Filter and projection state is well-formed")
  {
    auto &state = fixture.topopt_ctx->GetState();
    state.Resize(5);
    state.SetUniformDensity(0.5);
    state.GetDensity()(0) = 0.0;
    state.GetDensity()(4) = 1.0;
    state.ApplyFilterAndProjection(*fixture.filter, fixture.design_layout, 8.0, 0.5);

    REQUIRE(state.GetFilteredTrueDofs().Size() == fixture.filter->GetFESpace().GetTrueVSize());
    REQUIRE(state.GetFilteredDensity().Size() == state.GetDensity().Size());
    REQUIRE(state.GetProjectedDensity().Size() == state.GetDensity().Size());

    for (int i = 0; i < state.GetProjectedDensity().Size(); i++)
    {
      REQUIRE(std::isfinite(state.GetFilteredDensity()(i)));
      REQUIRE(state.GetProjectedDensity()(i) >= 0.0);
      REQUIRE(state.GetProjectedDensity()(i) <= 1.0);
    }
  }

  SECTION("Background density influences subset-design filtering")
  {
    mfem::Array<int> subset_elems(3), subset_attrs(3);
    subset_elems[0] = 1;
    subset_elems[1] = 2;
    subset_elems[2] = 3;
    subset_attrs[0] = 2;
    subset_attrs[1] = 3;
    subset_attrs[2] = 4;
    topopt::DesignLayout subset_layout(subset_elems, subset_attrs);

    topopt::DesignState void_state(3), solid_bg_state(3);
    void_state.SetUniformDensity(0.0);
    solid_bg_state.SetUniformDensity(0.0);

    void_state.ApplyFilter(*fixture.filter, subset_layout, 0.0);
    solid_bg_state.ApplyFilter(*fixture.filter, subset_layout, 1.0);

    REQUIRE(solid_bg_state.GetFilteredDensity()(0) > void_state.GetFilteredDensity()(0));
    REQUIRE(solid_bg_state.GetFilteredDensity()(2) > void_state.GetFilteredDensity()(2));
  }
}

TEST_CASE("TopOpt FD gradient baseline", "[topopt][gradient][Serial]")
{
  GradientFixture fixture;
  constexpr double beta = 8.0;
  constexpr double eta = 0.5;
  constexpr double omega = 0.75;

  mfem::Vector rho(5);
  rho = 0.5;

  const std::array<int, 3> interior = {1, 2, 3};
  const std::array<double, 3> delta = {1.0e-3, 1.0e-4, 1.0e-5};

  SECTION("T2.5 - Central differences are finite and converge as delta shrinks")
  {
    for (int idx : interior)
    {
      const double g0 = CentralDifference(fixture, rho, idx, delta[0], beta, eta, omega);
      const double g1 = CentralDifference(fixture, rho, idx, delta[1], beta, eta, omega);
      const double g2 = CentralDifference(fixture, rho, idx, delta[2], beta, eta, omega);

      REQUIRE(std::isfinite(g0));
      REQUIRE(std::isfinite(g1));
      REQUIRE(std::isfinite(g2));
      REQUIRE(std::abs(g2) > 1.0e-8);

      const double err_coarse = std::abs(g0 - g2);
      const double err_medium = std::abs(g1 - g2);
      REQUIRE(err_medium <= err_coarse + 1.0e-8);
    }
  }

  SECTION("T2.6 - Filter/projection adjoint matches FD")
  {
    mfem::Vector weights(5);
    for (int i = 0; i < weights.Size(); i++)
    {
      weights(i) = 1.0 + i;
    }

    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdateState(rho, beta, eta);

    mfem::Vector adj_grad;
    fixture.topopt_ctx->BackpropagateFilterProjection(weights, beta, eta, adj_grad);

    constexpr double delta_fd = 1.0e-5;
    for (int idx : interior)
    {
      mfem::Vector rho_plus(rho), rho_minus(rho);
      rho_plus(idx) += delta_fd;
      rho_minus(idx) -= delta_fd;

      const double fd =
          (EvaluateProjectedObjective(fixture, rho_plus, weights, beta, eta) -
           EvaluateProjectedObjective(fixture, rho_minus, weights, beta, eta)) /
          (2.0 * delta_fd);

      REQUIRE(std::isfinite(fd));
      REQUIRE(std::isfinite(adj_grad(idx)));
      REQUIRE(fd == Approx(adj_grad(idx)).epsilon(1.0e-3));
    }
  }

  SECTION("Proxy response chain rule matches full FD")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdateState(rho, beta, eta);
    const mfem::Vector rho_hat_base = fixture.topopt_ctx->GetState().GetProjectedDensity();

    mfem::Vector upstream(rho_hat_base.Size());
    constexpr double delta_hat = 1.0e-5;
    for (int i = 0; i < upstream.Size(); i++)
    {
      mfem::Vector rho_hat_plus(rho_hat_base), rho_hat_minus(rho_hat_base);
      rho_hat_plus(i) += delta_hat;
      rho_hat_minus(i) -= delta_hat;
      upstream(i) = (EvaluateResponseFromProjectedDensity(fixture, rho_hat_plus, omega) -
                     EvaluateResponseFromProjectedDensity(fixture, rho_hat_minus, omega)) /
                    (2.0 * delta_hat);
      REQUIRE(std::isfinite(upstream(i)));
    }

    mfem::Vector chain_grad;
    fixture.topopt_ctx->BackpropagateFilterProjection(upstream, beta, eta, chain_grad);

    constexpr double delta_fd = 1.0e-5;
    for (int idx : interior)
    {
      const double fd = CentralDifference(fixture, rho, idx, delta_fd, beta, eta, omega);
      REQUIRE(std::isfinite(fd));
      REQUIRE(std::isfinite(chain_grad(idx)));
      REQUIRE(fd == Approx(chain_grad(idx)).epsilon(2.0e-2));
    }
  }

  SECTION("Analytic projected response upstream matches projected FD")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdateState(rho, beta, eta);
    const mfem::Vector rho_hat_base = fixture.topopt_ctx->GetState().GetProjectedDensity();

    mfem::Vector analytic;
    fixture.topopt_ctx->ComputeResponseNormProjectedGradient(omega, 1.0, 3.0, analytic);

    constexpr double delta_hat = 1.0e-5;
    for (int i = 0; i < analytic.Size(); i++)
    {
      mfem::Vector rho_hat_plus(rho_hat_base), rho_hat_minus(rho_hat_base);
      rho_hat_plus(i) += delta_hat;
      rho_hat_minus(i) -= delta_hat;
      const double fd =
          (EvaluateResponseFromProjectedDensity(fixture, rho_hat_plus, omega) -
           EvaluateResponseFromProjectedDensity(fixture, rho_hat_minus, omega)) /
          (2.0 * delta_hat);

      REQUIRE(std::isfinite(fd));
      REQUIRE(std::isfinite(analytic(i)));
      REQUIRE(analytic(i) == Approx(fd).epsilon(1.0e-4));
    }
  }

  SECTION("Response proxy density gradient matches full FD")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);

    mfem::Vector chain_grad;
    fixture.topopt_ctx->ComputeResponseNormDensityGradient(omega, beta, eta, 1.0, 3.0,
                                                           chain_grad);

    constexpr double delta_fd = 1.0e-5;
    for (int idx : interior)
    {
      const double fd = CentralDifference(fixture, rho, idx, delta_fd, beta, eta, omega);
      REQUIRE(std::isfinite(fd));
      REQUIRE(std::isfinite(chain_grad(idx)));
      REQUIRE(fd == Approx(chain_grad(idx)).epsilon(2.0e-2));
    }
  }

  SECTION("Solved-state proxy density gradient matches full FD")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);

    mfem::Vector rhs(fixture.space_op->GetNDSpace().GetTrueVSize());
    mfem::Vector weights(fixture.space_op->GetNDSpace().GetTrueVSize());
    for (int i = 0; i < rhs.Size(); i++)
    {
      rhs(i) = 1.0;
      weights(i) = 1.0 + 0.1 * i;
    }

    mfem::Vector chain_grad;
    fixture.topopt_ctx->ComputeLinearSolvedDensityGradient(0.30, rhs, weights, beta, eta, 1.0,
                                                           3.0, chain_grad);

    constexpr double delta_fd = 1.0e-5;
    for (int idx : interior)
    {
      mfem::Vector rho_plus(rho), rho_minus(rho);
      rho_plus(idx) += delta_fd;
      rho_minus(idx) -= delta_fd;
      const double fd =
          (EvaluateSolvedObjective(fixture, rho_plus, beta, eta, 0.30, rhs, weights) -
           EvaluateSolvedObjective(fixture, rho_minus, beta, eta, 0.30, rhs, weights)) /
          (2.0 * delta_fd);

      REQUIRE(std::isfinite(fd));
      REQUIRE(std::isfinite(chain_grad(idx)));
      REQUIRE(fd == Approx(chain_grad(idx)).epsilon(3.0e-2));
    }
  }

  SECTION("Analytic projected solved-state upstream matches projected FD")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdateState(rho, beta, eta);
    const mfem::Vector rho_hat_base = fixture.topopt_ctx->GetState().GetProjectedDensity();

    mfem::Vector rhs(fixture.space_op->GetNDSpace().GetTrueVSize());
    mfem::Vector weights(fixture.space_op->GetNDSpace().GetTrueVSize());
    for (int i = 0; i < rhs.Size(); i++)
    {
      rhs(i) = 1.0;
      weights(i) = 1.0 + 0.1 * i;
    }

    mfem::Vector analytic;
    fixture.topopt_ctx->ComputeLinearSolvedProjectedGradient(0.30, rhs, weights, 1.0, 3.0,
                                                             analytic);

    constexpr double delta_hat = 1.0e-5;
    for (int i = 0; i < analytic.Size(); i++)
    {
      mfem::Vector rho_hat_plus(rho_hat_base), rho_hat_minus(rho_hat_base);
      rho_hat_plus(i) += delta_hat;
      rho_hat_minus(i) -= delta_hat;
      const double fd =
          (fixture.topopt_ctx->EvaluateLinearSolvedObjectiveFromProjectedDensity(
               rho_hat_plus, 0.30, rhs, weights, 1.0, 3.0) -
           fixture.topopt_ctx->EvaluateLinearSolvedObjectiveFromProjectedDensity(
               rho_hat_minus, 0.30, rhs, weights, 1.0, 3.0)) /
          (2.0 * delta_hat);

      REQUIRE(std::isfinite(fd));
      REQUIRE(std::isfinite(analytic(i)));
      REQUIRE(analytic(i) == Approx(fd).epsilon(1.0e-4));
    }
  }

  SECTION("Analytic projected stored-energy upstream matches projected FD")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdateState(rho, beta, eta);
    const mfem::Vector rho_hat_base = fixture.topopt_ctx->GetState().GetProjectedDensity();

    mfem::Vector rhs(fixture.space_op->GetNDSpace().GetTrueVSize());
    rhs = 1.0;

    mfem::Vector analytic;
    fixture.topopt_ctx->ComputeStoredEnergyProjectedGradient(0.30, rhs, 1.0, 3.0, analytic);

    constexpr double delta_hat = 1.0e-5;
    for (int i = 0; i < analytic.Size(); i++)
    {
      mfem::Vector rho_hat_plus(rho_hat_base), rho_hat_minus(rho_hat_base);
      rho_hat_plus(i) += delta_hat;
      rho_hat_minus(i) -= delta_hat;
      const double fd =
          (fixture.topopt_ctx->EvaluateStoredEnergyObjectiveFromProjectedDensity(
               rho_hat_plus, 0.30, rhs, 1.0, 3.0) -
           fixture.topopt_ctx->EvaluateStoredEnergyObjectiveFromProjectedDensity(
               rho_hat_minus, 0.30, rhs, 1.0, 3.0)) /
          (2.0 * delta_hat);

      REQUIRE(std::isfinite(fd));
      REQUIRE(std::isfinite(analytic(i)));
      REQUIRE(analytic(i) == Approx(fd).epsilon(1.0e-4));
    }
  }

  SECTION("Stored-energy density gradient matches full FD")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);

    mfem::Vector rhs(fixture.space_op->GetNDSpace().GetTrueVSize());
    rhs = 1.0;

    mfem::Vector chain_grad;
    fixture.topopt_ctx->ComputeStoredEnergyDensityGradient(0.30, rhs, beta, eta, 1.0, 3.0,
                                                           chain_grad);

    constexpr double delta_fd = 1.0e-5;
    for (int idx : interior)
    {
      mfem::Vector rho_plus(rho), rho_minus(rho);
      rho_plus(idx) += delta_fd;
      rho_minus(idx) -= delta_fd;
      const double fd =
          (EvaluateStoredEnergyObjective(fixture, rho_plus, beta, eta, 0.30, rhs) -
           EvaluateStoredEnergyObjective(fixture, rho_minus, beta, eta, 0.30, rhs)) /
          (2.0 * delta_fd);

      REQUIRE(std::isfinite(fd));
      REQUIRE(std::isfinite(chain_grad(idx)));
      REQUIRE(fd == Approx(chain_grad(idx)).epsilon(3.0e-2));
    }
  }

  SECTION("Projected mass sensitivity kernel reproduces response and solved-state upstreams")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdateState(rho, beta, eta);
    fixture.topopt_ctx->UpdatePermittivityFromProjectedDensity(
        fixture.topopt_ctx->GetState().GetProjectedDensity(), 1.0, 3.0);

    auto K = fixture.space_op->GetStiffnessMatrix<Operator>(Operator::DIAG_ZERO);
    auto M = fixture.space_op->GetMassMatrix<Operator>(Operator::DIAG_ZERO);
    auto A = fixture.space_op->GetSystemMatrix<Operator, double>(1.0, 0.0, -omega * omega,
                                                                 K.get(), nullptr, M.get());

    Vector x(fixture.space_op->GetNDSpace().GetTrueVSize());
    Vector y(fixture.space_op->GetNDSpace().GetTrueVSize());
    x = 1.0;
    y = 0.0;
    A->Mult(x, y);
    const double j = linalg::Norml2(fixture.space_op->GetComm(), y);
    REQUIRE(std::isfinite(j));
    REQUIRE(j > 1.0e-14);

    Vector g(y);
    g *= 1.0 / j;

    mfem::Vector response_kernel, response_upstream;
    fixture.topopt_ctx->ComputeProjectedMassSensitivity(x, g, -omega * omega, 1.0, 3.0,
                                                        response_kernel);
    fixture.topopt_ctx->ComputeResponseNormProjectedGradient(omega, 1.0, 3.0,
                                                             response_upstream);

    REQUIRE(response_kernel.Size() == response_upstream.Size());
    for (int i = 0; i < response_kernel.Size(); i++)
    {
      REQUIRE(response_kernel(i) ==
              Approx(response_upstream(i)).epsilon(1.0e-12));
    }

    mfem::Vector rhs(fixture.space_op->GetNDSpace().GetTrueVSize());
    mfem::Vector weights(fixture.space_op->GetNDSpace().GetTrueVSize());
    for (int i = 0; i < rhs.Size(); i++)
    {
      rhs(i) = 1.0;
      weights(i) = 1.0 + 0.1 * i;
    }

    Vector u, lambda;
    fixture.topopt_ctx->SolveProxyState(0.30, rhs, u);
    fixture.topopt_ctx->SolveProxyState(0.30, weights, lambda);

    mfem::Vector solved_kernel, solved_upstream;
    fixture.topopt_ctx->ComputeProjectedMassSensitivity(u, lambda, 0.30 * 0.30, 1.0, 3.0,
                                                        solved_kernel);
    fixture.topopt_ctx->ComputeLinearSolvedProjectedGradient(0.30, rhs, weights, 1.0, 3.0,
                                                             solved_upstream);

    REQUIRE(solved_kernel.Size() == solved_upstream.Size());
    for (int i = 0; i < solved_kernel.Size(); i++)
    {
      REQUIRE(solved_kernel(i) ==
              Approx(solved_upstream(i)).epsilon(1.0e-12));
    }
  }

  SECTION("High-level density gradient matches explicit chain")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);

    mfem::Vector manual_upstream, manual_grad, wrapped_grad;
    fixture.topopt_ctx->ComputeResponseNormProjectedGradient(omega, 1.0, 3.0, manual_upstream);
    fixture.topopt_ctx->BackpropagateFilterProjection(manual_upstream, beta, eta, manual_grad);
    fixture.topopt_ctx->ComputeResponseNormDensityGradient(omega, beta, eta, 1.0, 3.0,
                                                           wrapped_grad);

    REQUIRE(manual_grad.Size() == wrapped_grad.Size());
    for (int i = 0; i < manual_grad.Size(); i++)
    {
      REQUIRE(manual_grad(i) == Approx(wrapped_grad(i)).epsilon(1.0e-12));
    }
  }

  SECTION("External state-adjoint density bridge reproduces wrapped gradients")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);

    auto K = fixture.space_op->GetStiffnessMatrix<Operator>(Operator::DIAG_ZERO);
    auto M = fixture.space_op->GetMassMatrix<Operator>(Operator::DIAG_ZERO);
    auto A = fixture.space_op->GetSystemMatrix<Operator, double>(1.0, 0.0, -omega * omega,
                                                                 K.get(), nullptr, M.get());

    Vector x(fixture.space_op->GetNDSpace().GetTrueVSize());
    Vector y(fixture.space_op->GetNDSpace().GetTrueVSize());
    x = 1.0;
    y = 0.0;
    A->Mult(x, y);
    const double j = linalg::Norml2(fixture.space_op->GetComm(), y);
    REQUIRE(std::isfinite(j));
    REQUIRE(j > 1.0e-14);

    Vector g(y);
    g *= 1.0 / j;

    mfem::Vector response_external, response_wrapped;
    fixture.topopt_ctx->ComputeDensityGradientFromStateAdjoint(x, g, -omega * omega, beta, eta,
                                                               1.0, 3.0,
                                                               response_external);
    fixture.topopt_ctx->ComputeResponseNormDensityGradient(omega, beta, eta, 1.0, 3.0,
                                                           response_wrapped);

    REQUIRE(response_external.Size() == response_wrapped.Size());
    for (int i = 0; i < response_external.Size(); i++)
    {
      REQUIRE(response_external(i) ==
              Approx(response_wrapped(i)).epsilon(1.0e-12));
    }

    mfem::Vector rhs(fixture.space_op->GetNDSpace().GetTrueVSize());
    mfem::Vector weights(fixture.space_op->GetNDSpace().GetTrueVSize());
    for (int i = 0; i < rhs.Size(); i++)
    {
      rhs(i) = 1.0;
      weights(i) = 1.0 + 0.1 * i;
    }

    Vector u, lambda;
    fixture.topopt_ctx->SolveProxyState(0.30, rhs, u);
    fixture.topopt_ctx->SolveProxyState(0.30, weights, lambda);

    mfem::Vector solved_external, solved_wrapped;
    fixture.topopt_ctx->ComputeDensityGradientFromStateAdjoint(u, lambda, 0.30 * 0.30, beta,
                                                               eta, 1.0, 3.0,
                                                               solved_external);
    fixture.topopt_ctx->ComputeLinearSolvedDensityGradient(0.30, rhs, weights, beta, eta, 1.0,
                                                           3.0, solved_wrapped);

    REQUIRE(solved_external.Size() == solved_wrapped.Size());
    for (int i = 0; i < solved_external.Size(); i++)
    {
      REQUIRE(solved_external(i) ==
              Approx(solved_wrapped(i)).epsilon(1.0e-12));
    }
  }

  SECTION("Generic mass gradient bridge reproduces wrapped projected and density paths")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);

    auto K = fixture.space_op->GetStiffnessMatrix<Operator>(Operator::DIAG_ZERO);
    auto M = fixture.space_op->GetMassMatrix<Operator>(Operator::DIAG_ZERO);
    auto A = fixture.space_op->GetSystemMatrix<Operator, double>(1.0, 0.0, -omega * omega,
                                                                 K.get(), nullptr, M.get());

    Vector x(fixture.space_op->GetNDSpace().GetTrueVSize());
    Vector y(fixture.space_op->GetNDSpace().GetTrueVSize());
    x = 1.0;
    y = 0.0;
    A->Mult(x, y);
    const double j = linalg::Norml2(fixture.space_op->GetComm(), y);
    REQUIRE(std::isfinite(j));
    REQUIRE(j > 1.0e-14);

    Vector g(y);
    g *= 1.0 / j;

    mfem::Vector response_proj_generic, response_proj_wrapped;
    fixture.topopt_ctx->ComputeProjectedMassGradientFromStateAdjoint(
        x, g, 0.0, -omega * omega, 1.0, 3.0, response_proj_generic);
    fixture.topopt_ctx->ComputeResponseNormProjectedGradient(omega, 1.0, 3.0,
                                                             response_proj_wrapped);

    REQUIRE(response_proj_generic.Size() == response_proj_wrapped.Size());
    for (int i = 0; i < response_proj_generic.Size(); i++)
    {
      REQUIRE(response_proj_generic(i) ==
              Approx(response_proj_wrapped(i)).epsilon(1.0e-12));
    }

    mfem::Vector response_density_generic, response_density_wrapped;
    fixture.topopt_ctx->ComputeDensityMassGradientFromStateAdjoint(
        x, g, 0.0, -omega * omega, beta, eta, 1.0, 3.0, response_density_generic);
    fixture.topopt_ctx->ComputeResponseNormDensityGradient(omega, beta, eta, 1.0, 3.0,
                                                           response_density_wrapped);

    REQUIRE(response_density_generic.Size() == response_density_wrapped.Size());
    for (int i = 0; i < response_density_generic.Size(); i++)
    {
      REQUIRE(response_density_generic(i) ==
              Approx(response_density_wrapped(i)).epsilon(1.0e-12));
    }

    mfem::Vector rhs(fixture.space_op->GetNDSpace().GetTrueVSize());
    mfem::Vector weights(fixture.space_op->GetNDSpace().GetTrueVSize());
    for (int i = 0; i < rhs.Size(); i++)
    {
      rhs(i) = 1.0;
      weights(i) = 1.0 + 0.1 * i;
    }

    Vector u, lambda_linear;
    fixture.topopt_ctx->SolveProxyState(0.30, rhs, u);
    fixture.topopt_ctx->SolveProxyState(0.30, weights, lambda_linear);

    mfem::Vector solved_proj_generic, solved_proj_wrapped;
    fixture.topopt_ctx->ComputeProjectedMassGradientFromStateAdjoint(
        u, lambda_linear, 0.0, 0.30 * 0.30, 1.0, 3.0, solved_proj_generic);
    fixture.topopt_ctx->ComputeLinearSolvedProjectedGradient(0.30, rhs, weights, 1.0, 3.0,
                                                             solved_proj_wrapped);

    REQUIRE(solved_proj_generic.Size() == solved_proj_wrapped.Size());
    for (int i = 0; i < solved_proj_generic.Size(); i++)
    {
      REQUIRE(solved_proj_generic(i) ==
              Approx(solved_proj_wrapped(i)).epsilon(1.0e-12));
    }

    Vector Mu, lambda_energy;
    Mu.SetSize(u.Size());
    Mu = 0.0;
    M->Mult(u, Mu);
    fixture.topopt_ctx->SolveProxyState(0.30, Mu, lambda_energy);

    mfem::Vector energy_proj_generic, energy_proj_wrapped;
    fixture.topopt_ctx->ComputeProjectedMassGradientFromStateAdjoint(
        u, lambda_energy, 0.5, 0.30 * 0.30, 1.0, 3.0, energy_proj_generic);
    fixture.topopt_ctx->ComputeStoredEnergyProjectedGradientFromState(u, 0.30, 1.0, 3.0,
                                                                      energy_proj_wrapped);

    REQUIRE(energy_proj_generic.Size() == energy_proj_wrapped.Size());
    for (int i = 0; i < energy_proj_generic.Size(); i++)
    {
      REQUIRE(energy_proj_generic(i) ==
              Approx(energy_proj_wrapped(i)).epsilon(1.0e-12));
    }

    mfem::Vector energy_density_generic, energy_density_wrapped;
    fixture.topopt_ctx->ComputeDensityMassGradientFromStateAdjoint(
        u, lambda_energy, 0.5, 0.30 * 0.30, beta, eta, 1.0, 3.0, energy_density_generic);
    fixture.topopt_ctx->ComputeStoredEnergyDensityGradientFromState(u, 0.30, beta, eta, 1.0,
                                                                    3.0,
                                                                    energy_density_wrapped);

    REQUIRE(energy_density_generic.Size() == energy_density_wrapped.Size());
    for (int i = 0; i < energy_density_generic.Size(); i++)
    {
      REQUIRE(energy_density_generic(i) ==
              Approx(energy_density_wrapped(i)).epsilon(1.0e-12));
    }
  }

  SECTION("External forward-state stored-energy bridge reproduces wrapped gradients")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);

    mfem::Vector rhs(fixture.space_op->GetNDSpace().GetTrueVSize());
    rhs = 1.0;

    Vector u;
    fixture.topopt_ctx->SolveProxyState(0.30, rhs, u);

    mfem::Vector projected_external, projected_wrapped;
    fixture.topopt_ctx->ComputeStoredEnergyProjectedGradientFromState(u, 0.30, 1.0, 3.0,
                                                                      projected_external);
    fixture.topopt_ctx->ComputeStoredEnergyProjectedGradient(0.30, rhs, 1.0, 3.0,
                                                             projected_wrapped);

    REQUIRE(projected_external.Size() == projected_wrapped.Size());
    for (int i = 0; i < projected_external.Size(); i++)
    {
      REQUIRE(projected_external(i) ==
              Approx(projected_wrapped(i)).epsilon(1.0e-12));
    }

    mfem::Vector density_external, density_wrapped;
    fixture.topopt_ctx->ComputeStoredEnergyDensityGradientFromState(u, 0.30, beta, eta, 1.0,
                                                                    3.0,
                                                                    density_external);
    fixture.topopt_ctx->ComputeStoredEnergyDensityGradient(0.30, rhs, beta, eta, 1.0, 3.0,
                                                           density_wrapped);

    REQUIRE(density_external.Size() == density_wrapped.Size());
    for (int i = 0; i < density_external.Size(); i++)
    {
      REQUIRE(density_external(i) ==
              Approx(density_wrapped(i)).epsilon(1.0e-12));
    }
  }

  SECTION("External forward-state mass objective reproduces wrapped stored-energy objective")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);

    mfem::Vector rhs(fixture.space_op->GetNDSpace().GetTrueVSize());
    rhs = 1.0;

    Vector u;
    fixture.topopt_ctx->SolveProxyState(0.30, rhs, u);

    const double wrapped = fixture.topopt_ctx->EvaluateStoredEnergyObjective(0.30, rhs);
    const double from_state = fixture.topopt_ctx->EvaluateStoredEnergyObjectiveFromState(u);
    const double from_generic =
        fixture.topopt_ctx->EvaluateMassObjectiveFromStates(u, u, 0.5);

    REQUIRE(std::isfinite(wrapped));
    REQUIRE(std::isfinite(from_state));
    REQUIRE(std::isfinite(from_generic));
    REQUIRE(from_state == Approx(wrapped).epsilon(1.0e-12));
    REQUIRE(from_generic == Approx(wrapped).epsilon(1.0e-12));
  }

  SECTION("Combined external mass objective and gradient reproduces stored-energy path")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);

    mfem::Vector rhs(fixture.space_op->GetNDSpace().GetTrueVSize());
    rhs = 1.0;

    Vector u, Mu, lambda_energy;
    fixture.topopt_ctx->SolveProxyState(0.30, rhs, u);
    Mu.SetSize(u.Size());
    Mu = 0.0;

    auto M = fixture.space_op->GetMassMatrix<Operator>(Operator::DIAG_ZERO);
    M->Mult(u, Mu);
    fixture.topopt_ctx->SolveProxyState(0.30, Mu, lambda_energy);

    double projected_objective = 0.0;
    mfem::Vector projected_generic, projected_wrapped;
    fixture.topopt_ctx->EvaluateMassObjectiveAndProjectedGradientFromStates(
        u, u, u, lambda_energy, 0.5, 0.5, 0.30 * 0.30, 1.0, 3.0, projected_objective,
        projected_generic);
    fixture.topopt_ctx->ComputeStoredEnergyProjectedGradientFromState(u, 0.30, 1.0, 3.0,
                                                                      projected_wrapped);

    const double wrapped_objective = fixture.topopt_ctx->EvaluateStoredEnergyObjectiveFromState(u);
    REQUIRE(std::isfinite(projected_objective));
    REQUIRE(projected_objective == Approx(wrapped_objective).epsilon(1.0e-12));
    REQUIRE(projected_generic.Size() == projected_wrapped.Size());
    for (int i = 0; i < projected_generic.Size(); i++)
    {
      REQUIRE(projected_generic(i) ==
              Approx(projected_wrapped(i)).epsilon(1.0e-12));
    }

    double density_objective = 0.0;
    mfem::Vector density_generic, density_wrapped;
    fixture.topopt_ctx->EvaluateMassObjectiveAndDensityGradientFromStates(
        u, u, u, lambda_energy, 0.5, 0.5, 0.30 * 0.30, beta, eta, 1.0, 3.0,
        density_objective, density_generic);
    fixture.topopt_ctx->ComputeStoredEnergyDensityGradientFromState(u, 0.30, beta, eta, 1.0,
                                                                    3.0,
                                                                    density_wrapped);

    REQUIRE(std::isfinite(density_objective));
    REQUIRE(density_objective == Approx(wrapped_objective).epsilon(1.0e-12));
    REQUIRE(density_generic.Size() == density_wrapped.Size());
    for (int i = 0; i < density_generic.Size(); i++)
    {
      REQUIRE(density_generic(i) ==
              Approx(density_wrapped(i)).epsilon(1.0e-12));
    }
  }

  SECTION("Mass adjoint RHS builder reproduces explicit M*u and stored-energy bridge")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);

    mfem::Vector rhs(fixture.space_op->GetNDSpace().GetTrueVSize());
    rhs = 1.0;

    Vector u, Mu_explicit, Mu_built, lambda_energy;
    fixture.topopt_ctx->SolveProxyState(0.30, rhs, u);

    auto M = fixture.space_op->GetMassMatrix<Operator>(Operator::DIAG_ZERO);
    Mu_explicit.SetSize(u.Size());
    Mu_explicit = 0.0;
    M->Mult(u, Mu_explicit);

    fixture.topopt_ctx->ComputeMassAdjointRHSFromState(u, 1.0, Mu_built);
    REQUIRE(Mu_built.Size() == Mu_explicit.Size());
    for (int i = 0; i < Mu_built.Size(); i++)
    {
      REQUIRE(Mu_built(i) == Approx(Mu_explicit(i)).epsilon(1.0e-12));
    }

    fixture.topopt_ctx->SolveProxyState(0.30, Mu_built, lambda_energy);

    double objective_from_bridge = 0.0;
    mfem::Vector gradient_from_bridge, wrapped_gradient;
    fixture.topopt_ctx->EvaluateMassObjectiveAndDensityGradientFromStates(
        u, u, u, lambda_energy, 0.5, 0.5, 0.30 * 0.30, beta, eta, 1.0, 3.0,
        objective_from_bridge, gradient_from_bridge);
    fixture.topopt_ctx->ComputeStoredEnergyDensityGradientFromState(u, 0.30, beta, eta, 1.0,
                                                                    3.0,
                                                                    wrapped_gradient);

    const double wrapped_objective = fixture.topopt_ctx->EvaluateStoredEnergyObjectiveFromState(u);
    REQUIRE(std::isfinite(objective_from_bridge));
    REQUIRE(objective_from_bridge == Approx(wrapped_objective).epsilon(1.0e-12));
    REQUIRE(gradient_from_bridge.Size() == wrapped_gradient.Size());
    for (int i = 0; i < gradient_from_bridge.Size(); i++)
    {
      REQUIRE(gradient_from_bridge(i) ==
              Approx(wrapped_gradient(i)).epsilon(1.0e-12));
    }
  }

  SECTION("Stored-energy forward-state API reproduces wrapped objective and gradients")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);

    mfem::Vector rhs(fixture.space_op->GetNDSpace().GetTrueVSize());
    rhs = 1.0;

    Vector u, lambda_explicit;
    fixture.topopt_ctx->SolveProxyState(0.30, rhs, u);
    fixture.topopt_ctx->ComputeStoredEnergyAdjointFromState(u, 0.30, lambda_explicit);

    Vector Mu;
    Mu.SetSize(u.Size());
    Mu = 0.0;
    auto M = fixture.space_op->GetMassMatrix<Operator>(Operator::DIAG_ZERO);
    M->Mult(u, Mu);

    Vector lambda_manual;
    fixture.topopt_ctx->SolveProxyState(0.30, Mu, lambda_manual);
    REQUIRE(lambda_explicit.Size() == lambda_manual.Size());
    for (int i = 0; i < lambda_explicit.Size(); i++)
    {
      REQUIRE(lambda_explicit(i) == Approx(lambda_manual(i)).epsilon(1.0e-12));
    }

    double projected_objective = 0.0;
    mfem::Vector projected_api, projected_wrapped;
    fixture.topopt_ctx->EvaluateStoredEnergyObjectiveAndProjectedGradientFromState(
        u, 0.30, 1.0, 3.0, projected_objective, projected_api);
    fixture.topopt_ctx->ComputeStoredEnergyProjectedGradientFromState(u, 0.30, 1.0, 3.0,
                                                                      projected_wrapped);

    const double wrapped_objective = fixture.topopt_ctx->EvaluateStoredEnergyObjectiveFromState(u);
    REQUIRE(std::isfinite(projected_objective));
    REQUIRE(projected_objective == Approx(wrapped_objective).epsilon(1.0e-12));
    REQUIRE(projected_api.Size() == projected_wrapped.Size());
    for (int i = 0; i < projected_api.Size(); i++)
    {
      REQUIRE(projected_api(i) ==
              Approx(projected_wrapped(i)).epsilon(1.0e-12));
    }

    double density_objective = 0.0;
    mfem::Vector density_api, density_wrapped;
    fixture.topopt_ctx->EvaluateStoredEnergyObjectiveAndDensityGradientFromState(
        u, 0.30, beta, eta, 1.0, 3.0, density_objective, density_api);
    fixture.topopt_ctx->ComputeStoredEnergyDensityGradientFromState(u, 0.30, beta, eta, 1.0,
                                                                    3.0,
                                                                    density_wrapped);

    REQUIRE(std::isfinite(density_objective));
    REQUIRE(density_objective == Approx(wrapped_objective).epsilon(1.0e-12));
    REQUIRE(density_api.Size() == density_wrapped.Size());
    for (int i = 0; i < density_api.Size(); i++)
    {
      REQUIRE(density_api(i) ==
              Approx(density_wrapped(i)).epsilon(1.0e-12));
    }
  }

  SECTION("Linear solved forward-state API reproduces wrapped objective and gradients")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);

    mfem::Vector rhs(fixture.space_op->GetNDSpace().GetTrueVSize());
    mfem::Vector weights(fixture.space_op->GetNDSpace().GetTrueVSize());
    for (int i = 0; i < rhs.Size(); i++)
    {
      rhs(i) = 1.0;
      weights(i) = 1.0 + 0.1 * i;
    }

    Vector u, lambda_api, lambda_manual;
    fixture.topopt_ctx->SolveProxyState(0.30, rhs, u);
    fixture.topopt_ctx->ComputeLinearSolvedAdjointFromState(weights, 0.30, lambda_api);
    fixture.topopt_ctx->SolveProxyState(0.30, weights, lambda_manual);

    REQUIRE(lambda_api.Size() == lambda_manual.Size());
    for (int i = 0; i < lambda_api.Size(); i++)
    {
      REQUIRE(lambda_api(i) == Approx(lambda_manual(i)).epsilon(1.0e-12));
    }

    const double wrapped_objective =
        fixture.topopt_ctx->EvaluateLinearSolvedObjectiveFromState(u, weights);

    double projected_objective = 0.0;
    mfem::Vector projected_api, projected_wrapped;
    fixture.topopt_ctx->EvaluateLinearSolvedObjectiveAndProjectedGradientFromState(
        u, weights, 0.30, 1.0, 3.0, projected_objective, projected_api);
    fixture.topopt_ctx->ComputeLinearSolvedProjectedGradient(0.30, rhs, weights, 1.0, 3.0,
                                                             projected_wrapped);

    REQUIRE(std::isfinite(projected_objective));
    REQUIRE(projected_objective == Approx(wrapped_objective).epsilon(1.0e-12));
    REQUIRE(projected_api.Size() == projected_wrapped.Size());
    for (int i = 0; i < projected_api.Size(); i++)
    {
      REQUIRE(projected_api(i) ==
              Approx(projected_wrapped(i)).epsilon(1.0e-12));
    }

    double density_objective = 0.0;
    mfem::Vector density_api, density_wrapped;
    fixture.topopt_ctx->EvaluateLinearSolvedObjectiveAndDensityGradientFromState(
        u, weights, 0.30, beta, eta, 1.0, 3.0, density_objective, density_api);
    fixture.topopt_ctx->ComputeLinearSolvedDensityGradient(0.30, rhs, weights, beta, eta, 1.0,
                                                           3.0, density_wrapped);

    REQUIRE(std::isfinite(density_objective));
    REQUIRE(density_objective == Approx(wrapped_objective).epsilon(1.0e-12));
    REQUIRE(density_api.Size() == density_wrapped.Size());
    for (int i = 0; i < density_api.Size(); i++)
    {
      REQUIRE(density_api(i) ==
              Approx(density_wrapped(i)).epsilon(1.0e-12));
    }
  }

  SECTION("Response forward-state API reproduces wrapped objective and gradients")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);

    Vector x(fixture.space_op->GetNDSpace().GetTrueVSize());
    x = 1.0;

    Vector residual_api, residual_manual, adjoint_api, adjoint_manual;
    fixture.topopt_ctx->ComputeResponseResidualFromState(x, omega, residual_api);

    auto K = fixture.space_op->GetStiffnessMatrix<Operator>(Operator::DIAG_ZERO);
    auto M = fixture.space_op->GetMassMatrix<Operator>(Operator::DIAG_ZERO);
    auto A = fixture.space_op->GetSystemMatrix<Operator, double>(1.0, 0.0, -omega * omega,
                                                                 K.get(), nullptr, M.get());
    residual_manual.SetSize(x.Size());
    residual_manual = 0.0;
    A->Mult(x, residual_manual);

    REQUIRE(residual_api.Size() == residual_manual.Size());
    for (int i = 0; i < residual_api.Size(); i++)
    {
      REQUIRE(residual_api(i) == Approx(residual_manual(i)).epsilon(1.0e-12));
    }

    const double objective_manual = linalg::Norml2(fixture.space_op->GetComm(), residual_manual);
    const double objective_from_residual =
        fixture.topopt_ctx->ComputeResponseNormAdjointFromResidual(residual_api, adjoint_api);
    adjoint_manual = residual_manual;
    adjoint_manual *= 1.0 / objective_manual;

    REQUIRE(std::isfinite(objective_from_residual));
    REQUIRE(objective_from_residual == Approx(objective_manual).epsilon(1.0e-12));
    REQUIRE(adjoint_api.Size() == adjoint_manual.Size());
    for (int i = 0; i < adjoint_api.Size(); i++)
    {
      REQUIRE(adjoint_api(i) == Approx(adjoint_manual(i)).epsilon(1.0e-12));
    }

    double projected_objective = 0.0;
    mfem::Vector projected_api, projected_wrapped;
    fixture.topopt_ctx->EvaluateResponseNormObjectiveAndProjectedGradientFromState(
        x, omega, 1.0, 3.0, projected_objective, projected_api);
    fixture.topopt_ctx->ComputeResponseNormProjectedGradient(omega, 1.0, 3.0,
                                                             projected_wrapped);

    const double wrapped_objective = fixture.topopt_ctx->EvaluateResponseNorm(omega);
    REQUIRE(std::isfinite(projected_objective));
    REQUIRE(projected_objective == Approx(wrapped_objective).epsilon(1.0e-12));
    REQUIRE(projected_api.Size() == projected_wrapped.Size());
    for (int i = 0; i < projected_api.Size(); i++)
    {
      REQUIRE(projected_api(i) ==
              Approx(projected_wrapped(i)).epsilon(1.0e-12));
    }

    double density_objective = 0.0;
    mfem::Vector density_api, density_wrapped;
    fixture.topopt_ctx->EvaluateResponseNormObjectiveAndDensityGradientFromState(
        x, omega, beta, eta, 1.0, 3.0, density_objective, density_api);
    fixture.topopt_ctx->ComputeResponseNormDensityGradient(omega, beta, eta, 1.0, 3.0,
                                                           density_wrapped);

    REQUIRE(std::isfinite(density_objective));
    REQUIRE(density_objective == Approx(wrapped_objective).epsilon(1.0e-12));
    REQUIRE(density_api.Size() == density_wrapped.Size());
    for (int i = 0; i < density_api.Size(); i++)
    {
      REQUIRE(density_api(i) ==
              Approx(density_wrapped(i)).epsilon(1.0e-12));
    }
  }

  SECTION("Explicit current-system APIs reproduce wrapped apply and solve paths")
  {
    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);

    Vector x(fixture.space_op->GetNDSpace().GetTrueVSize());
    x = 1.0;

    Vector applied_api, applied_wrapped;
    fixture.topopt_ctx->ApplyCurrentSystemMatrix(omega, x, applied_api);
    fixture.topopt_ctx->ComputeResponseResidualFromState(x, omega, applied_wrapped);

    REQUIRE(applied_api.Size() == applied_wrapped.Size());
    for (int i = 0; i < applied_api.Size(); i++)
    {
      REQUIRE(applied_api(i) == Approx(applied_wrapped(i)).epsilon(1.0e-12));
    }

    auto K = fixture.space_op->GetStiffnessMatrix<Operator>(Operator::DIAG_ZERO);
    auto M = fixture.space_op->GetMassMatrix<Operator>(Operator::DIAG_ZERO);
    auto A = fixture.space_op->GetSystemMatrix<Operator, double>(1.0, 0.0, -0.30 * 0.30,
                                                                 K.get(), nullptr, M.get());

    mfem::Vector rhs(fixture.space_op->GetNDSpace().GetTrueVSize());
    for (int i = 0; i < rhs.Size(); i++)
    {
      rhs(i) = 1.0 + 0.05 * i;
    }

    Vector solved_api, solved_wrapped, solved_manual;
    fixture.topopt_ctx->SolveCurrentSystem(0.30, rhs, solved_api);
    fixture.topopt_ctx->SolveProxyState(0.30, rhs, solved_wrapped);

    mfem::GMRESSolver gmres(fixture.space_op->GetComm());
    gmres.iterative_mode = false;
    gmres.SetPrintLevel(0);
    gmres.SetRelTol(1.0e-12);
    gmres.SetAbsTol(0.0);
    gmres.SetMaxIter(500);
    gmres.SetOperator(*A);
    solved_manual.SetSize(rhs.Size());
    solved_manual = 0.0;
    gmres.Mult(rhs, solved_manual);

    REQUIRE(solved_api.Size() == solved_wrapped.Size());
    REQUIRE(solved_api.Size() == solved_manual.Size());
    for (int i = 0; i < solved_api.Size(); i++)
    {
      REQUIRE(solved_api(i) == Approx(solved_wrapped(i)).epsilon(1.0e-12));
      REQUIRE(solved_api(i) == Approx(solved_manual(i)).epsilon(1.0e-12));
    }
  }

  SECTION("Projected volume fraction density gradient matches full FD")
  {
    mfem::Vector element_volume(5);
    for (int i = 0; i < element_volume.Size(); i++)
    {
      element_volume(i) = 1.0 + 0.25 * i;
    }

    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    fixture.topopt_ctx->UpdateState(rho, beta, eta);

    mfem::Vector chain_grad;
    fixture.topopt_ctx->ComputeProjectedVolumeFractionDensityGradient(element_volume, beta, eta,
                                                                     chain_grad);

    constexpr double delta_fd = 1.0e-5;
    for (int idx : interior)
    {
      mfem::Vector rho_plus(rho), rho_minus(rho);
      rho_plus(idx) += delta_fd;
      rho_minus(idx) -= delta_fd;
      const double fd =
          (EvaluateProjectedVolumeFraction(fixture, rho_plus, element_volume, beta, eta) -
           EvaluateProjectedVolumeFraction(fixture, rho_minus, element_volume, beta, eta)) /
          (2.0 * delta_fd);

      REQUIRE(std::isfinite(fd));
      REQUIRE(std::isfinite(chain_grad(idx)));
      REQUIRE(fd == Approx(chain_grad(idx)).epsilon(1.0e-3));
    }
  }

  SECTION("Combined response and volume evaluation matches component APIs")
  {
    mfem::Vector element_volume(5);
    for (int i = 0; i < element_volume.Size(); i++)
    {
      element_volume(i) = 1.0 + 0.25 * i;
    }
    constexpr double volume_fraction_target = 0.40;

    fixture.topopt_ctx->SetBackgroundDensity(0.0);

    topopt::TopOptEvaluation combined;
    fixture.topopt_ctx->EvaluateResponseNormWithVolumeConstraint(
        rho, omega, beta, eta, 1.0, 3.0, element_volume, volume_fraction_target, combined);

    const double objective = EvaluateObjective(fixture, rho, beta, eta, omega);
    const double constraint =
        EvaluateProjectedVolumeFraction(fixture, rho, element_volume, beta, eta) -
        volume_fraction_target;

    mfem::Vector objective_grad, constraint_grad;
    fixture.topopt_ctx->UpdatePermittivityFromDensity(rho, beta, eta, 1.0, 3.0);
    fixture.topopt_ctx->ComputeResponseNormDensityGradient(omega, beta, eta, 1.0, 3.0,
                                                           objective_grad);
    fixture.topopt_ctx->ComputeProjectedVolumeFractionDensityGradient(element_volume, beta, eta,
                                                                     constraint_grad);

    REQUIRE(combined.objective == Approx(objective).epsilon(1.0e-12));
    REQUIRE(combined.constraint == Approx(constraint).epsilon(1.0e-12));
    REQUIRE(combined.objective_gradient.Size() == objective_grad.Size());
    REQUIRE(combined.constraint_gradient.Size() == constraint_grad.Size());

    for (int i = 0; i < objective_grad.Size(); i++)
    {
      REQUIRE(combined.objective_gradient(i) ==
              Approx(objective_grad(i)).epsilon(1.0e-12));
      REQUIRE(combined.constraint_gradient(i) ==
              Approx(constraint_grad(i)).epsilon(1.0e-12));
    }
  }

  SECTION("TopOptProblem exposes a stable one-step optimizer interface")
  {
    mfem::Vector element_volume(5);
    for (int i = 0; i < element_volume.Size(); i++)
    {
      element_volume(i) = 1.0 + 0.25 * i;
    }
    constexpr double volume_fraction_target = 0.40;
    constexpr double step_size = 1.0e-3;

    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    topopt::TopOptProblem problem(*fixture.topopt_ctx, omega, beta, eta, 1.0, 3.0,
                                  element_volume, volume_fraction_target);

    REQUIRE(problem.GetNumDesignVariables() == rho.Size());

    topopt::TopOptEvaluation eval_direct, eval_problem;
    fixture.topopt_ctx->EvaluateResponseNormWithVolumeConstraint(
        rho, omega, beta, eta, 1.0, 3.0, element_volume, volume_fraction_target, eval_direct);
    problem.Evaluate(rho, eval_problem);

    REQUIRE(eval_problem.objective == Approx(eval_direct.objective).epsilon(1.0e-12));
    REQUIRE(eval_problem.constraint == Approx(eval_direct.constraint).epsilon(1.0e-12));
    REQUIRE(eval_problem.objective_gradient.Size() == eval_direct.objective_gradient.Size());
    REQUIRE(eval_problem.constraint_gradient.Size() == eval_direct.constraint_gradient.Size());
    for (int i = 0; i < eval_direct.objective_gradient.Size(); i++)
    {
      REQUIRE(eval_problem.objective_gradient(i) ==
              Approx(eval_direct.objective_gradient(i)).epsilon(1.0e-12));
      REQUIRE(eval_problem.constraint_gradient(i) ==
              Approx(eval_direct.constraint_gradient(i)).epsilon(1.0e-12));
    }

    mfem::Vector rho_step(rho);
    topopt::TopOptEvaluation eval_step;
    problem.TakeObjectiveStep(rho_step, step_size, 0.0, 1.0, &eval_step);

    REQUIRE(eval_step.objective == Approx(eval_problem.objective).epsilon(1.0e-12));
    REQUIRE(eval_step.constraint == Approx(eval_problem.constraint).epsilon(1.0e-12));
    for (int i = 0; i < rho_step.Size(); i++)
    {
      const double expected = std::max(
          0.0, std::min(1.0, rho(i) - step_size * eval_problem.objective_gradient(i)));
      REQUIRE(rho_step(i) == Approx(expected).epsilon(1.0e-12));
      REQUIRE(rho_step(i) >= 0.0);
      REQUIRE(rho_step(i) <= 1.0);
    }

    topopt::TopOptEvaluation eval_after;
    problem.Evaluate(rho_step, eval_after);
    REQUIRE(std::isfinite(eval_after.objective));
    REQUIRE(std::isfinite(eval_after.constraint));
  }

  SECTION("TopOptProblem fixed-step loop keeps design bounded and histories finite")
  {
    mfem::Vector element_volume(5);
    for (int i = 0; i < element_volume.Size(); i++)
    {
      element_volume(i) = 1.0 + 0.25 * i;
    }
    constexpr double volume_fraction_target = 0.40;
    constexpr double step_size = 1.0e-3;
    constexpr int num_steps = 3;

    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    topopt::TopOptProblem problem(*fixture.topopt_ctx, omega, beta, eta, 1.0, 3.0,
                                  element_volume, volume_fraction_target);

    mfem::Vector rho_iter(rho);
    topopt::TopOptIterationHistory history;
    problem.RunFixedObjectiveSteps(rho_iter, num_steps, step_size, 0.0, 1.0, &history);

    REQUIRE(history.objective.Size() == num_steps);
    REQUIRE(history.constraint.Size() == num_steps);
    for (int k = 0; k < num_steps; k++)
    {
      REQUIRE(std::isfinite(history.objective(k)));
      REQUIRE(std::isfinite(history.constraint(k)));
    }

    double max_change = 0.0;
    for (int i = 0; i < rho_iter.Size(); i++)
    {
      REQUIRE(rho_iter(i) >= 0.0);
      REQUIRE(rho_iter(i) <= 1.0);
      max_change = std::max(max_change, std::abs(rho_iter(i) - rho(i)));
    }
    REQUIRE(max_change > 0.0);

    topopt::TopOptEvaluation eval_after;
    problem.Evaluate(rho_iter, eval_after);
    REQUIRE(std::isfinite(eval_after.objective));
    REQUIRE(std::isfinite(eval_after.constraint));
  }

  SECTION("TopOptProblem configurable run control supports full and early-stop modes")
  {
    mfem::Vector element_volume(5);
    for (int i = 0; i < element_volume.Size(); i++)
    {
      element_volume(i) = 1.0 + 0.25 * i;
    }
    constexpr double volume_fraction_target = 0.40;
    constexpr double step_size = 1.0e-3;
    constexpr int max_steps = 3;

    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    topopt::TopOptProblem problem(*fixture.topopt_ctx, omega, beta, eta, 1.0, 3.0,
                                  element_volume, volume_fraction_target);

    topopt::TopOptRunOptions run_opts;
    run_opts.max_steps = max_steps;
    run_opts.step_size = step_size;
    run_opts.gradient_tolerance = 0.0;

    mfem::Vector rho_full(rho);
    topopt::TopOptRunSummary full_summary;
    problem.RunObjectiveSteps(rho_full, run_opts, &full_summary);

    REQUIRE(full_summary.num_iterations == max_steps);
    REQUIRE(!full_summary.converged);
    REQUIRE(full_summary.history.objective.Size() == max_steps);
    REQUIRE(full_summary.history.constraint.Size() == max_steps);
    REQUIRE(full_summary.gradient_norm.Size() == max_steps);
    for (int k = 0; k < max_steps; k++)
    {
      REQUIRE(std::isfinite(full_summary.history.objective(k)));
      REQUIRE(std::isfinite(full_summary.history.constraint(k)));
      REQUIRE(std::isfinite(full_summary.gradient_norm(k)));
      REQUIRE(full_summary.gradient_norm(k) > 0.0);
    }

    topopt::TopOptRunOptions stop_opts = run_opts;
    stop_opts.gradient_tolerance = 1.0e300;

    mfem::Vector rho_stop(rho);
    topopt::TopOptRunSummary stop_summary;
    problem.RunObjectiveSteps(rho_stop, stop_opts, &stop_summary);

    REQUIRE(stop_summary.converged);
    REQUIRE(stop_summary.num_iterations == 1);
    REQUIRE(stop_summary.history.objective.Size() == 1);
    REQUIRE(stop_summary.history.constraint.Size() == 1);
    REQUIRE(stop_summary.gradient_norm.Size() == 1);
    REQUIRE(std::isfinite(stop_summary.history.objective(0)));
    REQUIRE(std::isfinite(stop_summary.history.constraint(0)));
    REQUIRE(std::isfinite(stop_summary.gradient_norm(0)));
    for (int i = 0; i < rho_stop.Size(); i++)
    {
      REQUIRE(rho_stop(i) == Approx(rho(i)).epsilon(1.0e-12));
    }
  }

  SECTION("TopOptProblem penalty step uses merit-decreasing acceptance")
  {
    mfem::Vector element_volume(5);
    for (int i = 0; i < element_volume.Size(); i++)
    {
      element_volume(i) = 1.0 + 0.25 * i;
    }
    constexpr double volume_fraction_target = 0.40;
    constexpr double initial_step = 1.0e-2;
    constexpr double penalty = 10.0;

    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    topopt::TopOptProblem problem(*fixture.topopt_ctx, omega, beta, eta, 1.0, 3.0,
                                  element_volume, volume_fraction_target);

    topopt::TopOptEvaluation eval_before;
    problem.Evaluate(rho, eval_before);
    const double merit_before =
        topopt::TopOptProblem::ComputeQuadraticPenaltyMerit(eval_before, penalty);

    mfem::Vector rho_trial(rho);
    topopt::TopOptEvaluation eval_after;
    double accepted_step = -1.0;
    const bool accepted =
        problem.TakePenaltyStep(rho_trial, initial_step, penalty, 8, 0.5, 0.0, 1.0,
                                &eval_after, &accepted_step);

    if (accepted)
    {
      const double merit_after =
          topopt::TopOptProblem::ComputeQuadraticPenaltyMerit(eval_after, penalty);
      const double tol = 1.0e-12 * std::max(1.0, std::abs(merit_before));

      REQUIRE(accepted_step > 0.0);
      REQUIRE(accepted_step <= initial_step);
      REQUIRE(merit_after <= merit_before + tol);

      topopt::TopOptEvaluation eval_check;
      problem.Evaluate(rho_trial, eval_check);
      REQUIRE(eval_after.objective == Approx(eval_check.objective).epsilon(1.0e-12));
      REQUIRE(eval_after.constraint == Approx(eval_check.constraint).epsilon(1.0e-12));
    }
    else
    {
      REQUIRE(accepted_step == Approx(0.0).epsilon(1.0e-12));
      REQUIRE(eval_after.objective == Approx(eval_before.objective).epsilon(1.0e-12));
      REQUIRE(eval_after.constraint == Approx(eval_before.constraint).epsilon(1.0e-12));
      for (int i = 0; i < rho_trial.Size(); i++)
      {
        REQUIRE(rho_trial(i) == Approx(rho(i)).epsilon(1.0e-12));
      }
    }

    for (int i = 0; i < rho_trial.Size(); i++)
    {
      REQUIRE(rho_trial(i) >= 0.0);
      REQUIRE(rho_trial(i) <= 1.0);
    }
  }

  SECTION("TopOptProblem penalty loop records merit and accepted steps")
  {
    mfem::Vector element_volume(5);
    for (int i = 0; i < element_volume.Size(); i++)
    {
      element_volume(i) = 1.0 + 0.25 * i;
    }
    constexpr double volume_fraction_target = 0.40;
    constexpr double initial_step = 1.0e-3;
    constexpr double penalty = 10.0;
    constexpr int num_steps = 3;

    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    topopt::TopOptProblem problem(*fixture.topopt_ctx, omega, beta, eta, 1.0, 3.0,
                                  element_volume, volume_fraction_target);

    mfem::Vector rho_penalty(rho);
    topopt::TopOptPenaltyRunSummary summary;
    problem.RunPenaltySteps(rho_penalty, num_steps, initial_step, penalty, 8, 0.5, 0.0, 1.0,
                            &summary);

    REQUIRE(summary.num_iterations == num_steps);
    REQUIRE(summary.history.objective.Size() == num_steps);
    REQUIRE(summary.history.constraint.Size() == num_steps);
    REQUIRE(summary.merit.Size() == num_steps);
    REQUIRE(summary.accepted_step.Size() == num_steps);
    REQUIRE(summary.accepted.Size() == num_steps);
    REQUIRE(summary.num_accepted >= 1);

    int accepted_count = 0;
    for (int k = 0; k < num_steps; k++)
    {
      REQUIRE(std::isfinite(summary.history.objective(k)));
      REQUIRE(std::isfinite(summary.history.constraint(k)));
      REQUIRE(std::isfinite(summary.merit(k)));
      REQUIRE(std::isfinite(summary.accepted_step(k)));
      REQUIRE(summary.accepted_step(k) >= 0.0);
      REQUIRE(summary.accepted_step(k) <= initial_step);
      REQUIRE(summary.accepted(k) == Approx(summary.accepted_step(k) > 0.0 ? 1.0 : 0.0)
                                        .epsilon(1.0e-12));
      accepted_count += (summary.accepted(k) > 0.5) ? 1 : 0;
    }
    REQUIRE(summary.num_accepted == accepted_count);

    for (int i = 0; i < rho_penalty.Size(); i++)
    {
      REQUIRE(rho_penalty(i) >= 0.0);
      REQUIRE(rho_penalty(i) <= 1.0);
    }

    topopt::TopOptEvaluation eval_final;
    problem.Evaluate(rho_penalty, eval_final);
    const double merit_final =
        topopt::TopOptProblem::ComputeQuadraticPenaltyMerit(eval_final, penalty);
    REQUIRE(std::isfinite(eval_final.objective));
    REQUIRE(std::isfinite(eval_final.constraint));
    REQUIRE(merit_final == Approx(summary.merit(num_steps - 1)).epsilon(1.0e-12));
  }

  SECTION("TopOptProblem configurable penalty run supports full, merit-stop, and reject-stop")
  {
    mfem::Vector element_volume(5);
    for (int i = 0; i < element_volume.Size(); i++)
    {
      element_volume(i) = 1.0 + 0.25 * i;
    }
    constexpr double volume_fraction_target = 0.40;

    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    topopt::TopOptProblem problem(*fixture.topopt_ctx, omega, beta, eta, 1.0, 3.0,
                                  element_volume, volume_fraction_target);

    topopt::TopOptPenaltyRunOptions full_opts;
    full_opts.max_steps = 3;
    full_opts.initial_step = 1.0e-3;
    full_opts.penalty = 10.0;

    mfem::Vector rho_full(rho);
    topopt::TopOptPenaltyRunSummary full_summary;
    problem.RunPenaltySteps(rho_full, full_opts, &full_summary);

    REQUIRE(full_summary.num_iterations == full_opts.max_steps);
    REQUIRE(full_summary.num_accepted >= 1);
    REQUIRE(full_summary.num_rejected >= 0);
    REQUIRE(!full_summary.converged);
    REQUIRE(!full_summary.stopped_on_reject_limit);

    topopt::TopOptPenaltyRunOptions merit_opts = full_opts;
    merit_opts.max_steps = 4;
    merit_opts.merit_tolerance = 1.0e300;

    mfem::Vector rho_merit(rho);
    topopt::TopOptPenaltyRunSummary merit_summary;
    problem.RunPenaltySteps(rho_merit, merit_opts, &merit_summary);

    REQUIRE(merit_summary.converged);
    REQUIRE(!merit_summary.stopped_on_reject_limit);
    REQUIRE(merit_summary.num_iterations == 2);
    REQUIRE(merit_summary.history.objective.Size() == 2);
    REQUIRE(merit_summary.history.constraint.Size() == 2);
    REQUIRE(merit_summary.merit.Size() == 2);
    REQUIRE(merit_summary.accepted_step.Size() == 2);
    REQUIRE(merit_summary.accepted.Size() == 2);

    topopt::TopOptPenaltyRunOptions reject_opts = full_opts;
    reject_opts.max_steps = 4;
    reject_opts.max_consecutive_rejects = 1;
    reject_opts.lower_bound = 0.5;
    reject_opts.upper_bound = 0.5;

    mfem::Vector rho_reject(rho);
    topopt::TopOptPenaltyRunSummary reject_summary;
    problem.RunPenaltySteps(rho_reject, reject_opts, &reject_summary);

    REQUIRE(!reject_summary.converged);
    REQUIRE(reject_summary.stopped_on_reject_limit);
    REQUIRE(reject_summary.num_iterations == 1);
    REQUIRE(reject_summary.num_accepted == 0);
    REQUIRE(reject_summary.num_rejected == 1);
    REQUIRE(reject_summary.history.objective.Size() == 1);
    REQUIRE(reject_summary.history.constraint.Size() == 1);
    REQUIRE(reject_summary.merit.Size() == 1);
    REQUIRE(reject_summary.accepted_step.Size() == 1);
    REQUIRE(reject_summary.accepted.Size() == 1);
    REQUIRE(reject_summary.accepted_step(0) == Approx(0.0).epsilon(1.0e-12));
    REQUIRE(reject_summary.accepted(0) == Approx(0.0).epsilon(1.0e-12));
    for (int i = 0; i < rho_reject.Size(); i++)
    {
      REQUIRE(rho_reject(i) == Approx(rho(i)).epsilon(1.0e-12));
    }
  }

  SECTION("TopOptProblem unified session wrapper matches objective and penalty runners")
  {
    mfem::Vector element_volume(5);
    for (int i = 0; i < element_volume.Size(); i++)
    {
      element_volume(i) = 1.0 + 0.25 * i;
    }
    constexpr double volume_fraction_target = 0.40;

    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    topopt::TopOptProblem problem(*fixture.topopt_ctx, omega, beta, eta, 1.0, 3.0,
                                  element_volume, volume_fraction_target);

    topopt::TopOptRunOptions objective_opts;
    objective_opts.max_steps = 3;
    objective_opts.step_size = 1.0e-3;
    objective_opts.gradient_tolerance = 0.0;

    mfem::Vector rho_obj_direct(rho), rho_obj_session(rho);
    topopt::TopOptRunSummary objective_direct;
    topopt::TopOptSessionOptions objective_session_opts;
    objective_session_opts.mode = topopt::TopOptSessionMode::OBJECTIVE;
    objective_session_opts.objective = objective_opts;
    topopt::TopOptSessionSummary objective_session;

    problem.RunObjectiveSteps(rho_obj_direct, objective_opts, &objective_direct);
    problem.RunSession(rho_obj_session, objective_session_opts, &objective_session);

    REQUIRE(objective_session.mode == topopt::TopOptSessionMode::OBJECTIVE);
    REQUIRE(objective_session.num_iterations == objective_direct.num_iterations);
    REQUIRE(objective_session.num_accepted == objective_direct.num_iterations);
    REQUIRE(objective_session.num_rejected == 0);
    REQUIRE(objective_session.converged == objective_direct.converged);
    REQUIRE(!objective_session.stopped_on_reject_limit);
    REQUIRE(objective_session.history.objective.Size() == objective_direct.history.objective.Size());
    REQUIRE(objective_session.history.constraint.Size() == objective_direct.history.constraint.Size());
    REQUIRE(objective_session.trace_value.Size() == objective_direct.gradient_norm.Size());
    REQUIRE(objective_session.accepted_step.Size() == objective_direct.num_iterations);
    REQUIRE(objective_session.accepted.Size() == objective_direct.num_iterations);
    for (int i = 0; i < objective_direct.num_iterations; i++)
    {
      REQUIRE(objective_session.history.objective(i) ==
              Approx(objective_direct.history.objective(i)).epsilon(1.0e-12));
      REQUIRE(objective_session.history.constraint(i) ==
              Approx(objective_direct.history.constraint(i)).epsilon(1.0e-12));
      REQUIRE(objective_session.trace_value(i) ==
              Approx(objective_direct.gradient_norm(i)).epsilon(1.0e-12));
      REQUIRE(objective_session.accepted_step(i) ==
              Approx(objective_opts.step_size).epsilon(1.0e-12));
      REQUIRE(objective_session.accepted(i) == Approx(1.0).epsilon(1.0e-12));
    }
    for (int i = 0; i < rho_obj_direct.Size(); i++)
    {
      REQUIRE(rho_obj_session(i) == Approx(rho_obj_direct(i)).epsilon(1.0e-12));
    }

    topopt::TopOptPenaltyRunOptions penalty_opts;
    penalty_opts.max_steps = 3;
    penalty_opts.initial_step = 1.0e-3;
    penalty_opts.penalty = 10.0;

    mfem::Vector rho_penalty_direct(rho), rho_penalty_session(rho);
    topopt::TopOptPenaltyRunSummary penalty_direct;
    topopt::TopOptSessionOptions penalty_session_opts;
    penalty_session_opts.mode = topopt::TopOptSessionMode::PENALTY;
    penalty_session_opts.penalty = penalty_opts;
    topopt::TopOptSessionSummary penalty_session;

    problem.RunPenaltySteps(rho_penalty_direct, penalty_opts, &penalty_direct);
    problem.RunSession(rho_penalty_session, penalty_session_opts, &penalty_session);

    REQUIRE(penalty_session.mode == topopt::TopOptSessionMode::PENALTY);
    REQUIRE(penalty_session.num_iterations == penalty_direct.num_iterations);
    REQUIRE(penalty_session.num_accepted == penalty_direct.num_accepted);
    REQUIRE(penalty_session.num_rejected == penalty_direct.num_rejected);
    REQUIRE(penalty_session.converged == penalty_direct.converged);
    REQUIRE(penalty_session.stopped_on_reject_limit ==
            penalty_direct.stopped_on_reject_limit);
    REQUIRE(penalty_session.history.objective.Size() == penalty_direct.history.objective.Size());
    REQUIRE(penalty_session.history.constraint.Size() == penalty_direct.history.constraint.Size());
    REQUIRE(penalty_session.trace_value.Size() == penalty_direct.merit.Size());
    REQUIRE(penalty_session.accepted_step.Size() == penalty_direct.accepted_step.Size());
    REQUIRE(penalty_session.accepted.Size() == penalty_direct.accepted.Size());
    for (int i = 0; i < penalty_direct.num_iterations; i++)
    {
      REQUIRE(penalty_session.history.objective(i) ==
              Approx(penalty_direct.history.objective(i)).epsilon(1.0e-12));
      REQUIRE(penalty_session.history.constraint(i) ==
              Approx(penalty_direct.history.constraint(i)).epsilon(1.0e-12));
      REQUIRE(penalty_session.trace_value(i) ==
              Approx(penalty_direct.merit(i)).epsilon(1.0e-12));
      REQUIRE(penalty_session.accepted_step(i) ==
              Approx(penalty_direct.accepted_step(i)).epsilon(1.0e-12));
      REQUIRE(penalty_session.accepted(i) ==
              Approx(penalty_direct.accepted(i)).epsilon(1.0e-12));
    }
    for (int i = 0; i < rho_penalty_direct.Size(); i++)
    {
      REQUIRE(rho_penalty_session(i) == Approx(rho_penalty_direct(i)).epsilon(1.0e-12));
    }
  }

  SECTION("TopOptDriver owns design, options, and latest session summary")
  {
    mfem::Vector element_volume(5);
    for (int i = 0; i < element_volume.Size(); i++)
    {
      element_volume(i) = 1.0 + 0.25 * i;
    }
    constexpr double volume_fraction_target = 0.40;

    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    topopt::TopOptProblem problem(*fixture.topopt_ctx, omega, beta, eta, 1.0, 3.0,
                                  element_volume, volume_fraction_target);

    topopt::TopOptSessionOptions objective_opts;
    objective_opts.mode = topopt::TopOptSessionMode::OBJECTIVE;
    objective_opts.objective.max_steps = 3;
    objective_opts.objective.step_size = 1.0e-3;
    objective_opts.objective.gradient_tolerance = 0.0;

    topopt::TopOptDriver driver(problem, objective_opts, rho);
    REQUIRE(driver.GetNumDesignVariables() == rho.Size());
    REQUIRE(driver.GetOptions().mode == topopt::TopOptSessionMode::OBJECTIVE);

    mfem::Vector rho_objective_direct(rho);
    topopt::TopOptSessionSummary objective_direct;
    problem.RunSession(rho_objective_direct, objective_opts, &objective_direct);
    const auto &objective_driver = driver.Run();

    REQUIRE(objective_driver.mode == objective_direct.mode);
    REQUIRE(objective_driver.num_iterations == objective_direct.num_iterations);
    REQUIRE(objective_driver.num_accepted == objective_direct.num_accepted);
    REQUIRE(objective_driver.num_rejected == objective_direct.num_rejected);
    REQUIRE(objective_driver.converged == objective_direct.converged);
    REQUIRE(objective_driver.stopped_on_reject_limit ==
            objective_direct.stopped_on_reject_limit);
    REQUIRE(driver.GetSummary().num_iterations == objective_driver.num_iterations);
    for (int i = 0; i < objective_direct.num_iterations; i++)
    {
      REQUIRE(objective_driver.history.objective(i) ==
              Approx(objective_direct.history.objective(i)).epsilon(1.0e-12));
      REQUIRE(objective_driver.history.constraint(i) ==
              Approx(objective_direct.history.constraint(i)).epsilon(1.0e-12));
      REQUIRE(objective_driver.trace_value(i) ==
              Approx(objective_direct.trace_value(i)).epsilon(1.0e-12));
      REQUIRE(objective_driver.accepted_step(i) ==
              Approx(objective_direct.accepted_step(i)).epsilon(1.0e-12));
      REQUIRE(objective_driver.accepted(i) ==
              Approx(objective_direct.accepted(i)).epsilon(1.0e-12));
    }
    for (int i = 0; i < rho.Size(); i++)
    {
      REQUIRE(driver.GetDesign()(i) == Approx(rho_objective_direct(i)).epsilon(1.0e-12));
    }

    topopt::TopOptSessionOptions penalty_opts;
    penalty_opts.mode = topopt::TopOptSessionMode::PENALTY;
    penalty_opts.penalty.max_steps = 3;
    penalty_opts.penalty.initial_step = 1.0e-3;
    penalty_opts.penalty.penalty = 10.0;

    driver.SetOptions(penalty_opts);
    driver.SetDesign(rho);
    REQUIRE(driver.GetOptions().mode == topopt::TopOptSessionMode::PENALTY);
    driver.SetUniformDesign(0.5);
    for (int i = 0; i < driver.GetDesign().Size(); i++)
    {
      REQUIRE(driver.GetDesign()(i) == Approx(0.5).epsilon(1.0e-12));
    }

    mfem::Vector rho_penalty_direct(5);
    rho_penalty_direct = 0.5;
    topopt::TopOptSessionSummary penalty_direct;
    problem.RunSession(rho_penalty_direct, penalty_opts, &penalty_direct);
    const auto &penalty_driver = driver.Run();

    REQUIRE(penalty_driver.mode == penalty_direct.mode);
    REQUIRE(penalty_driver.num_iterations == penalty_direct.num_iterations);
    REQUIRE(penalty_driver.num_accepted == penalty_direct.num_accepted);
    REQUIRE(penalty_driver.num_rejected == penalty_direct.num_rejected);
    REQUIRE(penalty_driver.converged == penalty_direct.converged);
    REQUIRE(penalty_driver.stopped_on_reject_limit ==
            penalty_direct.stopped_on_reject_limit);
    for (int i = 0; i < penalty_direct.num_iterations; i++)
    {
      REQUIRE(penalty_driver.history.objective(i) ==
              Approx(penalty_direct.history.objective(i)).epsilon(1.0e-12));
      REQUIRE(penalty_driver.history.constraint(i) ==
              Approx(penalty_direct.history.constraint(i)).epsilon(1.0e-12));
      REQUIRE(penalty_driver.trace_value(i) ==
              Approx(penalty_direct.trace_value(i)).epsilon(1.0e-12));
      REQUIRE(penalty_driver.accepted_step(i) ==
              Approx(penalty_direct.accepted_step(i)).epsilon(1.0e-12));
      REQUIRE(penalty_driver.accepted(i) ==
              Approx(penalty_direct.accepted(i)).epsilon(1.0e-12));
    }
    for (int i = 0; i < rho_penalty_direct.Size(); i++)
    {
      REQUIRE(driver.GetDesign()(i) == Approx(rho_penalty_direct(i)).epsilon(1.0e-12));
    }
  }

  SECTION("TopOptDriver exposes latest scalars and accumulates run history")
  {
    mfem::Vector element_volume(5);
    for (int i = 0; i < element_volume.Size(); i++)
    {
      element_volume(i) = 1.0 + 0.25 * i;
    }
    constexpr double volume_fraction_target = 0.40;

    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    topopt::TopOptProblem problem(*fixture.topopt_ctx, omega, beta, eta, 1.0, 3.0,
                                  element_volume, volume_fraction_target);

    topopt::TopOptSessionOptions opts;
    opts.mode = topopt::TopOptSessionMode::OBJECTIVE;
    opts.objective.max_steps = 3;
    opts.objective.step_size = 1.0e-3;
    opts.objective.gradient_tolerance = 0.0;

    topopt::TopOptDriver driver(problem, opts, rho);
    REQUIRE(driver.GetRunHistory().empty());

    const auto &first = driver.Run();
    REQUIRE(driver.GetRunHistory().size() == 1);
    REQUIRE(driver.GetLatestObjective() ==
            Approx(first.history.objective(first.num_iterations - 1)).epsilon(1.0e-12));
    REQUIRE(driver.GetLatestConstraint() ==
            Approx(first.history.constraint(first.num_iterations - 1)).epsilon(1.0e-12));
    REQUIRE(driver.GetLatestTraceValue() ==
            Approx(first.trace_value(first.num_iterations - 1)).epsilon(1.0e-12));

    const auto &history_first = driver.GetRunHistory().front();
    REQUIRE(history_first.mode == first.mode);
    REQUIRE(history_first.num_iterations == first.num_iterations);
    REQUIRE(history_first.num_accepted == first.num_accepted);
    REQUIRE(history_first.num_rejected == first.num_rejected);

    driver.SetDesign(rho);
    const auto &second = driver.Run(false);
    REQUIRE(driver.GetRunHistory().size() == 1);
    REQUIRE(driver.GetLatestObjective() ==
            Approx(second.history.objective(second.num_iterations - 1)).epsilon(1.0e-12));

    driver.SetDesign(rho);
    driver.Run();
    REQUIRE(driver.GetRunHistory().size() == 2);
    const auto &history_last = driver.GetRunHistory().back();
    REQUIRE(history_last.mode == driver.GetSummary().mode);
    REQUIRE(history_last.num_iterations == driver.GetSummary().num_iterations);
    REQUIRE(history_last.num_accepted == driver.GetSummary().num_accepted);
    REQUIRE(history_last.num_rejected == driver.GetSummary().num_rejected);
    REQUIRE(driver.GetLatestObjective() ==
            Approx(history_last.history.objective(history_last.num_iterations - 1))
                .epsilon(1.0e-12));
    REQUIRE(driver.GetLatestConstraint() ==
            Approx(history_last.history.constraint(history_last.num_iterations - 1))
                .epsilon(1.0e-12));
    REQUIRE(driver.GetLatestTraceValue() ==
            Approx(history_last.trace_value(history_last.num_iterations - 1))
                .epsilon(1.0e-12));

    driver.ClearRunHistory();
    REQUIRE(driver.GetRunHistory().empty());
  }

  SECTION("TopOptDriver latest report mirrors the latest summary")
  {
    mfem::Vector element_volume(5);
    for (int i = 0; i < element_volume.Size(); i++)
    {
      element_volume(i) = 1.0 + 0.25 * i;
    }
    constexpr double volume_fraction_target = 0.40;

    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    topopt::TopOptProblem problem(*fixture.topopt_ctx, omega, beta, eta, 1.0, 3.0,
                                  element_volume, volume_fraction_target);

    topopt::TopOptSessionOptions opts;
    opts.mode = topopt::TopOptSessionMode::OBJECTIVE;
    opts.objective.max_steps = 3;
    opts.objective.step_size = 1.0e-3;
    opts.objective.gradient_tolerance = 0.0;

    topopt::TopOptDriver driver(problem, opts, rho);
    REQUIRE(!driver.HasRun());

    const auto &first = driver.Run();
    REQUIRE(driver.HasRun());
    auto first_report = driver.GetLatestReport();
    REQUIRE(first_report.mode == first.mode);
    REQUIRE(first_report.objective ==
            Approx(first.history.objective(first.num_iterations - 1)).epsilon(1.0e-12));
    REQUIRE(first_report.constraint ==
            Approx(first.history.constraint(first.num_iterations - 1)).epsilon(1.0e-12));
    REQUIRE(first_report.trace_value ==
            Approx(first.trace_value(first.num_iterations - 1)).epsilon(1.0e-12));
    REQUIRE(first_report.num_iterations == first.num_iterations);
    REQUIRE(first_report.num_accepted == first.num_accepted);
    REQUIRE(first_report.num_rejected == first.num_rejected);
    REQUIRE(first_report.converged == first.converged);
    REQUIRE(first_report.stopped_on_reject_limit == first.stopped_on_reject_limit);
    REQUIRE(first_report.num_recorded_runs == 1);

    driver.SetDesign(rho);
    const auto &second = driver.Run(false);
    auto second_report = driver.GetLatestReport();
    REQUIRE(second_report.mode == second.mode);
    REQUIRE(second_report.objective ==
            Approx(second.history.objective(second.num_iterations - 1)).epsilon(1.0e-12));
    REQUIRE(second_report.constraint ==
            Approx(second.history.constraint(second.num_iterations - 1)).epsilon(1.0e-12));
    REQUIRE(second_report.trace_value ==
            Approx(second.trace_value(second.num_iterations - 1)).epsilon(1.0e-12));
    REQUIRE(second_report.num_iterations == second.num_iterations);
    REQUIRE(second_report.num_accepted == second.num_accepted);
    REQUIRE(second_report.num_rejected == second.num_rejected);
    REQUIRE(second_report.converged == second.converged);
    REQUIRE(second_report.stopped_on_reject_limit == second.stopped_on_reject_limit);
    REQUIRE(second_report.num_recorded_runs == 1);

    topopt::TopOptSessionOptions penalty_opts;
    penalty_opts.mode = topopt::TopOptSessionMode::PENALTY;
    penalty_opts.penalty.max_steps = 3;
    penalty_opts.penalty.initial_step = 1.0e-3;
    penalty_opts.penalty.penalty = 10.0;
    driver.SetOptions(penalty_opts);
    driver.SetUniformDesign(0.5);
    const auto &third = driver.Run();
    auto third_report = driver.GetLatestReport();
    REQUIRE(third_report.mode == third.mode);
    REQUIRE(third_report.objective ==
            Approx(third.history.objective(third.num_iterations - 1)).epsilon(1.0e-12));
    REQUIRE(third_report.constraint ==
            Approx(third.history.constraint(third.num_iterations - 1)).epsilon(1.0e-12));
    REQUIRE(third_report.trace_value ==
            Approx(third.trace_value(third.num_iterations - 1)).epsilon(1.0e-12));
    REQUIRE(third_report.num_iterations == third.num_iterations);
    REQUIRE(third_report.num_accepted == third.num_accepted);
    REQUIRE(third_report.num_rejected == third.num_rejected);
    REQUIRE(third_report.converged == third.converged);
    REQUIRE(third_report.stopped_on_reject_limit == third.stopped_on_reject_limit);
    REQUIRE(third_report.num_recorded_runs == 2);
  }

  SECTION("TopOptDriver report export helpers expose CSV-friendly output")
  {
    mfem::Vector element_volume(5);
    for (int i = 0; i < element_volume.Size(); i++)
    {
      element_volume(i) = 1.0 + 0.25 * i;
    }
    constexpr double volume_fraction_target = 0.40;

    fixture.topopt_ctx->SetBackgroundDensity(0.0);
    topopt::TopOptProblem problem(*fixture.topopt_ctx, omega, beta, eta, 1.0, 3.0,
                                  element_volume, volume_fraction_target);

    topopt::TopOptSessionOptions objective_opts;
    objective_opts.mode = topopt::TopOptSessionMode::OBJECTIVE;
    objective_opts.objective.max_steps = 3;
    objective_opts.objective.step_size = 1.0e-3;
    objective_opts.objective.gradient_tolerance = 0.0;

    topopt::TopOptDriver driver(problem, objective_opts, rho);
    driver.Run();

    topopt::TopOptSessionOptions penalty_opts;
    penalty_opts.mode = topopt::TopOptSessionMode::PENALTY;
    penalty_opts.penalty.max_steps = 3;
    penalty_opts.penalty.initial_step = 1.0e-3;
    penalty_opts.penalty.penalty = 10.0;
    driver.SetOptions(penalty_opts);
    driver.SetUniformDesign(0.5);
    driver.Run();

    const std::string header = topopt::TopOptDriverReport::CsvHeader();
    REQUIRE(header ==
            "mode,objective,constraint,trace_value,num_iterations,num_accepted,"
            "num_rejected,converged,stopped_on_reject_limit,num_recorded_runs");

    const auto latest = driver.GetLatestReport();
    const std::string latest_csv = latest.ToCsvRow();
    REQUIRE(!latest_csv.empty());
    REQUIRE(latest_csv.find("penalty,") == 0);

    std::vector<std::string> latest_fields;
    {
      std::stringstream ss(latest_csv);
      std::string field;
      while (std::getline(ss, field, ','))
      {
        latest_fields.push_back(field);
      }
    }
    REQUIRE(latest_fields.size() == 10);
    REQUIRE(latest_fields[0] == "penalty");
    REQUIRE(latest_fields[4] == std::to_string(latest.num_iterations));
    REQUIRE(latest_fields[5] == std::to_string(latest.num_accepted));
    REQUIRE(latest_fields[6] == std::to_string(latest.num_rejected));
    REQUIRE(latest_fields[7] == (latest.converged ? "1" : "0"));
    REQUIRE(latest_fields[8] == (latest.stopped_on_reject_limit ? "1" : "0"));
    REQUIRE(latest_fields[9] == std::to_string(latest.num_recorded_runs));

    const auto reports = driver.GetRunReports();
    REQUIRE(reports.size() == 2);
    REQUIRE(reports[0].mode == topopt::TopOptSessionMode::OBJECTIVE);
    REQUIRE(reports[0].num_recorded_runs == 1);
    REQUIRE(reports[1].mode == topopt::TopOptSessionMode::PENALTY);
    REQUIRE(reports[1].num_recorded_runs == 2);
    REQUIRE(reports[1].objective == Approx(latest.objective).epsilon(1.0e-12));
    REQUIRE(reports[1].constraint == Approx(latest.constraint).epsilon(1.0e-12));
    REQUIRE(reports[1].trace_value == Approx(latest.trace_value).epsilon(1.0e-12));
  }
}

}  // namespace palace
