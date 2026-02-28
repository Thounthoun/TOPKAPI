// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <memory>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fem/mesh.hpp"
#include "linalg/vector.hpp"
#include "models/spaceoperator.hpp"
#include "utils/communication.hpp"
#include "utils/iodata.hpp"
#include "utils/topopt.hpp"
#include "utils/units.hpp"

namespace palace
{
using namespace Catch;

namespace
{

struct ForwardSpaceOperatorFixture
{
  Units units;
  IoData iodata;
  std::unique_ptr<mfem::Mesh> serial_mesh;
  std::vector<std::unique_ptr<Mesh>> mesh;
  std::unique_ptr<SpaceOperator> space_op;
  topopt::DesignLayout design_layout;

  ForwardSpaceOperatorFixture()
    : units(1.0, 1.0), iodata(units),
      serial_mesh(std::make_unique<mfem::Mesh>(
          mfem::Mesh::MakeCartesian3D(2, 1, 1, mfem::Element::HEXAHEDRON)))
  {
    auto &material = iodata.domains.materials.emplace_back();
    material.attributes = {1};
    material.epsilon_r.s = {1.0, 1.0, 1.0};
    material.mu_r.s = {1.0, 1.0, 1.0};
    iodata.problem.type = ProblemType::EIGENMODE;
    iodata.solver.order = 1;
    iodata.CheckConfiguration();

    mfem::Array<int> attrs(1);
    attrs[0] = 1;
    design_layout.SetSequential(attrs);

    auto par_mesh = std::make_unique<mfem::ParMesh>(Mpi::World(), *serial_mesh);
    iodata.NondimensionalizeInputs(*par_mesh);
    mesh.push_back(std::make_unique<Mesh>(std::move(par_mesh)));
    space_op = std::make_unique<SpaceOperator>(iodata, mesh);
  }
};

double DiagonalChecksum(const Operator &op, MPI_Comm comm)
{
  Vector diag;
  op.AssembleDiagonal(diag);
  double sum = 0.0;
  for (int i = 0; i < diag.Size(); i++)
  {
    sum += diag(i);
  }
  Mpi::GlobalSum(1, &sum, comm);
  return sum;
}

double ResponseNorm(SpaceOperator &space_op, double omega)
{
  auto K = space_op.GetStiffnessMatrix<Operator>(Operator::DIAG_ZERO);
  auto M = space_op.GetMassMatrix<Operator>(Operator::DIAG_ZERO);
  auto A = space_op.GetSystemMatrix<Operator, double>(1.0, 0.0, -omega * omega, K.get(),
                                                      nullptr, M.get());

  Vector x(space_op.GetNDSpace().GetTrueVSize()), y(space_op.GetNDSpace().GetTrueVSize());
  x = 1.0;
  y = 0.0;
  A->Mult(x, y);
  return linalg::Norml2(space_op.GetComm(), y);
}

void UpdateUniformDensity(SpaceOperator &space_op, const topopt::DesignLayout &design_layout,
                         double rho_hat)
{
  topopt::DesignState state(design_layout.Size());
  state.SetUniformDensity(rho_hat);
  space_op.GetMaterialOp().UpdatePermittivityNSquared(design_layout.GetAttributes(),
                                                      state.GetDensity(), 1.0, 3.0);
}

}  // namespace

TEST_CASE("TopOpt forward reassembly path", "[topopt][forward][Serial]")
{
  SECTION("T2.1 - Manual density updates produce distinct responses")
  {
    ForwardSpaceOperatorFixture fixture;
    constexpr double omega = 0.75;
    std::array<double, 3> response{};
    std::array<double, 3> mass_diag{};
    const std::array<double, 3> rho_hat = {0.0, 0.5, 1.0};

    for (std::size_t i = 0; i < rho_hat.size(); i++)
    {
      UpdateUniformDensity(*fixture.space_op, fixture.design_layout, rho_hat[i]);
      fixture.iodata.CheckConfiguration();
      auto M = fixture.space_op->GetMassMatrix<Operator>(Operator::DIAG_ZERO);
      mass_diag[i] = DiagonalChecksum(*M, fixture.space_op->GetComm());
      response[i] = ResponseNorm(*fixture.space_op, omega);

      REQUIRE(std::isfinite(mass_diag[i]));
      REQUIRE(std::isfinite(response[i]));
      REQUIRE(response[i] > 0.0);
    }

    REQUIRE(mass_diag[0] < mass_diag[1]);
    REQUIRE(mass_diag[1] < mass_diag[2]);
    REQUIRE(response[0] != Approx(response[1]).epsilon(1.0e-8));
    REQUIRE(response[1] != Approx(response[2]).epsilon(1.0e-8));
    REQUIRE(response[0] != Approx(response[2]).epsilon(1.0e-8));
  }

  SECTION("T2.2 - Mass matrix reassembly changes while stiffness stays fixed")
  {
    ForwardSpaceOperatorFixture fixture;
    constexpr double omega = 0.75;

    fixture.iodata.CheckConfiguration();
    auto K0 = fixture.space_op->GetStiffnessMatrix<Operator>(Operator::DIAG_ZERO);
    auto M0 = fixture.space_op->GetMassMatrix<Operator>(Operator::DIAG_ZERO);
    const double k0 = DiagonalChecksum(*K0, fixture.space_op->GetComm());
    const double m0 = DiagonalChecksum(*M0, fixture.space_op->GetComm());
    const double r0 = ResponseNorm(*fixture.space_op, omega);

    UpdateUniformDensity(*fixture.space_op, fixture.design_layout, 0.5);

    fixture.iodata.CheckConfiguration();
    auto K1 = fixture.space_op->GetStiffnessMatrix<Operator>(Operator::DIAG_ZERO);
    auto M1 = fixture.space_op->GetMassMatrix<Operator>(Operator::DIAG_ZERO);
    const double k1 = DiagonalChecksum(*K1, fixture.space_op->GetComm());
    const double m1 = DiagonalChecksum(*M1, fixture.space_op->GetComm());
    const double r1 = ResponseNorm(*fixture.space_op, omega);

    REQUIRE(k1 == Approx(k0).margin(1.0e-12));
    REQUIRE(m1 != Approx(m0).epsilon(1.0e-8));
    REQUIRE(m1 > m0);
    REQUIRE(r1 != Approx(r0).epsilon(1.0e-8));
  }
}

}  // namespace palace
