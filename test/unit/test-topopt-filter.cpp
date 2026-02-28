// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <utility>
#include <vector>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "linalg/vector.hpp"
#include "utils/communication.hpp"
#include "utils/topopt.hpp"

namespace palace
{
using namespace Catch;

namespace
{

double Dot(const mfem::Vector &x, const mfem::Vector &y)
{
  REQUIRE(x.Size() == y.Size());
  return linalg::Dot(Mpi::World(), x, y);
}

double ElementCenterX(mfem::ParMesh &mesh, int elem)
{
  mfem::ElementTransformation *T = mesh.GetElementTransformation(elem);
  const mfem::IntegrationPoint &ip = mfem::Geometries.GetCenter(T->GetGeometryType());
  mfem::Vector x(mesh.SpaceDimension());
  T->Transform(ip, x);
  return x(0);
}

std::vector<std::pair<double, double>> SampleElementCenters(mfem::ParMesh &mesh,
                                                            const topopt::HelmholtzFilter &filter,
                                                            const mfem::Vector &rho_tilde)
{
  const auto &fespace = filter.GetFESpace();
  mfem::ParGridFunction gf(const_cast<mfem::ParFiniteElementSpace *>(&fespace));
  gf.SetFromTrueDofs(rho_tilde);

  std::vector<std::pair<double, double>> samples;
  samples.reserve(mesh.GetNE());
  for (int i = 0; i < mesh.GetNE(); i++)
  {
    mfem::ElementTransformation *T = mesh.GetElementTransformation(i);
    const mfem::IntegrationPoint &ip = mfem::Geometries.GetCenter(T->GetGeometryType());
    samples.emplace_back(ElementCenterX(mesh, i), gf.GetValue(i, ip));
  }
  std::sort(samples.begin(), samples.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  return samples;
}

}  // namespace

TEST_CASE("TopOpt Helmholtz filter", "[topopt][filter][Serial]")
{
  SECTION("T2.3 - Helmholtz smoothing")
  {
    auto serial_mesh = std::make_unique<mfem::Mesh>(
        mfem::Mesh::MakeCartesian3D(24, 1, 1, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0));
    auto par_mesh = std::make_unique<mfem::ParMesh>(Mpi::World(), *serial_mesh);

    const double h = 1.0 / 24.0;
    topopt::HelmholtzFilter filter(*par_mesh, 1, 3.0 * h);

    mfem::Vector rho(par_mesh->GetNE());
    for (int i = 0; i < par_mesh->GetNE(); i++)
    {
      rho(i) = (ElementCenterX(*par_mesh, i) >= 0.5) ? 1.0 : 0.0;
    }

    mfem::Vector rho_tilde;
    filter.Filter(rho, rho_tilde);
    const auto samples = SampleElementCenters(*par_mesh, filter, rho_tilde);

    REQUIRE(samples.front().second < 0.08);
    REQUIRE(samples.back().second > 0.92);
    for (std::size_t i = 1; i < samples.size(); i++)
    {
      REQUIRE(samples[i - 1].second <= samples[i].second + 1.0e-10);
    }

    int transition = 0;
    for (const auto &[x, value] : samples)
    {
      (void)x;
      if (value > 0.1 && value < 0.9)
      {
        transition++;
      }
    }
    REQUIRE(transition >= 4);
    REQUIRE(transition <= 12);
  }

  SECTION("T2.4 - Filter self-adjointness")
  {
    auto serial_mesh = std::make_unique<mfem::Mesh>(
        mfem::Mesh::MakeCartesian3D(8, 1, 1, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0));
    auto par_mesh = std::make_unique<mfem::ParMesh>(Mpi::World(), *serial_mesh);

    const double h = 1.0 / 8.0;
    topopt::HelmholtzFilter filter(*par_mesh, 1, 2.0 * h);

    mfem::Vector rho_a(par_mesh->GetNE());
    mfem::Vector rho_b(par_mesh->GetNE());
    for (int i = 0; i < par_mesh->GetNE(); i++)
    {
      const double x = ElementCenterX(*par_mesh, i);
      rho_a(i) = x;
      rho_b(i) = 1.0 - 0.5 * x;
    }

    mfem::Vector b_a, b_b, x_a, x_b;
    filter.AssembleElementRHS(rho_a, b_a);
    filter.AssembleElementRHS(rho_b, b_b);
    filter.Solve(b_a, x_a);
    filter.Solve(b_b, x_b);

    REQUIRE(Dot(b_a, x_a) > 0.0);
    REQUIRE(Dot(b_a, x_b) == Approx(Dot(x_a, b_b)).margin(1.0e-10));
  }
}

}  // namespace palace
