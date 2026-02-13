// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "surfaceconductivityoperator.hpp"

#include <set>
#include "models/materialoperator.hpp"
#include "utils/communication.hpp"
#include "utils/geodata.hpp"
#include "utils/iodata.hpp"
#include "utils/prettyprint.hpp"

namespace palace
{

using namespace std::complex_literals;

SurfaceConductivityOperator::SurfaceConductivityOperator(const IoData &iodata,
                                                         const MaterialOperator &mat_op,
                                                         const mfem::ParMesh &mesh)
  : mat_op(mat_op)
{
  // Print out BC info for all finite conductivity boundary attributes.
  SetUpBoundaryProperties(iodata, mesh);
  PrintBoundaryInfo(iodata, mesh);
}

void SurfaceConductivityOperator::SetUpBoundaryProperties(const IoData &iodata,
                                                          const mfem::ParMesh &mesh)
{
  // Check that conductivity boundary attributes have been specified correctly.
  int bdr_attr_max = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
  mfem::Array<int> bdr_attr_marker;
  if (!iodata.boundaries.conductivity.empty())
  {
    mfem::Array<int> conductivity_marker(bdr_attr_max);
    bdr_attr_marker.SetSize(bdr_attr_max);
    bdr_attr_marker = 0;
    conductivity_marker = 0;
    for (auto attr : mesh.bdr_attributes)
    {
      bdr_attr_marker[attr - 1] = 1;
    }
    std::set<int> bdr_warn_list;
    for (const auto &data : iodata.boundaries.conductivity)
    {
      for (auto attr : data.attributes)
      {
        if (attr <= 0 || attr > bdr_attr_max)
        {
          bdr_warn_list.insert(attr);
          continue;
        }
        MFEM_VERIFY(!conductivity_marker[attr - 1],
                    "Multiple definitions of conductivity boundary properties for boundary "
                    "attribute "
                        << attr << "!");
        conductivity_marker[attr - 1] = 1;
        // MFEM_VERIFY(bdr_attr_marker[attr - 1],
        //             "Unknown conductivity boundary attribute " << attr << "!");
        if (!bdr_attr_marker[attr - 1])
        {
          bdr_warn_list.insert(attr);
        }
      }
    }
    if (!bdr_warn_list.empty())
    {
      Mpi::Print("\n");
      Mpi::Warning(
          "Unknown conductivity boundary attributes!\nSolver will just ignore them!");
      utils::PrettyPrint(bdr_warn_list, "Boundary attribute list:");
      Mpi::Print("\n");
    }
  }

  // Finite conductivity boundaries are defined using the user provided surface conductivity
  // and optionally conductor thickness.
  boundaries.reserve(iodata.boundaries.conductivity.size());
  for (const auto &data : iodata.boundaries.conductivity)
  {
    MFEM_VERIFY(data.sigma > 0.0 && data.mu_r > 0.0,
                "Conductivity boundary has no conductivity or no "
                "permeability defined!");
    MFEM_VERIFY(data.h >= 0.0, "Conductivity boundary should have non-negative thickness!");
    auto &bdr = boundaries.emplace_back();
    bdr.sigma = data.sigma;
    bdr.mu = data.mu_r;
    bdr.h = data.h;
    if (data.external)
    {
      // External surfaces have twice the effective thickness since the BC is applied at one
      // side.
      bdr.h *= 2.0;
    }
    bdr.attr_list.Reserve(static_cast<int>(data.attributes.size()));
    for (auto attr : data.attributes)
    {
      if (attr <= 0 || attr > bdr_attr_max || !bdr_attr_marker[attr - 1])
      {
        continue;  // Can just ignore if wrong
      }
      bdr.attr_list.Append(attr);
    }
  }
  MFEM_VERIFY(boundaries.empty() || iodata.problem.type == ProblemType::DRIVEN ||
                  iodata.problem.type == ProblemType::EIGENMODE,
              "Finite conductivity boundaries are only available for frequency "
              "domain simulations!");
}

void SurfaceConductivityOperator::PrintBoundaryInfo(const IoData &iodata,
                                                    const mfem::ParMesh &mesh)
{
  if (boundaries.empty())
  {
    return;
  }
  Mpi::Print("\nConfiguring Robin finite conductivity BC at attributes:\n");
  for (const auto &bdr : boundaries)
  {
    for (auto attr : bdr.attr_list)
    {
      Mpi::Print(" {:d}: σ = {:.3e} S/m", attr,
                 iodata.units.Dimensionalize<Units::ValueType::CONDUCTIVITY>(bdr.sigma));
      if (bdr.h > 0.0)
      {
        Mpi::Print(", h = {:.3e} m",
                   iodata.units.Dimensionalize<Units::ValueType::LENGTH>(bdr.h));
      }
      Mpi::Print(", n = ({:+.1f})\n", fmt::join(mesh::GetSurfaceNormal(mesh, attr), ","));
    }
  }
}

mfem::Array<int> SurfaceConductivityOperator::GetAttrList() const
{
  mfem::Array<int> attr_list;
  for (const auto &bdr : boundaries)
  {
    attr_list.Append(bdr.attr_list);
  }
  return attr_list;
}

void SurfaceConductivityOperator::AddExtraSystemBdrCoefficients(
    double omega, MaterialPropertyCoefficient &fbr, MaterialPropertyCoefficient &fbi)
{
  // If the provided conductor thickness is empty (zero), prescribe a surface impedance
  // (1+i)/σδ, where δ is the skin depth. If it is nonzero, use a finite thickness
  // modification which correctly produces the DC limit when h << δ. See the Ansys HFSS
  // user manual section titled "Surface Impedance Boundary Condition for Metal Traces of
  // Finite Thickness."
  for (const auto &bdr : boundaries)
  {
    if (std::abs(bdr.sigma) > 0.0)
    {
      double delta = std::sqrt(2.0 / (bdr.mu * bdr.sigma * omega));
      std::complex<double> Z = 1.0 / (bdr.sigma * delta);
      Z.imag(Z.real());
      if (bdr.h > 0.0)
      {
        double nu = bdr.h / delta;
        double ratio_re, ratio_im;
        if (nu < 0.1)
        {
          // Use a series expansion for small nu to avoid catastrophic cancellation in
          // cosh(nu) - cos(nu) and sinh(nu) - sin(nu).
          const double nu2 = nu * nu;
          const double nu3 = nu2 * nu;
          const double nu4 = nu2 * nu2;
          const double nu5 = nu4 * nu;
          const double nu6 = nu3 * nu3;
          const double nu7 = nu6 * nu;
          const double den = nu2 + nu6 / 360.0;
          ratio_re = (2.0 * nu + nu5 / 60.0) / den;
          ratio_im = (nu3 / 3.0 + nu7 / 2520.0) / den;
        }
        else
        {
          const double den = std::cosh(nu) - std::cos(nu);
          ratio_re = (std::sinh(nu) + std::sin(nu)) / den;
          ratio_im = (std::sinh(nu) - std::sin(nu)) / den;
        }
        Z.real(Z.real() * ratio_re);
        Z.imag(Z.imag() * ratio_im);
      }
      // The BC term has coefficient iω/Z (like for standard lumped surface impedance).
      std::complex<double> s(1i * omega / Z);
      fbr.AddMaterialProperty(mat_op.GetCeedBdrAttributes(bdr.attr_list), s.real());
      fbi.AddMaterialProperty(mat_op.GetCeedBdrAttributes(bdr.attr_list), s.imag());
    }
  }
}

}  // namespace palace
