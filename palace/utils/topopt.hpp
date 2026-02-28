// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef PALACE_UTILS_TOPOPT_HPP
#define PALACE_UTILS_TOPOPT_HPP

#include <cmath>
#include <memory>
#include <mfem.hpp>

namespace palace::topopt
{

inline double InterpolateRefractiveIndex(double rho_hat, double n_low, double n_high)
{
  return n_low + rho_hat * (n_high - n_low);
}

inline double InterpolatePermittivityNSquared(double rho_hat, double n_low, double n_high)
{
  const double n = InterpolateRefractiveIndex(rho_hat, n_low, n_high);
  return n * n;
}

inline double InterpolatePermittivityNSquaredDerivative(double rho_hat, double n_low,
                                                        double n_high)
{
  return 2.0 * InterpolateRefractiveIndex(rho_hat, n_low, n_high) * (n_high - n_low);
}

inline double InterpolateConductivityLogLinear(double rho_hat, double sigma_d, double sigma_m,
                                               double sigma_0)
{
  MFEM_VERIFY(sigma_d > 0.0 && sigma_m > 0.0 && sigma_0 > 0.0,
              "Log-linear conductivity interpolation requires positive conductivities!");
  const double log_sigma = std::log10(sigma_d / sigma_0) +
                           rho_hat * (std::log10(sigma_m / sigma_0) -
                                      std::log10(sigma_d / sigma_0));
  return sigma_0 * std::pow(10.0, log_sigma);
}

inline double InterpolateConductivityLogLinearDerivative(double rho_hat, double sigma_d,
                                                         double sigma_m, double sigma_0)
{
  const double sigma =
      InterpolateConductivityLogLinear(rho_hat, sigma_d, sigma_m, sigma_0);
  const double log_span =
      std::log10(sigma_m / sigma_0) - std::log10(sigma_d / sigma_0);
  return sigma * std::log(10.0) * log_span;
}

inline double HeavisideProjection(double rho_tilde, double beta, double eta)
{
  const double denom = std::tanh(beta * eta) + std::tanh(beta * (1.0 - eta));
  MFEM_VERIFY(std::isfinite(denom) && std::abs(denom) > 0.0,
              "Invalid Heaviside projection denominator!");
  return (std::tanh(beta * eta) + std::tanh(beta * (rho_tilde - eta))) / denom;
}

inline double HeavisideProjectionDerivative(double rho_tilde, double beta, double eta)
{
  const double denom = std::tanh(beta * eta) + std::tanh(beta * (1.0 - eta));
  MFEM_VERIFY(std::isfinite(denom) && std::abs(denom) > 0.0,
              "Invalid Heaviside projection denominator!");
  const double t = std::tanh(beta * (rho_tilde - eta));
  return beta * (1.0 - t * t) / denom;
}

inline double VolumeFraction(const mfem::Vector &rho, const mfem::Vector &element_volume)
{
  MFEM_VERIFY(rho.Size() == element_volume.Size(), "VolumeFraction size mismatch!");
  double total_volume = 0.0;
  double weighted_rho = 0.0;
  for (int i = 0; i < rho.Size(); i++)
  {
    total_volume += element_volume(i);
    weighted_rho += rho(i) * element_volume(i);
  }
  MFEM_VERIFY(total_volume > 0.0, "VolumeFraction requires positive total volume!");
  return weighted_rho / total_volume;
}

inline mfem::Vector VolumeFractionGradient(const mfem::Vector &element_volume)
{
  double total_volume = 0.0;
  for (int i = 0; i < element_volume.Size(); i++)
  {
    total_volume += element_volume(i);
  }
  MFEM_VERIFY(total_volume > 0.0,
              "VolumeFractionGradient requires positive total volume!");

  mfem::Vector grad(element_volume.Size());
  for (int i = 0; i < element_volume.Size(); i++)
  {
    grad(i) = element_volume(i) / total_volume;
  }
  return grad;
}

inline double BinarizationMeasure(const mfem::Vector &rho_hat,
                                  const mfem::Vector &element_volume)
{
  MFEM_VERIFY(rho_hat.Size() == element_volume.Size(),
              "BinarizationMeasure size mismatch!");
  double total_volume = 0.0;
  double grayness = 0.0;
  for (int i = 0; i < rho_hat.Size(); i++)
  {
    total_volume += element_volume(i);
    grayness += rho_hat(i) * (1.0 - rho_hat(i)) * element_volume(i);
  }
  MFEM_VERIFY(total_volume > 0.0, "BinarizationMeasure requires positive total volume!");
  return 4.0 * grayness / total_volume;
}

namespace internal
{

class ElementwiseCoefficient : public mfem::Coefficient
{
private:
  const mfem::Vector &element_value;

public:
  explicit ElementwiseCoefficient(const mfem::Vector &element_value)
    : element_value(element_value)
  {
  }

  double Eval(mfem::ElementTransformation &T,
              const mfem::IntegrationPoint &) override
  {
    const int elem = T.ElementNo;
    MFEM_ASSERT(elem >= 0 && elem < element_value.Size(), "Element index out of range!");
    return element_value(elem);
  }
};

}  // namespace internal

class HelmholtzFilter
{
private:
  std::unique_ptr<mfem::H1_FECollection> fec;
  std::unique_ptr<mfem::ParFiniteElementSpace> fespace;
  std::unique_ptr<mfem::HypreParMatrix> A;
  std::unique_ptr<mfem::HypreBoomerAMG> pc;
  mutable mfem::CGSolver solver;

public:
  HelmholtzFilter(mfem::ParMesh &mesh, int order, double radius)
    : fec(std::make_unique<mfem::H1_FECollection>(order, mesh.Dimension())),
      fespace(std::make_unique<mfem::ParFiniteElementSpace>(&mesh, fec.get())),
      solver(mesh.GetComm())
  {
    MFEM_VERIFY(order >= 1, "HelmholtzFilter requires order >= 1!");
    MFEM_VERIFY(std::isfinite(radius) && radius > 0.0,
                "HelmholtzFilter requires a positive finite radius!");

    const double radius_sq = radius * radius;
    mfem::ConstantCoefficient one(1.0);
    mfem::ConstantCoefficient diffusion(radius_sq);
    mfem::ParBilinearForm a(fespace.get());
    a.AddDomainIntegrator(new mfem::DiffusionIntegrator(diffusion));
    a.AddDomainIntegrator(new mfem::MassIntegrator(one));
    a.Assemble();
    a.Finalize();

    A.reset(a.ParallelAssemble());
    pc = std::make_unique<mfem::HypreBoomerAMG>(*A);
    pc->SetPrintLevel(0);

    solver.iterative_mode = false;
    solver.SetPrintLevel(0);
    solver.SetRelTol(1.0e-12);
    solver.SetAbsTol(0.0);
    solver.SetMaxIter(200);
    solver.SetPreconditioner(*pc);
    solver.SetOperator(*A);
  }

  const mfem::ParFiniteElementSpace &GetFESpace() const { return *fespace; }
  const mfem::HypreParMatrix &GetMatrix() const { return *A; }

  void AssembleElementRHS(const mfem::Vector &element_value, mfem::Vector &rhs) const
  {
    const mfem::ParMesh *mesh = fespace->GetParMesh();
    MFEM_VERIFY(element_value.Size() == mesh->GetNE(),
                "HelmholtzFilter element RHS size mismatch!");

    internal::ElementwiseCoefficient density(element_value);
    mfem::ParLinearForm b(fespace.get());
    b.AddDomainIntegrator(new mfem::DomainLFIntegrator(density));
    b.Assemble();

    rhs.SetSize(fespace->GetTrueVSize());
    rhs.UseDevice(true);
    b.ParallelAssemble(rhs);
  }

  void Solve(const mfem::Vector &rhs, mfem::Vector &solution) const
  {
    MFEM_VERIFY(rhs.Size() == fespace->GetTrueVSize(),
                "HelmholtzFilter nodal RHS size mismatch!");
    solution.SetSize(fespace->GetTrueVSize());
    solution.UseDevice(true);
    solution = 0.0;
    solver.Mult(rhs, solution);
  }

  void Filter(const mfem::Vector &element_value, mfem::Vector &rho_tilde) const
  {
    mfem::Vector rhs;
    AssembleElementRHS(element_value, rhs);
    Solve(rhs, rho_tilde);
  }

  void AssembleElementAdjoint(const mfem::Vector &true_dofs, mfem::Vector &element_grad) const
  {
    auto *mesh = const_cast<mfem::ParMesh *>(fespace->GetParMesh());
    MFEM_VERIFY(true_dofs.Size() == fespace->GetTrueVSize(),
                "HelmholtzFilter adjoint true dof size mismatch!");

    mfem::ParGridFunction z(fespace.get());
    z.SetFromTrueDofs(true_dofs);

    mfem::ConstantCoefficient one(1.0);
    mfem::DomainLFIntegrator integ(one);
    mfem::Array<int> vdofs;
    mfem::Vector el_rhs, el_sol;

    element_grad.SetSize(mesh->GetNE());
    for (int i = 0; i < mesh->GetNE(); i++)
    {
      const auto *fe = fespace->GetFE(i);
      mfem::ElementTransformation *T = mesh->GetElementTransformation(i);
      el_rhs.SetSize(fe->GetDof());
      integ.AssembleRHSElementVect(*fe, *T, el_rhs);
      fespace->GetElementVDofs(i, vdofs);
      z.GetSubVector(vdofs, el_sol);

      double dot = 0.0;
      for (int j = 0; j < el_rhs.Size(); j++)
      {
        dot += el_rhs(j) * el_sol(j);
      }
      element_grad(i) = dot;
    }
  }

  void FilterAdjoint(const mfem::Vector &true_rhs, mfem::Vector &element_grad) const
  {
    mfem::Vector z_true;
    Solve(true_rhs, z_true);
    AssembleElementAdjoint(z_true, element_grad);
  }
};

class DesignLayout
{
private:
  mfem::Array<int> local_elements;
  mfem::Array<int> attributes;

public:
  DesignLayout() = default;

  explicit DesignLayout(const mfem::Array<int> &attributes) { SetSequential(attributes); }

  DesignLayout(const mfem::Array<int> &local_elements, const mfem::Array<int> &attributes)
  {
    Set(local_elements, attributes);
  }

  void SetSequential(const mfem::Array<int> &attributes_in)
  {
    attributes = attributes_in;
    local_elements.SetSize(attributes.Size());
    for (int i = 0; i < local_elements.Size(); i++)
    {
      local_elements[i] = i;
    }
  }

  void Set(const mfem::Array<int> &local_elements_in, const mfem::Array<int> &attributes_in)
  {
    MFEM_VERIFY(local_elements_in.Size() == attributes_in.Size(),
                "TopOpt design layout size mismatch!");
    local_elements = local_elements_in;
    attributes = attributes_in;
  }

  int Size() const { return attributes.Size(); }

  const mfem::Array<int> &GetLocalElements() const { return local_elements; }
  const mfem::Array<int> &GetAttributes() const { return attributes; }
};

inline void SampleElementCenters(const mfem::ParFiniteElementSpace &fespace,
                                 const mfem::Vector &true_dofs, mfem::Vector &elem_value)
{
  auto *mesh = const_cast<mfem::ParMesh *>(fespace.GetParMesh());
  MFEM_VERIFY(mesh, "TopOpt sampling requires a parallel mesh!");
  MFEM_VERIFY(true_dofs.Size() == fespace.GetTrueVSize(),
              "TopOpt sampling true dof size mismatch!");

  mfem::ParGridFunction gf(const_cast<mfem::ParFiniteElementSpace *>(&fespace));
  gf.SetFromTrueDofs(true_dofs);

  elem_value.SetSize(mesh->GetNE());
  for (int i = 0; i < mesh->GetNE(); i++)
  {
    mfem::ElementTransformation *T = mesh->GetElementTransformation(i);
    const mfem::IntegrationPoint &ip = mfem::Geometries.GetCenter(T->GetGeometryType());
    elem_value(i) = gf.GetValue(i, ip);
  }
}

inline void SampleElementCenters(const mfem::ParFiniteElementSpace &fespace,
                                 const mfem::Vector &true_dofs,
                                 const DesignLayout &layout, mfem::Vector &elem_value)
{
  auto *mesh = const_cast<mfem::ParMesh *>(fespace.GetParMesh());
  MFEM_VERIFY(mesh, "TopOpt sampling requires a parallel mesh!");
  MFEM_VERIFY(true_dofs.Size() == fespace.GetTrueVSize(),
              "TopOpt sampling true dof size mismatch!");

  mfem::ParGridFunction gf(const_cast<mfem::ParFiniteElementSpace *>(&fespace));
  gf.SetFromTrueDofs(true_dofs);

  elem_value.SetSize(layout.Size());
  for (int i = 0; i < layout.Size(); i++)
  {
    const int elem = layout.GetLocalElements()[i];
    MFEM_VERIFY(elem >= 0 && elem < mesh->GetNE(), "TopOpt design element index out of range!");
    mfem::ElementTransformation *T = mesh->GetElementTransformation(elem);
    const mfem::IntegrationPoint &ip = mfem::Geometries.GetCenter(T->GetGeometryType());
    elem_value(i) = gf.GetValue(elem, ip);
  }
}

inline void AssembleElementCenterAdjoint(const mfem::ParFiniteElementSpace &fespace,
                                         const DesignLayout &layout,
                                         const mfem::Vector &elem_grad,
                                         mfem::Vector &true_rhs)
{
  auto *mesh = const_cast<mfem::ParMesh *>(fespace.GetParMesh());
  MFEM_VERIFY(mesh, "TopOpt sampling adjoint requires a parallel mesh!");
  MFEM_VERIFY(elem_grad.Size() == layout.Size(),
              "TopOpt sampling adjoint size mismatch!");

  mfem::ParGridFunction gf(const_cast<mfem::ParFiniteElementSpace *>(&fespace));
  gf = 0.0;

  mfem::Array<int> vdofs;
  mfem::Vector shape;
  for (int i = 0; i < layout.Size(); i++)
  {
    const int elem = layout.GetLocalElements()[i];
    MFEM_VERIFY(elem >= 0 && elem < mesh->GetNE(), "TopOpt design element index out of range!");
    const auto *fe = fespace.GetFE(elem);
    mfem::ElementTransformation *T = mesh->GetElementTransformation(elem);
    const mfem::IntegrationPoint &ip = mfem::Geometries.GetCenter(T->GetGeometryType());
    shape.SetSize(fe->GetDof());
    fe->CalcShape(ip, shape);
    fespace.GetElementVDofs(elem, vdofs);
    for (int j = 0; j < vdofs.Size(); j++)
    {
      gf(vdofs[j]) += elem_grad(i) * shape(j);
    }
  }

  true_rhs.SetSize(fespace.GetTrueVSize());
  gf.ParallelAssemble(true_rhs);
}

class DesignState
{
private:
  mfem::Vector rho;
  mfem::Vector rho_tilde_true;
  mfem::Vector rho_tilde_elem;
  mfem::Vector rho_hat;

public:
  DesignState() = default;
  explicit DesignState(int local_ne) { Resize(local_ne); }

  void Resize(int local_ne)
  {
    MFEM_VERIFY(local_ne >= 0, "TopOpt design state size must be non-negative!");
    rho.SetSize(local_ne);
    rho = 0.0;
    rho_tilde_true.SetSize(0);
    rho_tilde_elem.SetSize(local_ne);
    rho_tilde_elem = 0.0;
    rho_hat.SetSize(local_ne);
    rho_hat = 0.0;
  }

  auto &GetDensity() { return rho; }
  const auto &GetDensity() const { return rho; }
  const auto &GetFilteredTrueDofs() const { return rho_tilde_true; }
  const auto &GetFilteredDensity() const { return rho_tilde_elem; }
  const auto &GetProjectedDensity() const { return rho_hat; }

  void SetDensity(const mfem::Vector &rho_in)
  {
    rho.SetSize(rho_in.Size());
    rho = rho_in;
    if (rho_tilde_elem.Size() != rho.Size())
    {
      rho_tilde_elem.SetSize(rho.Size());
      rho_hat.SetSize(rho.Size());
    }
  }

  void SetUniformDensity(double value)
  {
    rho = value;
  }

  void ApplyFilter(const HelmholtzFilter &filter)
  {
    MFEM_VERIFY(rho.Size() == filter.GetFESpace().GetParMesh()->GetNE(),
                "TopOpt filter input size mismatch!");
    filter.Filter(rho, rho_tilde_true);
    SampleElementCenters(filter.GetFESpace(), rho_tilde_true, rho_tilde_elem);
  }

  void ApplyFilter(const HelmholtzFilter &filter, const DesignLayout &layout,
                   double background_density = 0.0)
  {
    MFEM_VERIFY(rho.Size() == layout.Size(),
                "TopOpt design state and layout size mismatch!");
    MFEM_VERIFY(layout.Size() <= filter.GetFESpace().GetParMesh()->GetNE(),
                "TopOpt layout is larger than local mesh!");
    mfem::Vector rho_local(filter.GetFESpace().GetParMesh()->GetNE());
    rho_local = background_density;
    for (int i = 0; i < layout.Size(); i++)
    {
      const int elem = layout.GetLocalElements()[i];
      MFEM_VERIFY(elem >= 0 && elem < rho_local.Size(),
                  "TopOpt design element index out of range!");
      rho_local(elem) = rho(i);
    }
    filter.Filter(rho_local, rho_tilde_true);
    SampleElementCenters(filter.GetFESpace(), rho_tilde_true, layout, rho_tilde_elem);
  }

  void ApplyProjection(double beta, double eta)
  {
    MFEM_VERIFY(rho_hat.Size() == rho_tilde_elem.Size(),
                "TopOpt projection input size mismatch!");
    for (int i = 0; i < rho_tilde_elem.Size(); i++)
    {
      rho_hat(i) = HeavisideProjection(rho_tilde_elem(i), beta, eta);
    }
  }

  void ApplyProjectionAdjoint(const mfem::Vector &grad_rho_hat, double beta, double eta,
                              mfem::Vector &grad_rho_tilde) const
  {
    MFEM_VERIFY(grad_rho_hat.Size() == rho_tilde_elem.Size(),
                "TopOpt projection adjoint size mismatch!");
    grad_rho_tilde.SetSize(rho_tilde_elem.Size());
    for (int i = 0; i < rho_tilde_elem.Size(); i++)
    {
      grad_rho_tilde(i) =
          grad_rho_hat(i) * HeavisideProjectionDerivative(rho_tilde_elem(i), beta, eta);
    }
  }

  void ApplyFilterAndProjection(const HelmholtzFilter &filter, double beta, double eta)
  {
    ApplyFilter(filter);
    ApplyProjection(beta, eta);
  }

  void ApplyFilterAndProjection(const HelmholtzFilter &filter, const DesignLayout &layout,
                                double beta, double eta,
                                double background_density = 0.0)
  {
    ApplyFilter(filter, layout, background_density);
    ApplyProjection(beta, eta);
  }
};

}  // namespace palace::topopt

#endif  // PALACE_UTILS_TOPOPT_HPP
