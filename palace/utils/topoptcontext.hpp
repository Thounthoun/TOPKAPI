// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef PALACE_UTILS_TOPOPTCONTEXT_HPP
#define PALACE_UTILS_TOPOPTCONTEXT_HPP

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "models/spaceoperator.hpp"
#include "utils/iodata.hpp"
#include "utils/topopt.hpp"

namespace palace::topopt
{

struct TopOptEvaluation
{
  double objective = 0.0;
  double constraint = 0.0;
  mfem::Vector objective_gradient;
  mfem::Vector constraint_gradient;
};

struct TopOptIterationHistory
{
  mfem::Vector objective;
  mfem::Vector constraint;
};

struct TopOptRunOptions
{
  int max_steps = 0;
  double step_size = 0.0;
  double gradient_tolerance = 0.0;
  double lower_bound = 0.0;
  double upper_bound = 1.0;
};

struct TopOptRunSummary
{
  TopOptIterationHistory history;
  mfem::Vector gradient_norm;
  int num_iterations = 0;
  bool converged = false;
};

struct TopOptPenaltyRunSummary
{
  TopOptIterationHistory history;
  mfem::Vector merit;
  mfem::Vector accepted_step;
  mfem::Vector accepted;
  int num_iterations = 0;
  int num_accepted = 0;
  int num_rejected = 0;
  bool converged = false;
  bool stopped_on_reject_limit = false;
};

struct TopOptPenaltyRunOptions
{
  int max_steps = 0;
  double initial_step = 0.0;
  double penalty = 0.0;
  int max_backtracks = 8;
  double backtrack_factor = 0.5;
  double lower_bound = 0.0;
  double upper_bound = 1.0;
  double merit_tolerance = 0.0;
  int max_consecutive_rejects = 0;
};

enum class TopOptSessionMode
{
  OBJECTIVE,
  PENALTY
};

struct TopOptSessionOptions
{
  TopOptSessionMode mode = TopOptSessionMode::OBJECTIVE;
  TopOptRunOptions objective;
  TopOptPenaltyRunOptions penalty;
};

struct TopOptSessionSummary
{
  TopOptSessionMode mode = TopOptSessionMode::OBJECTIVE;
  TopOptIterationHistory history;
  mfem::Vector trace_value;
  mfem::Vector accepted_step;
  mfem::Vector accepted;
  int num_iterations = 0;
  int num_accepted = 0;
  int num_rejected = 0;
  bool converged = false;
  bool stopped_on_reject_limit = false;
};

struct TopOptDriverReport
{
  TopOptSessionMode mode = TopOptSessionMode::OBJECTIVE;
  double objective = 0.0;
  double constraint = 0.0;
  double trace_value = 0.0;
  int num_iterations = 0;
  int num_accepted = 0;
  int num_rejected = 0;
  bool converged = false;
  bool stopped_on_reject_limit = false;
  int num_recorded_runs = 0;

  static const char *ModeString(TopOptSessionMode mode)
  {
    return (mode == TopOptSessionMode::OBJECTIVE) ? "objective" : "penalty";
  }

  static std::string CsvHeader()
  {
    return "mode,objective,constraint,trace_value,num_iterations,num_accepted,"
           "num_rejected,converged,stopped_on_reject_limit,num_recorded_runs";
  }

  std::string ToCsvRow() const
  {
    std::ostringstream os;
    os << ModeString(mode) << ',' << objective << ',' << constraint << ',' << trace_value
       << ',' << num_iterations << ',' << num_accepted << ',' << num_rejected << ','
       << (converged ? 1 : 0) << ',' << (stopped_on_reject_limit ? 1 : 0) << ','
       << num_recorded_runs;
    return os.str();
  }
};

class TopOptContext
{
private:
  IoData &iodata;
  SpaceOperator &space_op;
  const HelmholtzFilter &filter;
  DesignLayout layout;
  DesignState state;
  double background_density;

  template <typename ObjectiveFunc>
  void ComputeProjectedFiniteDifferenceGradient(ObjectiveFunc &&objective, double n_low,
                                                double n_high, mfem::Vector &dJ_drho_hat,
                                                double delta = 1.0e-5)
  {
    const mfem::Vector rho_hat_base = state.GetProjectedDensity();

    dJ_drho_hat.SetSize(layout.Size());
    for (int i = 0; i < layout.Size(); i++)
    {
      mfem::Vector rho_hat_plus(rho_hat_base), rho_hat_minus(rho_hat_base);
      rho_hat_plus(i) += delta;
      rho_hat_minus(i) -= delta;

      UpdatePermittivityFromProjectedDensity(rho_hat_plus, n_low, n_high);
      const double j_plus = objective();
      UpdatePermittivityFromProjectedDensity(rho_hat_minus, n_low, n_high);
      const double j_minus = objective();
      dJ_drho_hat(i) = (j_plus - j_minus) / (2.0 * delta);
    }
    UpdatePermittivityFromProjectedDensity(rho_hat_base, n_low, n_high);
  }

public:
  TopOptContext(IoData &iodata, SpaceOperator &space_op, const DesignLayout &layout,
                const HelmholtzFilter &filter, double background_density = 0.0)
    : iodata(iodata), space_op(space_op), filter(filter), layout(layout),
      state(layout.Size()), background_density(background_density)
  {
  }

  auto &GetState() { return state; }
  const auto &GetState() const { return state; }
  const auto &GetLayout() const { return layout; }
  MPI_Comm GetComm() const { return space_op.GetComm(); }

  void SetBackgroundDensity(double rho_background) { background_density = rho_background; }
  double GetBackgroundDensity() const { return background_density; }

  void UpdatePermittivityFromDensity(const mfem::Vector &rho, double beta, double eta,
                                     double n_low, double n_high)
  {
    UpdateState(rho, beta, eta);
    UpdatePermittivityFromProjectedDensity(state.GetProjectedDensity(), n_low, n_high);
  }

  void UpdatePermittivityFromUniformDensity(double rho, double beta, double eta, double n_low,
                                            double n_high)
  {
    UpdateStateFromUniformDensity(rho, beta, eta);
    UpdatePermittivityFromProjectedDensity(state.GetProjectedDensity(), n_low, n_high);
  }

  void UpdatePermittivityFromProjectedDensity(const mfem::Vector &rho_hat, double n_low,
                                              double n_high)
  {
    MFEM_VERIFY(rho_hat.Size() == layout.Size(),
                "TopOpt projected density and layout size mismatch!");
    space_op.GetMaterialOp().UpdatePermittivityNSquared(layout.GetAttributes(),
                                                        rho_hat, n_low, n_high);
    // Reinitialize global quadrature defaults in case another test/config changed them.
    iodata.CheckConfiguration();
  }

  void UpdateState(const mfem::Vector &rho, double beta, double eta)
  {
    state.SetDensity(rho);
    state.ApplyFilterAndProjection(filter, layout, beta, eta, background_density);
    // Reinitialize global quadrature defaults in case another test/config changed them.
    iodata.CheckConfiguration();
  }

  void UpdateStateFromUniformDensity(double rho, double beta, double eta)
  {
    state.Resize(layout.Size());
    state.SetUniformDensity(rho);
    state.ApplyFilterAndProjection(filter, layout, beta, eta, background_density);
    iodata.CheckConfiguration();
  }

  void ApplyCurrentSystemMatrix(double omega, const mfem::Vector &x, mfem::Vector &y) const
  {
    auto K = space_op.GetStiffnessMatrix<Operator>(Operator::DIAG_ZERO);
    auto M = space_op.GetMassMatrix<Operator>(Operator::DIAG_ZERO);
    auto A = space_op.GetSystemMatrix<Operator, double>(1.0, 0.0, -omega * omega, K.get(),
                                                        nullptr, M.get());
    MFEM_VERIFY(x.Size() == A->Width(), "TopOpt system apply vector size mismatch!");

    y.SetSize(A->Height());
    y = 0.0;
    A->Mult(x, y);
  }

  void SolveCurrentSystem(double omega, const mfem::Vector &rhs, mfem::Vector &u) const
  {
    auto K = space_op.GetStiffnessMatrix<Operator>(Operator::DIAG_ZERO);
    auto M = space_op.GetMassMatrix<Operator>(Operator::DIAG_ZERO);
    auto A = space_op.GetSystemMatrix<Operator, double>(1.0, 0.0, -omega * omega, K.get(),
                                                        nullptr, M.get());
    MFEM_VERIFY(rhs.Size() == A->Height(), "TopOpt proxy solve RHS size mismatch!");

    mfem::GMRESSolver gmres(space_op.GetComm());
    gmres.iterative_mode = false;
    gmres.SetPrintLevel(0);
    gmres.SetRelTol(1.0e-12);
    gmres.SetAbsTol(0.0);
    gmres.SetMaxIter(500);
    gmres.SetOperator(*A);

    u.SetSize(A->Width());
    u = 0.0;
    gmres.Mult(rhs, u);
  }

  double EvaluateResponseNorm(double omega) const
  {
    Vector x(space_op.GetNDSpace().GetTrueVSize()), y;
    x = 1.0;
    ApplyCurrentSystemMatrix(omega, x, y);
    return linalg::Norml2(space_op.GetComm(), y);
  }

  double EvaluateResponseNormFromProjectedDensity(const mfem::Vector &rho_hat, double omega,
                                                  double n_low, double n_high)
  {
    UpdatePermittivityFromProjectedDensity(rho_hat, n_low, n_high);
    return EvaluateResponseNorm(omega);
  }

  void SolveProxyState(double omega, const mfem::Vector &rhs, mfem::Vector &u) const
  {
    SolveCurrentSystem(omega, rhs, u);
  }

  double EvaluateLinearSolvedObjective(double omega, const mfem::Vector &rhs,
                                       const mfem::Vector &weights) const
  {
    mfem::Vector u;
    SolveProxyState(omega, rhs, u);
    return EvaluateLinearSolvedObjectiveFromState(u, weights);
  }

  double EvaluateLinearSolvedObjectiveFromState(const mfem::Vector &u,
                                                const mfem::Vector &weights) const
  {
    MFEM_VERIFY(weights.Size() == u.Size(),
                "TopOpt proxy objective weight size mismatch!");
    return linalg::Dot(space_op.GetComm(), weights, u);
  }

  double EvaluateLinearSolvedObjectiveFromProjectedDensity(const mfem::Vector &rho_hat,
                                                           double omega,
                                                           const mfem::Vector &rhs,
                                                           const mfem::Vector &weights,
                                                           double n_low, double n_high)
  {
    UpdatePermittivityFromProjectedDensity(rho_hat, n_low, n_high);
    return EvaluateLinearSolvedObjective(omega, rhs, weights);
  }

  double EvaluateStoredEnergyObjective(double omega, const mfem::Vector &rhs) const
  {
    mfem::Vector u;
    SolveProxyState(omega, rhs, u);
    return EvaluateStoredEnergyObjectiveFromState(u);
  }

  double EvaluateMassObjectiveFromStates(const mfem::Vector &left_state,
                                         const mfem::Vector &right_state,
                                         double mass_scale) const
  {
    const auto nd_true_size = space_op.GetNDSpace().GetTrueVSize();
    MFEM_VERIFY(left_state.Size() == nd_true_size && right_state.Size() == nd_true_size,
                "TopOpt mass objective state size mismatch!");
    MFEM_VERIFY(std::isfinite(mass_scale),
                "TopOpt mass objective scale must be finite!");

    mfem::Vector Mr(right_state);
    auto M = space_op.GetMassMatrix<Operator>(Operator::DIAG_ZERO);
    Mr = 0.0;
    M->Mult(right_state, Mr);
    return mass_scale * linalg::Dot(space_op.GetComm(), left_state, Mr);
  }

  void ComputeMassAdjointRHSFromState(const mfem::Vector &state_vec, double mass_scale,
                                      mfem::Vector &adjoint_rhs) const
  {
    const auto nd_true_size = space_op.GetNDSpace().GetTrueVSize();
    MFEM_VERIFY(state_vec.Size() == nd_true_size,
                "TopOpt mass adjoint RHS state size mismatch!");
    MFEM_VERIFY(std::isfinite(mass_scale),
                "TopOpt mass adjoint RHS scale must be finite!");

    auto M = space_op.GetMassMatrix<Operator>(Operator::DIAG_ZERO);
    adjoint_rhs.SetSize(nd_true_size);
    adjoint_rhs = 0.0;
    M->Mult(state_vec, adjoint_rhs);
    adjoint_rhs *= mass_scale;
  }

  double EvaluateStoredEnergyObjectiveFromState(const mfem::Vector &u) const
  {
    return EvaluateMassObjectiveFromStates(u, u, 0.5);
  }

  double EvaluateStoredEnergyObjectiveFromProjectedDensity(const mfem::Vector &rho_hat,
                                                           double omega,
                                                           const mfem::Vector &rhs,
                                                           double n_low, double n_high)
  {
    UpdatePermittivityFromProjectedDensity(rho_hat, n_low, n_high);
    return EvaluateStoredEnergyObjective(omega, rhs);
  }

  double EvaluateProjectedVolumeFraction(const mfem::Vector &element_volume) const
  {
    return VolumeFraction(state.GetProjectedDensity(), element_volume);
  }

  void ComputeProjectedMassSensitivity(const mfem::Vector &state_vec,
                                       const mfem::Vector &adjoint_vec, double mass_scale,
                                       double n_low, double n_high,
                                       mfem::Vector &dJ_drho_hat)
  {
    MFEM_VERIFY(std::isfinite(mass_scale),
                "TopOpt mass sensitivity scale must be finite!");

    const auto &rho_hat = state.GetProjectedDensity();
    MFEM_VERIFY(rho_hat.Size() == layout.Size(),
                "TopOpt projected density and layout size mismatch!");
    UpdatePermittivityFromProjectedDensity(rho_hat, n_low, n_high);

    const auto &nd_fes = space_op.GetNDSpace();
    auto &mfem_nd_fes = const_cast<mfem::ParFiniteElementSpace &>(nd_fes.Get());
    MFEM_VERIFY(state_vec.Size() == nd_fes.GetTrueVSize() &&
                    adjoint_vec.Size() == nd_fes.GetTrueVSize(),
                "TopOpt state/adjoint size mismatch for projected mass sensitivity!");

    dJ_drho_hat.SetSize(layout.Size());
    dJ_drho_hat = 0.0;

    mfem::ParGridFunction state_gf(&mfem_nd_fes), adjoint_gf(&mfem_nd_fes);
    state_gf = 0.0;
    adjoint_gf = 0.0;
    state_gf.SetFromTrueDofs(state_vec);
    adjoint_gf.SetFromTrueDofs(adjoint_vec);

    // This is the reusable EM-side seam: given a primal/adjoint pair in the ND true-dof
    // space, apply the local unit-mass kernel and scale by dε/dρ̂.
    mfem::VectorFEMassIntegrator unit_mass;
    mfem::Array<int> vdofs;
    mfem::DenseMatrix elmat;
    mfem::Vector state_el, adjoint_el, work;
    for (int i = 0; i < layout.Size(); i++)
    {
      const int elem = layout.GetLocalElements()[i];
      MFEM_VERIFY(elem >= 0 && elem < mfem_nd_fes.GetNE(),
                  "TopOpt design layout local element is out of range!");

      const auto *fe = mfem_nd_fes.GetFE(elem);
      auto *trans = mfem_nd_fes.GetParMesh()->GetElementTransformation(elem);
      MFEM_VERIFY(fe != nullptr && trans != nullptr,
                  "TopOpt failed to access ND element data for mass sensitivity!");

      unit_mass.AssembleElementMatrix(*fe, *trans, elmat);

      mfem_nd_fes.GetElementVDofs(elem, vdofs);
      state_el.SetSize(vdofs.Size());
      adjoint_el.SetSize(vdofs.Size());
      work.SetSize(vdofs.Size());
      state_gf.GetSubVector(vdofs, state_el);
      adjoint_gf.GetSubVector(vdofs, adjoint_el);
      elmat.Mult(state_el, work);

      const double local_energy = adjoint_el * work;
      const double deps =
          InterpolatePermittivityNSquaredDerivative(rho_hat(i), n_low, n_high);
      dJ_drho_hat(i) = mass_scale * deps * local_energy;
    }
  }

  void ComputeDensityGradientFromStateAdjoint(const mfem::Vector &state_vec,
                                              const mfem::Vector &adjoint_vec,
                                              double mass_scale, double beta,
                                              double eta, double n_low,
                                              double n_high,
                                              mfem::Vector &dJ_drho)
  {
    mfem::Vector dJ_drho_hat;
    ComputeProjectedMassSensitivity(state_vec, adjoint_vec, mass_scale, n_low, n_high,
                                    dJ_drho_hat);
    BackpropagateFilterProjection(dJ_drho_hat, beta, eta, dJ_drho);
  }

  void ComputeProjectedMassGradientFromStateAdjoint(const mfem::Vector &state_vec,
                                                    const mfem::Vector &adjoint_vec,
                                                    double direct_mass_scale,
                                                    double implicit_mass_scale,
                                                    double n_low, double n_high,
                                                    mfem::Vector &dJ_drho_hat)
  {
    mfem::Vector direct_term, implicit_term;
    ComputeProjectedMassSensitivity(state_vec, state_vec, direct_mass_scale, n_low, n_high,
                                    direct_term);
    ComputeProjectedMassSensitivity(state_vec, adjoint_vec, implicit_mass_scale, n_low, n_high,
                                    implicit_term);

    dJ_drho_hat.SetSize(layout.Size());
    dJ_drho_hat = direct_term;
    dJ_drho_hat += implicit_term;
  }

  void ComputeDensityMassGradientFromStateAdjoint(const mfem::Vector &state_vec,
                                                  const mfem::Vector &adjoint_vec,
                                                  double direct_mass_scale,
                                                  double implicit_mass_scale,
                                                  double beta, double eta,
                                                  double n_low, double n_high,
                                                  mfem::Vector &dJ_drho)
  {
    mfem::Vector dJ_drho_hat;
    ComputeProjectedMassGradientFromStateAdjoint(state_vec, adjoint_vec, direct_mass_scale,
                                                 implicit_mass_scale, n_low, n_high,
                                                 dJ_drho_hat);
    BackpropagateFilterProjection(dJ_drho_hat, beta, eta, dJ_drho);
  }

  void EvaluateMassObjectiveAndProjectedGradientFromStates(
      const mfem::Vector &left_state, const mfem::Vector &right_state,
      const mfem::Vector &state_vec, const mfem::Vector &adjoint_vec,
      double objective_mass_scale, double direct_mass_scale,
      double implicit_mass_scale, double n_low, double n_high, double &objective,
      mfem::Vector &dJ_drho_hat)
  {
    objective =
        EvaluateMassObjectiveFromStates(left_state, right_state, objective_mass_scale);
    ComputeProjectedMassGradientFromStateAdjoint(state_vec, adjoint_vec, direct_mass_scale,
                                                 implicit_mass_scale, n_low, n_high,
                                                 dJ_drho_hat);
  }

  void EvaluateMassObjectiveAndDensityGradientFromStates(
      const mfem::Vector &left_state, const mfem::Vector &right_state,
      const mfem::Vector &state_vec, const mfem::Vector &adjoint_vec,
      double objective_mass_scale, double direct_mass_scale,
      double implicit_mass_scale, double beta, double eta, double n_low,
      double n_high, double &objective, mfem::Vector &dJ_drho)
  {
    objective =
        EvaluateMassObjectiveFromStates(left_state, right_state, objective_mass_scale);
    ComputeDensityMassGradientFromStateAdjoint(state_vec, adjoint_vec, direct_mass_scale,
                                               implicit_mass_scale, beta, eta, n_low, n_high,
                                               dJ_drho);
  }

  void ComputeResponseResidualFromState(const mfem::Vector &state_vec, double omega,
                                        mfem::Vector &residual) const
  {
    MFEM_VERIFY(state_vec.Size() == space_op.GetNDSpace().GetTrueVSize(),
                "TopOpt response state size mismatch!");
    ApplyCurrentSystemMatrix(omega, state_vec, residual);
  }

  double ComputeResponseNormAdjointFromResidual(const mfem::Vector &residual,
                                                mfem::Vector &adjoint) const
  {
    const double objective = linalg::Norml2(space_op.GetComm(), residual);
    adjoint.SetSize(residual.Size());
    adjoint = 0.0;
    if (!(std::isfinite(objective) && objective > 1.0e-14))
    {
      return 0.0;
    }

    adjoint = residual;
    adjoint *= 1.0 / objective;
    return objective;
  }

  void EvaluateResponseNormObjectiveAndProjectedGradientFromState(
      const mfem::Vector &state_vec, double omega, double n_low, double n_high,
      double &objective, mfem::Vector &dJ_drho_hat)
  {
    UpdatePermittivityFromProjectedDensity(state.GetProjectedDensity(), n_low, n_high);

    mfem::Vector residual, adjoint;
    ComputeResponseResidualFromState(state_vec, omega, residual);
    objective = ComputeResponseNormAdjointFromResidual(residual, adjoint);

    dJ_drho_hat.SetSize(layout.Size());
    dJ_drho_hat = 0.0;
    if (!(std::isfinite(objective) && objective > 1.0e-14))
    {
      return;
    }

    ComputeProjectedMassGradientFromStateAdjoint(state_vec, adjoint, 0.0, -omega * omega,
                                                 n_low, n_high, dJ_drho_hat);
  }

  void EvaluateResponseNormObjectiveAndDensityGradientFromState(
      const mfem::Vector &state_vec, double omega, double beta, double eta, double n_low,
      double n_high, double &objective, mfem::Vector &dJ_drho)
  {
    mfem::Vector dJ_drho_hat;
    EvaluateResponseNormObjectiveAndProjectedGradientFromState(state_vec, omega, n_low, n_high,
                                                              objective, dJ_drho_hat);
    BackpropagateFilterProjection(dJ_drho_hat, beta, eta, dJ_drho);
  }

  void ComputeResponseNormProjectedGradient(double omega, double n_low, double n_high,
                                            mfem::Vector &dJ_drho_hat)
  {
    double objective = 0.0;
    Vector x(space_op.GetNDSpace().GetTrueVSize());
    x = 1.0;
    EvaluateResponseNormObjectiveAndProjectedGradientFromState(x, omega, n_low, n_high,
                                                              objective, dJ_drho_hat);
  }

  void ComputeLinearSolvedProjectedGradient(double omega, const mfem::Vector &rhs,
                                            const mfem::Vector &weights, double n_low,
                                            double n_high, mfem::Vector &dJ_drho_hat)
  {
    UpdatePermittivityFromProjectedDensity(state.GetProjectedDensity(), n_low, n_high);
    Vector u, lambda;
    SolveProxyState(omega, rhs, u);
    // The current proxy operator is real and symmetric, so the adjoint solve reuses A.
    SolveProxyState(omega, weights, lambda);
    ComputeProjectedMassSensitivity(u, lambda, omega * omega, n_low, n_high,
                                    dJ_drho_hat);
  }

  void ComputeStoredEnergyProjectedGradient(double omega, const mfem::Vector &rhs,
                                            double n_low, double n_high,
                                            mfem::Vector &dJ_drho_hat)
  {
    UpdatePermittivityFromProjectedDensity(state.GetProjectedDensity(), n_low, n_high);

    mfem::Vector u;
    SolveProxyState(omega, rhs, u);
    ComputeStoredEnergyProjectedGradientFromState(u, omega, n_low, n_high, dJ_drho_hat);
  }

  void ComputeLinearSolvedAdjointFromState(const mfem::Vector &weights, double omega,
                                           mfem::Vector &lambda) const
  {
    // The current proxy operator is real and symmetric, so the adjoint solve reuses A.
    SolveProxyState(omega, weights, lambda);
  }

  void EvaluateLinearSolvedObjectiveAndProjectedGradientFromState(
      const mfem::Vector &u, const mfem::Vector &weights, double omega, double n_low,
      double n_high, double &objective, mfem::Vector &dJ_drho_hat)
  {
    mfem::Vector lambda;
    ComputeLinearSolvedAdjointFromState(weights, omega, lambda);
    objective = EvaluateLinearSolvedObjectiveFromState(u, weights);
    ComputeProjectedMassGradientFromStateAdjoint(u, lambda, 0.0, omega * omega, n_low,
                                                 n_high, dJ_drho_hat);
  }

  void EvaluateLinearSolvedObjectiveAndDensityGradientFromState(
      const mfem::Vector &u, const mfem::Vector &weights, double omega, double beta,
      double eta, double n_low, double n_high, double &objective, mfem::Vector &dJ_drho)
  {
    mfem::Vector lambda;
    ComputeLinearSolvedAdjointFromState(weights, omega, lambda);
    objective = EvaluateLinearSolvedObjectiveFromState(u, weights);
    ComputeDensityMassGradientFromStateAdjoint(u, lambda, 0.0, omega * omega, beta, eta,
                                               n_low, n_high, dJ_drho);
  }

  void ComputeStoredEnergyAdjointFromState(const mfem::Vector &u, double omega,
                                           mfem::Vector &lambda) const
  {
    mfem::Vector adjoint_rhs;
    ComputeMassAdjointRHSFromState(u, 1.0, adjoint_rhs);
    // The current proxy operator is real and symmetric, so the adjoint solve reuses A.
    SolveProxyState(omega, adjoint_rhs, lambda);
  }

  void EvaluateStoredEnergyObjectiveAndProjectedGradientFromState(
      const mfem::Vector &u, double omega, double n_low, double n_high,
      double &objective, mfem::Vector &dJ_drho_hat)
  {
    mfem::Vector lambda;
    ComputeStoredEnergyAdjointFromState(u, omega, lambda);
    EvaluateMassObjectiveAndProjectedGradientFromStates(u, u, u, lambda, 0.5, 0.5,
                                                        omega * omega, n_low, n_high,
                                                        objective, dJ_drho_hat);
  }

  void EvaluateStoredEnergyObjectiveAndDensityGradientFromState(
      const mfem::Vector &u, double omega, double beta, double eta, double n_low,
      double n_high, double &objective, mfem::Vector &dJ_drho)
  {
    mfem::Vector lambda;
    ComputeStoredEnergyAdjointFromState(u, omega, lambda);
    EvaluateMassObjectiveAndDensityGradientFromStates(
        u, u, u, lambda, 0.5, 0.5, omega * omega, beta, eta, n_low, n_high,
        objective, dJ_drho);
  }

  void ComputeStoredEnergyProjectedGradientFromState(const mfem::Vector &u, double omega,
                                                     double n_low, double n_high,
                                                     mfem::Vector &dJ_drho_hat)
  {
    UpdatePermittivityFromProjectedDensity(state.GetProjectedDensity(), n_low, n_high);

    MFEM_VERIFY(u.Size() == space_op.GetNDSpace().GetTrueVSize(),
                "TopOpt stored-energy state size mismatch!");

    mfem::Vector Mu, lambda;
    auto M = space_op.GetMassMatrix<Operator>(Operator::DIAG_ZERO);
    Mu.SetSize(u.Size());
    Mu = 0.0;
    M->Mult(u, Mu);

    // The current proxy operator is real and symmetric, so the adjoint solve reuses A.
    SolveProxyState(omega, Mu, lambda);
    ComputeProjectedMassGradientFromStateAdjoint(u, lambda, 0.5, omega * omega, n_low,
                                                 n_high, dJ_drho_hat);
  }

  void BackpropagateFilterProjection(const mfem::Vector &dJ_drho_hat, double beta, double eta,
                                     mfem::Vector &dJ_drho) const
  {
    mfem::Vector dJ_drho_tilde_elem, dJ_drho_tilde_true, dJ_drho_full;
    state.ApplyProjectionAdjoint(dJ_drho_hat, beta, eta, dJ_drho_tilde_elem);
    AssembleElementCenterAdjoint(filter.GetFESpace(), layout, dJ_drho_tilde_elem,
                                 dJ_drho_tilde_true);
    filter.FilterAdjoint(dJ_drho_tilde_true, dJ_drho_full);

    dJ_drho.SetSize(layout.Size());
    for (int i = 0; i < layout.Size(); i++)
    {
      dJ_drho(i) = dJ_drho_full(layout.GetLocalElements()[i]);
    }
  }

  void ComputeResponseNormDensityGradient(double omega, double beta, double eta, double n_low,
                                          double n_high, mfem::Vector &dJ_drho)
  {
    mfem::Vector dJ_drho_hat;
    ComputeResponseNormProjectedGradient(omega, n_low, n_high, dJ_drho_hat);
    BackpropagateFilterProjection(dJ_drho_hat, beta, eta, dJ_drho);
  }

  void ComputeLinearSolvedDensityGradient(double omega, const mfem::Vector &rhs,
                                          const mfem::Vector &weights, double beta,
                                          double eta, double n_low, double n_high,
                                          mfem::Vector &dJ_drho)
  {
    mfem::Vector dJ_drho_hat;
    ComputeLinearSolvedProjectedGradient(omega, rhs, weights, n_low, n_high, dJ_drho_hat);
    BackpropagateFilterProjection(dJ_drho_hat, beta, eta, dJ_drho);
  }

  void ComputeStoredEnergyDensityGradient(double omega, const mfem::Vector &rhs, double beta,
                                          double eta, double n_low, double n_high,
                                          mfem::Vector &dJ_drho)
  {
    mfem::Vector u;
    UpdatePermittivityFromProjectedDensity(state.GetProjectedDensity(), n_low, n_high);
    SolveProxyState(omega, rhs, u);
    ComputeStoredEnergyDensityGradientFromState(u, omega, beta, eta, n_low, n_high, dJ_drho);
  }

  void ComputeStoredEnergyDensityGradientFromState(const mfem::Vector &u, double omega,
                                                   double beta, double eta, double n_low,
                                                   double n_high, mfem::Vector &dJ_drho)
  {
    mfem::Vector dJ_drho_hat;
    ComputeStoredEnergyProjectedGradientFromState(u, omega, n_low, n_high, dJ_drho_hat);
    BackpropagateFilterProjection(dJ_drho_hat, beta, eta, dJ_drho);
  }

  void ComputeProjectedVolumeFractionDensityGradient(const mfem::Vector &element_volume,
                                                     double beta, double eta,
                                                     mfem::Vector &dV_drho) const
  {
    const mfem::Vector dV_drho_hat = VolumeFractionGradient(element_volume);
    BackpropagateFilterProjection(dV_drho_hat, beta, eta, dV_drho);
  }

  void EvaluateResponseNormWithVolumeConstraint(const mfem::Vector &rho, double omega,
                                                double beta, double eta, double n_low,
                                                double n_high,
                                                const mfem::Vector &element_volume,
                                                double volume_fraction_target,
                                                TopOptEvaluation &eval)
  {
    UpdatePermittivityFromDensity(rho, beta, eta, n_low, n_high);

    eval.objective = EvaluateResponseNorm(omega);
    eval.constraint =
        EvaluateProjectedVolumeFraction(element_volume) - volume_fraction_target;

    ComputeResponseNormDensityGradient(omega, beta, eta, n_low, n_high,
                                       eval.objective_gradient);
    ComputeProjectedVolumeFractionDensityGradient(element_volume, beta, eta,
                                                 eval.constraint_gradient);
  }
};

class TopOptProblem
{
private:
  TopOptContext &ctx;
  double omega;
  double beta;
  double eta;
  double n_low;
  double n_high;
  mfem::Vector element_volume;
  double volume_fraction_target;

  static void ApplyClampedObjectiveStep(mfem::Vector &rho, const mfem::Vector &gradient,
                                        double step_size, double lower_bound,
                                        double upper_bound)
  {
    MFEM_VERIFY(rho.Size() == gradient.Size(),
                "TopOpt design vector and objective gradient size mismatch!");
    for (int i = 0; i < rho.Size(); i++)
    {
      const double trial = rho(i) - step_size * gradient(i);
      if (trial < lower_bound)
      {
        rho(i) = lower_bound;
      }
      else if (trial > upper_bound)
      {
        rho(i) = upper_bound;
      }
      else
      {
        rho(i) = trial;
      }
    }
  }

public:
  TopOptProblem(TopOptContext &ctx, double omega, double beta, double eta, double n_low,
                double n_high, const mfem::Vector &element_volume,
                double volume_fraction_target)
    : ctx(ctx), omega(omega), beta(beta), eta(eta), n_low(n_low), n_high(n_high),
      element_volume(element_volume), volume_fraction_target(volume_fraction_target)
  {
  }

  static double ComputeQuadraticPenaltyMerit(const TopOptEvaluation &eval, double penalty)
  {
    MFEM_VERIFY(std::isfinite(penalty) && penalty >= 0.0,
                "TopOpt penalty must be finite and nonnegative!");
    const double violation = std::max(0.0, eval.constraint);
    return eval.objective + penalty * violation * violation;
  }

  static void ComputeQuadraticPenaltyGradient(const TopOptEvaluation &eval, double penalty,
                                              mfem::Vector &gradient)
  {
    MFEM_VERIFY(std::isfinite(penalty) && penalty >= 0.0,
                "TopOpt penalty must be finite and nonnegative!");

    gradient.SetSize(eval.objective_gradient.Size());
    gradient = eval.objective_gradient;
    if (eval.constraint > 0.0)
    {
      gradient.Add(2.0 * penalty * eval.constraint, eval.constraint_gradient);
    }
  }

  int GetNumDesignVariables() const { return ctx.GetLayout().Size(); }

  void Evaluate(const mfem::Vector &rho, TopOptEvaluation &eval)
  {
    ctx.EvaluateResponseNormWithVolumeConstraint(rho, omega, beta, eta, n_low, n_high,
                                                element_volume, volume_fraction_target, eval);
  }

  void TakeObjectiveStep(mfem::Vector &rho, double step_size, double lower_bound = 0.0,
                         double upper_bound = 1.0, TopOptEvaluation *eval_out = nullptr)
  {
    MFEM_VERIFY(std::isfinite(step_size) && step_size >= 0.0,
                "TopOpt step size must be finite and nonnegative!");
    MFEM_VERIFY(std::isfinite(lower_bound) && std::isfinite(upper_bound) &&
                    lower_bound <= upper_bound,
                "TopOpt step bounds are invalid!");

    TopOptEvaluation eval;
    Evaluate(rho, eval);
    if (eval_out)
    {
      *eval_out = eval;
    }

    ApplyClampedObjectiveStep(rho, eval.objective_gradient, step_size, lower_bound,
                              upper_bound);
  }

  bool TakePenaltyStep(mfem::Vector &rho, double initial_step, double penalty,
                       int max_backtracks = 8, double backtrack_factor = 0.5,
                       double lower_bound = 0.0, double upper_bound = 1.0,
                       TopOptEvaluation *eval_out = nullptr,
                       double *accepted_step_out = nullptr)
  {
    MFEM_VERIFY(std::isfinite(initial_step) && initial_step > 0.0,
                "TopOpt initial step must be finite and positive!");
    MFEM_VERIFY(std::isfinite(backtrack_factor) && backtrack_factor > 0.0 &&
                    backtrack_factor < 1.0,
                "TopOpt backtrack factor must be in (0, 1)!");
    MFEM_VERIFY(max_backtracks >= 0, "TopOpt max backtracks must be nonnegative!");
    MFEM_VERIFY(std::isfinite(lower_bound) && std::isfinite(upper_bound) &&
                    lower_bound <= upper_bound,
                "TopOpt step bounds are invalid!");

    TopOptEvaluation current_eval;
    Evaluate(rho, current_eval);
    const double current_merit = ComputeQuadraticPenaltyMerit(current_eval, penalty);

    mfem::Vector merit_gradient;
    ComputeQuadraticPenaltyGradient(current_eval, penalty, merit_gradient);

    double step = initial_step;
    mfem::Vector rho_trial(rho);
    for (int k = 0; k <= max_backtracks; k++)
    {
      rho_trial = rho;
      ApplyClampedObjectiveStep(rho_trial, merit_gradient, step, lower_bound, upper_bound);

      bool changed = false;
      for (int i = 0; i < rho.Size(); i++)
      {
        if (rho_trial(i) != rho(i))
        {
          changed = true;
          break;
        }
      }
      if (!changed)
      {
        break;
      }

      TopOptEvaluation trial_eval;
      Evaluate(rho_trial, trial_eval);
      const double trial_merit = ComputeQuadraticPenaltyMerit(trial_eval, penalty);
      const double tol = 1.0e-12 * std::max(1.0, std::abs(current_merit));
      if (std::isfinite(trial_merit) && trial_merit <= current_merit + tol)
      {
        rho = rho_trial;
        if (eval_out)
        {
          *eval_out = trial_eval;
        }
        if (accepted_step_out)
        {
          *accepted_step_out = step;
        }
        return true;
      }

      step *= backtrack_factor;
    }

    if (eval_out)
    {
      *eval_out = current_eval;
    }
    if (accepted_step_out)
    {
      *accepted_step_out = 0.0;
    }
    return false;
  }

  void RunFixedObjectiveSteps(mfem::Vector &rho, int num_steps, double step_size,
                              double lower_bound = 0.0, double upper_bound = 1.0,
                              TopOptIterationHistory *history = nullptr)
  {
    MFEM_VERIFY(num_steps >= 0, "TopOpt iteration count must be nonnegative!");

    if (history)
    {
      history->objective.SetSize(num_steps);
      history->constraint.SetSize(num_steps);
    }

    for (int k = 0; k < num_steps; k++)
    {
      TopOptEvaluation eval;
      TakeObjectiveStep(rho, step_size, lower_bound, upper_bound, &eval);

      if (history)
      {
        history->objective(k) = eval.objective;
        history->constraint(k) = eval.constraint;
      }
    }
  }

  void RunObjectiveSteps(mfem::Vector &rho, const TopOptRunOptions &options,
                         TopOptRunSummary *summary = nullptr)
  {
    MFEM_VERIFY(options.max_steps >= 0, "TopOpt iteration count must be nonnegative!");
    MFEM_VERIFY(std::isfinite(options.step_size) && options.step_size >= 0.0,
                "TopOpt step size must be finite and nonnegative!");
    MFEM_VERIFY(std::isfinite(options.gradient_tolerance) &&
                    options.gradient_tolerance >= 0.0,
                "TopOpt gradient tolerance must be finite and nonnegative!");
    MFEM_VERIFY(std::isfinite(options.lower_bound) && std::isfinite(options.upper_bound) &&
                    options.lower_bound <= options.upper_bound,
                "TopOpt step bounds are invalid!");

    mfem::Vector objective_hist(options.max_steps), constraint_hist(options.max_steps),
        gradient_hist(options.max_steps);
    int num_iterations = 0;
    bool converged = false;
    for (int k = 0; k < options.max_steps; k++)
    {
      TopOptEvaluation eval;
      Evaluate(rho, eval);

      const double grad_norm = linalg::Norml2(ctx.GetComm(), eval.objective_gradient);
      objective_hist(k) = eval.objective;
      constraint_hist(k) = eval.constraint;
      gradient_hist(k) = grad_norm;
      num_iterations++;

      if (options.gradient_tolerance > 0.0 && grad_norm <= options.gradient_tolerance)
      {
        converged = true;
        break;
      }

      ApplyClampedObjectiveStep(rho, eval.objective_gradient, options.step_size,
                                options.lower_bound, options.upper_bound);
    }

    if (summary)
    {
      summary->history.objective.SetSize(num_iterations);
      summary->history.constraint.SetSize(num_iterations);
      summary->gradient_norm.SetSize(num_iterations);
      for (int i = 0; i < num_iterations; i++)
      {
        summary->history.objective(i) = objective_hist(i);
        summary->history.constraint(i) = constraint_hist(i);
        summary->gradient_norm(i) = gradient_hist(i);
      }
      summary->num_iterations = num_iterations;
      summary->converged = converged;
    }
  }

  void RunPenaltySteps(mfem::Vector &rho, const TopOptPenaltyRunOptions &options,
                       TopOptPenaltyRunSummary *summary = nullptr)
  {
    MFEM_VERIFY(options.max_steps >= 0, "TopOpt iteration count must be nonnegative!");
    MFEM_VERIFY(std::isfinite(options.initial_step) && options.initial_step > 0.0,
                "TopOpt initial step must be finite and positive!");
    MFEM_VERIFY(std::isfinite(options.penalty) && options.penalty >= 0.0,
                "TopOpt penalty must be finite and nonnegative!");
    MFEM_VERIFY(std::isfinite(options.backtrack_factor) && options.backtrack_factor > 0.0 &&
                    options.backtrack_factor < 1.0,
                "TopOpt backtrack factor must be in (0, 1)!");
    MFEM_VERIFY(options.max_backtracks >= 0, "TopOpt max backtracks must be nonnegative!");
    MFEM_VERIFY(std::isfinite(options.lower_bound) && std::isfinite(options.upper_bound) &&
                    options.lower_bound <= options.upper_bound,
                "TopOpt step bounds are invalid!");
    MFEM_VERIFY(std::isfinite(options.merit_tolerance) && options.merit_tolerance >= 0.0,
                "TopOpt merit tolerance must be finite and nonnegative!");
    MFEM_VERIFY(options.max_consecutive_rejects >= 0,
                "TopOpt reject limit must be nonnegative!");

    if (summary)
    {
      summary->history.objective.SetSize(options.max_steps);
      summary->history.constraint.SetSize(options.max_steps);
      summary->merit.SetSize(options.max_steps);
      summary->accepted_step.SetSize(options.max_steps);
      summary->accepted.SetSize(options.max_steps);
      summary->num_iterations = options.max_steps;
      summary->num_accepted = 0;
      summary->num_rejected = 0;
      summary->converged = false;
      summary->stopped_on_reject_limit = false;
    }

    mfem::Vector objective_hist(options.max_steps), constraint_hist(options.max_steps),
        merit_hist(options.max_steps), accepted_step_hist(options.max_steps),
        accepted_hist(options.max_steps);
    int num_iterations = 0;
    int num_accepted = 0;
    int num_rejected = 0;
    int consecutive_rejects = 0;
    bool converged = false;
    bool stopped_on_reject_limit = false;
    for (int k = 0; k < options.max_steps; k++)
    {
      TopOptEvaluation eval;
      double accepted_step = 0.0;
      const bool accepted =
          TakePenaltyStep(rho, options.initial_step, options.penalty, options.max_backtracks,
                          options.backtrack_factor, options.lower_bound, options.upper_bound,
                          &eval, &accepted_step);
      if (accepted)
      {
        num_accepted++;
        consecutive_rejects = 0;
      }
      else
      {
        num_rejected++;
        consecutive_rejects++;
      }

      objective_hist(k) = eval.objective;
      constraint_hist(k) = eval.constraint;
      merit_hist(k) = ComputeQuadraticPenaltyMerit(eval, options.penalty);
      accepted_step_hist(k) = accepted_step;
      accepted_hist(k) = accepted ? 1.0 : 0.0;
      num_iterations++;

      if (k > 0 && options.merit_tolerance > 0.0)
      {
        const double merit_diff = std::abs(merit_hist(k) - merit_hist(k - 1));
        const double merit_tol =
            options.merit_tolerance * std::max(1.0, std::abs(merit_hist(k - 1)));
        if (merit_diff <= merit_tol)
        {
          converged = true;
          break;
        }
      }
      if (options.max_consecutive_rejects > 0 &&
          consecutive_rejects >= options.max_consecutive_rejects)
      {
        stopped_on_reject_limit = true;
        break;
      }
    }

    if (summary)
    {
      summary->history.objective.SetSize(num_iterations);
      summary->history.constraint.SetSize(num_iterations);
      summary->merit.SetSize(num_iterations);
      summary->accepted_step.SetSize(num_iterations);
      summary->accepted.SetSize(num_iterations);
      for (int i = 0; i < num_iterations; i++)
      {
        summary->history.objective(i) = objective_hist(i);
        summary->history.constraint(i) = constraint_hist(i);
        summary->merit(i) = merit_hist(i);
        summary->accepted_step(i) = accepted_step_hist(i);
        summary->accepted(i) = accepted_hist(i);
      }
      summary->num_iterations = num_iterations;
      summary->num_accepted = num_accepted;
      summary->num_rejected = num_rejected;
      summary->converged = converged;
      summary->stopped_on_reject_limit = stopped_on_reject_limit;
    }
  }

  void RunPenaltySteps(mfem::Vector &rho, int num_steps, double initial_step, double penalty,
                       int max_backtracks = 8, double backtrack_factor = 0.5,
                       double lower_bound = 0.0, double upper_bound = 1.0,
                       TopOptPenaltyRunSummary *summary = nullptr)
  {
    TopOptPenaltyRunOptions options;
    options.max_steps = num_steps;
    options.initial_step = initial_step;
    options.penalty = penalty;
    options.max_backtracks = max_backtracks;
    options.backtrack_factor = backtrack_factor;
    options.lower_bound = lower_bound;
    options.upper_bound = upper_bound;
    RunPenaltySteps(rho, options, summary);
  }

  void RunSession(mfem::Vector &rho, const TopOptSessionOptions &options,
                  TopOptSessionSummary *summary = nullptr)
  {
    if (options.mode == TopOptSessionMode::OBJECTIVE)
    {
      TopOptRunSummary objective_summary;
      RunObjectiveSteps(rho, options.objective, summary ? &objective_summary : nullptr);

      if (summary)
      {
        summary->mode = TopOptSessionMode::OBJECTIVE;
        summary->history = objective_summary.history;
        summary->trace_value = objective_summary.gradient_norm;
        summary->accepted_step.SetSize(objective_summary.num_iterations);
        summary->accepted.SetSize(objective_summary.num_iterations);
        for (int i = 0; i < objective_summary.num_iterations; i++)
        {
          summary->accepted_step(i) = options.objective.step_size;
          summary->accepted(i) = 1.0;
        }
        summary->num_iterations = objective_summary.num_iterations;
        summary->num_accepted = objective_summary.num_iterations;
        summary->num_rejected = 0;
        summary->converged = objective_summary.converged;
        summary->stopped_on_reject_limit = false;
      }
      return;
    }

    TopOptPenaltyRunSummary penalty_summary;
    RunPenaltySteps(rho, options.penalty, summary ? &penalty_summary : nullptr);

    if (summary)
    {
      summary->mode = TopOptSessionMode::PENALTY;
      summary->history = penalty_summary.history;
      summary->trace_value = penalty_summary.merit;
      summary->accepted_step = penalty_summary.accepted_step;
      summary->accepted = penalty_summary.accepted;
      summary->num_iterations = penalty_summary.num_iterations;
      summary->num_accepted = penalty_summary.num_accepted;
      summary->num_rejected = penalty_summary.num_rejected;
      summary->converged = penalty_summary.converged;
      summary->stopped_on_reject_limit = penalty_summary.stopped_on_reject_limit;
    }
  }
};

class TopOptDriver
{
private:
  TopOptProblem &problem;
  TopOptSessionOptions options;
  mfem::Vector design;
  TopOptSessionSummary summary;
  std::vector<TopOptSessionSummary> run_history;

  static TopOptDriverReport MakeReport(const TopOptSessionSummary &summary,
                                       int num_recorded_runs)
  {
    MFEM_VERIFY(summary.num_iterations > 0, "TopOptDriver has no completed run!");

    TopOptDriverReport report;
    report.mode = summary.mode;
    report.objective = summary.history.objective(summary.num_iterations - 1);
    report.constraint = summary.history.constraint(summary.num_iterations - 1);
    report.trace_value = summary.trace_value(summary.num_iterations - 1);
    report.num_iterations = summary.num_iterations;
    report.num_accepted = summary.num_accepted;
    report.num_rejected = summary.num_rejected;
    report.converged = summary.converged;
    report.stopped_on_reject_limit = summary.stopped_on_reject_limit;
    report.num_recorded_runs = num_recorded_runs;
    return report;
  }

public:
  TopOptDriver(TopOptProblem &problem, const TopOptSessionOptions &options,
               const mfem::Vector &initial_design)
    : problem(problem), options(options), design(initial_design)
  {
    MFEM_VERIFY(design.Size() == problem.GetNumDesignVariables(),
                "TopOpt initial design size mismatch!");
  }

  int GetNumDesignVariables() const { return design.Size(); }

  const auto &GetOptions() const { return options; }
  void SetOptions(const TopOptSessionOptions &new_options) { options = new_options; }

  const auto &GetDesign() const { return design; }
  auto &GetDesign() { return design; }

  void SetDesign(const mfem::Vector &rho)
  {
    MFEM_VERIFY(rho.Size() == problem.GetNumDesignVariables(),
                "TopOpt design size mismatch!");
    design = rho;
  }

  void SetUniformDesign(double rho) { design = rho; }

  const auto &GetSummary() const { return summary; }
  const auto &GetRunHistory() const { return run_history; }
  std::vector<TopOptDriverReport> GetRunReports() const
  {
    std::vector<TopOptDriverReport> reports;
    reports.reserve(run_history.size());
    for (std::size_t i = 0; i < run_history.size(); i++)
    {
      reports.push_back(MakeReport(run_history[i], static_cast<int>(i + 1)));
    }
    return reports;
  }
  bool HasRun() const { return summary.num_iterations > 0; }

  void ClearRunHistory() { run_history.clear(); }

  double GetLatestObjective() const
  {
    MFEM_VERIFY(summary.num_iterations > 0, "TopOptDriver has no completed run!");
    return summary.history.objective(summary.num_iterations - 1);
  }

  double GetLatestConstraint() const
  {
    MFEM_VERIFY(summary.num_iterations > 0, "TopOptDriver has no completed run!");
    return summary.history.constraint(summary.num_iterations - 1);
  }

  double GetLatestTraceValue() const
  {
    MFEM_VERIFY(summary.num_iterations > 0, "TopOptDriver has no completed run!");
    return summary.trace_value(summary.num_iterations - 1);
  }

  TopOptDriverReport GetLatestReport() const
  {
    return MakeReport(summary, static_cast<int>(run_history.size()));
  }

  const TopOptSessionSummary &Run(bool record_history = true)
  {
    problem.RunSession(design, options, &summary);
    if (record_history)
    {
      run_history.push_back(summary);
    }
    return summary;
  }
};

}  // namespace palace::topopt

#endif  // PALACE_UTILS_TOPOPTCONTEXT_HPP
