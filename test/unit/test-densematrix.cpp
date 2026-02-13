// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <mfem.hpp>

#include "linalg/densematrix.hpp"

namespace palace
{
namespace
{

mfem::DenseMatrix MakeSymmetric3x3(const std::array<double, 9> &vals)
{
  mfem::DenseMatrix M(3);
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      M(i, j) = vals[3 * i + j];
    }
  }
  return M;
}

double MaxAbsEntry(const mfem::DenseMatrix &M)
{
  double m = 0.0;
  for (int i = 0; i < M.Height(); i++)
  {
    for (int j = 0; j < M.Width(); j++)
    {
      m = std::max(m, std::abs(M(i, j)));
    }
  }
  return m;
}

double MaxAbsDiff(const mfem::DenseMatrix &A, const mfem::DenseMatrix &B)
{
  double m = 0.0;
  for (int i = 0; i < A.Height(); i++)
  {
    for (int j = 0; j < A.Width(); j++)
    {
      m = std::max(m, std::abs(A(i, j) - B(i, j)));
    }
  }
  return m;
}

void CheckMatrixSqrtConsistency(const mfem::DenseMatrix &M)
{
  auto S = linalg::MatrixSqrt(M);
  auto P = linalg::MatrixPow(M, 0.5);

  mfem::DenseMatrix SS(3);
  mfem::Mult(S, S, SS);

  const double scale = std::max(MaxAbsEntry(M), 1.0);
  const double tol = 5.0e-12 * scale;

  INFO("max |S*S - M| = " << MaxAbsDiff(SS, M) << ", tol = " << tol);
  CHECK(MaxAbsDiff(SS, M) < tol);

  INFO("max |MatrixPow(M,0.5) - MatrixSqrt(M)| = " << MaxAbsDiff(P, S)
                                                   << ", tol = " << tol);
  CHECK(MaxAbsDiff(P, S) < tol);
}

}  // namespace

TEST_CASE("DenseMatrix 3x3 special-case matrix functions are normalized", "[densematrix][Serial]")
{
  SECTION("Only (0,1)/(1,0) off-diagonal entries are nonzero")
  {
    const auto M = MakeSymmetric3x3({
        4.0, 1.0, 0.0,  //
        1.0, 9.0, 0.0,  //
        0.0, 0.0, 16.0  //
    });
    CheckMatrixSqrtConsistency(M);
  }

  SECTION("Only (1,2)/(2,1) off-diagonal entries are nonzero")
  {
    const auto M = MakeSymmetric3x3({
        25.0, 0.0, 0.0,  //
        0.0, 9.0, 2.0,   //
        0.0, 2.0, 4.0    //
    });
    CheckMatrixSqrtConsistency(M);
  }

  SECTION("Only (0,2)/(2,0) off-diagonal entries are nonzero")
  {
    const auto M = MakeSymmetric3x3({
        9.0, 0.0, 3.0,   //
        0.0, 16.0, 0.0,  //
        3.0, 0.0, 4.0    //
    });
    CheckMatrixSqrtConsistency(M);
  }
}

}  // namespace palace
