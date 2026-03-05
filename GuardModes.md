# Silent Data Corruption in Wave Port Mode Normalization

## Summary

The `Normalize()` function in `palace/models/waveportoperator.cpp` can silently produce corrupted mode fields and S-parameters when the wave port eigensolver converges to a spurious mode. The corruption is caused by dividing by a near-zero power integral without any guard. The bug is triggered by specific MPI mesh partitioning configurations and is not caught or reported — the simulation completes with wrong results.

## How Wave Port Mode Solving Works

When Palace encounters a wave port boundary, it needs to know what the electromagnetic mode looks like on that port face. This is how it finds it:

### Step 1: Solve a 2D eigenvalue problem on the port face

The port boundary is a 2D cross-section (e.g., the face of a waveguide). Palace extracts this 2D surface, builds a finite element mesh on it, and solves a generalized eigenvalue problem to find the propagating mode shape. This is done in `WavePortData::Initialize(omega)` (waveportoperator.cpp, line ~849).

The eigenvalue `lambda` gives the propagation constant:

```
kn0 = sqrt(-sigma - 1/lambda)
```

The eigenvector gives the transverse electric field `E0t` and, after dividing by `i*kn0`, the longitudinal component `E0n`.

### Step 2: Compute the magnetic field and Poynting vector forms

From the electric mode field (`E0t`, `E0n`) and the propagation constant (`kn0`), Palace computes the corresponding magnetic field `n x H` on the port surface. This is done via `BdrSubmeshHVectorCoefficient`, which uses the relationship between E and H for a waveguide mode.

Two linear forms are assembled:
- `port_sr` — the real part of `n x H` projected onto the finite element space
- `port_si` — the imaginary part

These are used later to extract S-parameters by computing inner products with the solved 3D electric field.

### Step 3: Normalize to unit power

The mode fields from the eigensolver have arbitrary magnitude. Palace normalizes them so that the power flowing through the port is exactly 1. This is done in the `Normalize()` function (waveportoperator.cpp, line ~287).

The normalization computes two quantities:

```
dot[0] = integral of (E . S0t) over the port     — the "phase" dot product
dot[1] = integral of |E x H*| . n over the port  — the mode power
```

where `S0t` is a reference polarization field used to fix the phase convention.

The mode fields are then scaled by:

```
scale = |dot[0]| / (dot[0] * sqrt(|dot[1]|))
```

This simultaneously:
- Divides by `sqrt(|dot[1]|)` to normalize to unit power
- Multiplies by `|dot[0]| / dot[0]` to fix the phase

After scaling, `E0t`, `E0n`, `sr`, and `si` all carry the correctly normalized mode. S-parameters extracted from the 3D solve will be correct if and only if this normalization is correct.

## The Bug

The upstream code (Palace v0.15.0) computes the scale factor in a single line:

```cpp
auto scale = std::abs(dot[0]) / (dot[0] * std::sqrt(std::abs(dot[1])));
```

There is no check on the magnitude of `dot[0]` or `dot[1]`. Two things go wrong when the eigensolver produces a spurious mode:

### Problem 1: `dot[1]` near zero — division by near-zero power

When the eigensolver converges to a spurious (non-propagating) mode, the power integral `|E x H*| . n` is essentially zero. The `sqrt(|dot[1]|)` in the denominator then produces a very large number, and the scale factor blows up to ~10⁹ or more. The mode fields `E0t`, `E0n` and the S-parameter forms `sr`, `si` are all multiplied by this enormous factor, becoming nonsense.

### Problem 2: `dot[0]` near zero — phase from noise

The phase correction `|dot[0]| / dot[0]` extracts the direction of the complex number `dot[0]`. When `dot[0]` is near zero (which happens when the mode has no meaningful projection onto the reference polarization), this computes the angle of a number dominated by floating-point noise. The resulting "phase" is random.

### Combined effect

The normalized mode fields are scaled by a near-infinite factor and rotated by a random phase. Every downstream quantity that depends on these fields — S-parameters, port excitation vectors, impedance matrices — is corrupted. The simulation does not crash or warn. It simply produces wrong answers.

## Numerical Evidence

We discovered this bug by running the `adapter/hybrid` regression test (2 wave ports, eigenmode problem) with different numbers of MPI processes.

### At 4 MPI processes — all modes are healthy

Every call to `Normalize()` shows `dot1_abs` (the mode power) in the range **0.028 to 0.157**:

```
dot1_abs = 2.846745e-02, dot0_abs = 1.821855e-03
dot1_abs = 6.958186e-02, dot0_abs = 6.345721e-03
dot1_abs = 1.568079e-01, dot0_abs = 1.003536e-02
...  (all similar, hundreds of calls, all O(10^-2) to O(10^-1))
```

### At 6 MPI processes — one spurious mode

The first three normalizations are healthy. The fourth:

```
dot1_abs = 4.342828e-18, dot0_abs = 2.319290e-12
```

`dot1_abs = 4.34e-18` is **16 orders of magnitude** smaller than any legitimate mode power value. This is machine zero — the eigensolver converged to a mode with no power flow through the port. The subsequent `MPI_ABORT` in our guarded version shows this triggers 6 additional failures across all ranks.

### What upstream does with this value

Upstream computes:

```
scale = |2.32e-12| / (2.32e-12 * sqrt(4.34e-18))
      = 2.32e-12 / (2.32e-12 * 6.59e-10)
      ≈ 1 / 6.59e-10
      ≈ 1.52e+9
```

The mode fields are multiplied by ~1.5 billion. Since `dot[0]` is also near-zero, the phase `|dot[0]|/dot[0]` is computed from noise. The result: mode fields are scaled to ~10⁹ in a random direction. These corrupted fields propagate into S-parameter extraction for every subsequent frequency evaluation at this port.

## Root Cause

The 2D eigenvalue problem on the port face is solved using SLEPc's shift-and-invert eigensolver. When the 3D mesh is partitioned across MPI ranks, the 2D port submesh is also partitioned. At 6 ranks, the port face partition is different from 4 ranks.

For certain partitions, the eigensolver converges to a spurious mode — one that satisfies the eigenvalue equation numerically but does not correspond to a physical propagating mode. This can happen because:

1. The port submesh partition creates small disconnected or poorly-conditioned pieces
2. The shift-and-invert preconditioner becomes ill-conditioned for that partition
3. The eigensolver converges to a local minimum (non-physical solution) instead of the desired propagating mode

The key point: **the eigensolver reports convergence** (`num_conv >= mode_idx` passes), and the eigenvalue itself may look reasonable. The spurious nature only becomes apparent in the power integral, which the current code never checks.

## How to Reproduce on Upstream Palace

To demonstrate this to the Palace developers using unmodified upstream code, follow these steps:

### Step 1: Add diagnostic printing to upstream Palace

Apply this minimal patch to `palace/models/waveportoperator.cpp` in the `Normalize()` function. Change the single line:

```cpp
// BEFORE (upstream, line ~300):
auto scale = std::abs(dot[0]) / (dot[0] * std::sqrt(std::abs(dot[1])));
```

to:

```cpp
// AFTER (with diagnostics):
const double dot0_abs = std::abs(dot[0]);
const double dot1_abs = std::abs(dot[1]);
Mpi::Print("Wave port normalization: dot1_abs = {:.6e}, dot0_abs = {:.6e}, "
           "scale_magnitude = {:.6e}\n",
           dot1_abs, dot0_abs,
           dot1_abs > 0 ? dot0_abs / (dot0_abs * std::sqrt(dot1_abs)) : -1.0);
auto scale = dot0_abs / (dot[0] * std::sqrt(dot1_abs));
```

### Step 2: Rebuild and run `adapter/hybrid` at 4 and 6 MPI processes

```bash
# Build
cd build && make -j$(nproc)

# Run at 4 processes (expect all healthy)
mpirun -np 4 ./bin/palace-x86_64.bin ../examples/adapter/hybrid.json 2>&1 | grep "dot1_abs"

# Run at 6 processes (expect one spurious mode)
mpirun -np 6 ./bin/palace-x86_64.bin ../examples/adapter/hybrid.json 2>&1 | grep "dot1_abs"
```

### Step 3: Look for the signature

In the 6-process output, look for any line where `dot1_abs` drops to O(10^-15) or smaller while the other values are O(10^-2). That is the spurious mode. The `scale_magnitude` for that line will be O(10^8) or larger — confirming the mode fields are being blown up.

### Step 4: Compare S-parameters

Run the full regression test at both 4 and 6 processes and compare the S-parameter CSV output. The 6-process results will show corrupted values for the frequencies where the spurious mode was used.

## Suggested Fix

The `Normalize()` function should guard against near-zero `dot[0]` and `dot[1]`:

```cpp
void Normalize(const GridFunction &S0t, GridFunction &E0t, GridFunction &E0n,
               mfem::LinearForm &sr, mfem::LinearForm &si)
{
  std::complex<double> dot[2] = {
      {sr * S0t.Real(), si * S0t.Real()},
      {-(sr * E0t.Real()) - (si * E0t.Imag()), -(sr * E0t.Imag()) + (si * E0t.Real())}};
  Mpi::GlobalSum(2, dot, S0t.ParFESpace()->GetComm());

  constexpr double tol = 10.0 * std::numeric_limits<double>::epsilon();
  const double dot0_abs = std::abs(dot[0]);
  const double dot1_abs = std::abs(dot[1]);

  // Guard: dot[1] is the mode power integral. A near-zero value means the
  // eigensolver converged to a spurious (non-propagating) mode. Dividing by
  // sqrt(dot1_abs) would blow up the mode fields to nonsense values.
  MFEM_VERIFY(std::isfinite(dot1_abs) && dot1_abs > tol,
              "Invalid wave port normalization: near-zero mode power! "
              "(dot1_abs = " << dot1_abs << "). "
              "The wave port eigensolver may have converged to a spurious mode. "
              "Try a different number of MPI processes.");

  // Guard: dot[0] is the phase reference. When near-zero, the phase direction
  // is dominated by floating-point noise. Default to phase = 1.
  const std::complex<double> phase =
      (dot0_abs > tol) ? (dot[0] / dot0_abs) : std::complex<double>(1.0, 0.0);

  auto scale = 1.0 / (phase * std::sqrt(dot1_abs));
  ComplexVector::AXPBY(scale, E0t.Real(), E0t.Imag(), 0.0, E0t.Real(), E0t.Imag());
  ComplexVector::AXPBY(scale, E0n.Real(), E0n.Imag(), 0.0, E0n.Real(), E0n.Imag());
  ComplexVector::AXPBY(scale, sr, si, 0.0, sr, si);
}
```

This requires adding `#include <cmath>` and `#include <limits>` at the top of the file.

### Additionally: guard the eigenvalue and kn0 division in `Initialize()`

In `WavePortData::Initialize()`, the eigenvalue extraction and the `1/(i*kn0)` division should also be guarded:

```cpp
// After extracting lambda:
MFEM_VERIFY(std::abs(lambda) > tol,
            "Wave port eigensolver produced near-zero eigenvalue!");
kn0 = std::sqrt(-sigma - 1.0 / lambda);

// Before dividing by kn0 to get longitudinal field:
if (std::abs(kn0) > tol)
{
  ComplexVector::AXPBY(1.0 / (1i * kn0), e0nr, e0ni, 0.0, e0nr, e0ni);
}
else
{
  // At cutoff, the longitudinal component is zero
  e0nr = 0.0;
  e0ni = 0.0;
}
```

## Summary of Changes in This Document

| Location | Upstream behavior | Problem | Fix |
|----------|------------------|---------|-----|
| `Normalize()`: `dot[1]` near zero | Divides by `sqrt(~0)` → scale ~10⁹ | Mode fields blown up, S-params corrupted | `MFEM_VERIFY` that `dot1_abs > tol` |
| `Normalize()`: `dot[0]` near zero | Computes `\|~0\| / ~0` → random phase | Phase from noise | Default to phase = 1.0 when below tolerance |
| `Initialize()`: `lambda` near zero | Computes `1/~0` → overflow in `kn0` | Propagation constant is garbage | `MFEM_VERIFY` that `\|lambda\| > tol` |
| `Initialize()`: `kn0` near zero | Divides `e0n` by `i*~0` → overflow | Longitudinal field blown up | Set `e0n = 0` (correct at cutoff) |
