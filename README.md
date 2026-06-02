# Master equation in discrete time — Huggett example

Discrete-time companion to the continuous-time master-equation framework of
Bilal, *Solving Heterogeneous Agent Models with the Master Equation*. The code
solves the steady state of a discrete-time Huggett economy (idiosyncratic income,
a single implicitly priced bond with `1/q = 1 + r`) and computes the interest-rate
impulse response to a wealth-redistribution shock **three independent ways**:

- **FAME** — First-order Approximation to the Master Equation: solves for the
  Impulse Value `v(x,ξ) = ∂V(x)/∂g(ξ)` and the linearized law of motion `T' + G`.
- **SSJ** — Sequence-Space Jacobian / "fake-news" algorithm of
  Auclert, Bardóczy, Rognlie & Straub (2021).
- **MIT** — fully nonlinear perfect-foresight transition (ground-truth validation).

The three agree up to truncation and numerical-differentiation error, which
cross-validates the master-equation solution.

## Repository structure

| path | description |
|---|---|
| `huggett_fame.jl` | Model library: parameters, Rouwenhorst income (via QuantEcon), EGM household solver, steady state, the FAME solver `solve_fame`, the `redistribution_shock`, and the `fame_irf` helper. Functions only. |
| `huggett_ssj.jl` | Sequence-space Jacobian: fake-news primitives, expectation vectors, Jacobian assembly, and `ssj_irf`. Functions only. |
| `huggett_mit.jl` | Nonlinear perfect-foresight (MIT-shock) transition `mit_transition`. Functions only. |
| `toplevel_huggett.jl` | **Toplevel driver.** Loads the three libraries, solves the model, computes the FAME/SSJ/MIT interest-rate IRFs for one shock, prints a comparison table, and saves the combined figure. |
| `bench_ssj_fame.jl` | Performance benchmark producing the summary table below. |
| `figure/` | Generated figures (`huggett_r_response_kappa*.pdf`). |

The `document/` folder (LaTeX derivation `KrusellSmith_DiscreteTime.tex` + compiled
PDF) and `literature/` (reference papers) are kept locally but are **git-ignored**.

## Running

```bash
julia toplevel_huggett.jl     # solve + compare + save figure/huggett_r_response_*.pdf
julia bench_ssj_fame.jl       # performance table
```

Dependencies: `LinearAlgebra`, `Printf`, `Random`, `Plots`, `QuantEcon`
(and `BenchmarkTools` for the benchmark). Tested on Julia 1.12.

## Performance

Time to compute one interest-rate impulse response on the same steady state
(`n = 300` states; SSJ horizon `T = 300`; MIT horizon `T = 120`):

| method | median time | memory | speedup | output |
|---|---:|---:|---:|---|
| **SSJ** (sequence-space Jacobian) | **15.9 ms** | 20 MiB | **1750×** | price-path IRF |
| **FAME** (master equation) | ~0.4 s | 1.4 GiB | 70× | Impulse Value `v` + law of motion `T'+G` |
| **MIT** (nonlinear transition) | 27.8 s | 57 GiB | 1× | full nonlinear path |

Speedup is relative to the slowest method (MIT). SSJ is fastest because it needs
only one backward iteration plus a single `T×T` solve; FAME returns the richer
Impulse Value object (useful for welfare and stability analysis) at the cost of a
matrix fixed-point iteration; MIT re-solves household policies and re-clears the
market every period and is used only as a ground-truth check.

## The shock

A *genuine* wealth redistribution: a smooth compression of the asset distribution
toward its cross-sectional mean, `a ↦ a + κ(ā − a)` (savers → borrowers),
spread back onto the grid with the model's lottery. It conserves both population
and aggregate assets exactly (`Σ h₀ = 0`, `Σ aᵢ h₀ᵢ = 0`), so the bond market
only re-prices a reshuffling of the fixed asset stock.
