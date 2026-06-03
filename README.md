# Master equation in discrete time — Huggett example

[Lecture note link](https://masaofukui.github.io/741_Fukui_2025_topic10.pdf)

The code
solves the steady state of a discrete-time Huggett economy (idiosyncratic income,
a single implicitly priced bond with `1/q = 1 + r`) and computes the interest-rate
impulse response to a wealth-redistribution shock **three independent ways**:

- **FAME (discrete-time)** — First-order Approximation to the Master Equation: solves for the
  Impulse Value `v(x,ξ) = ∂V(x)/∂g(ξ)` and the linearized law of motion `T' + G`. This is Discrete-time companion to the continuous-time master-equation framework of Bilal, *Solving Heterogeneous Agent Models with the Master Equation*. 
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
| `figure/` | Generated figures (`huggett_r_response_kappa*.pdf`, and the spatial outputs `spatial_irf_compare.pdf`, `spatial_timing.pdf` + `spatial_timing.csv`). |
| `Spatial/` | Multi-region extension (trade + migration); see the section below. |

The `document/` folder (LaTeX derivation `KrusellSmith_DiscreteTime.tex` + compiled
PDF) and `literature/` (reference papers) are kept locally but are **git-ignored**.

## Running

```bash
julia toplevel_huggett.jl     # solve + compare + save figure/huggett_r_response_kappa*.pdf
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


## Spatial extension (multi-region Huggett with trade & migration)

The `Spatial/` folder scales the same FAME-vs-SSJ comparison up to a richer model:
a multi-region Huggett economy with savings, CES goods **trade** across regions,
and logit (type-I EV) **migration** — the status-quo economy of Donald, Fukui &
Miyauchi (2025, *Optimal Dynamic Spatial Policy*), specialized to infinitely-lived
households. The aggregate state is the distribution over `(assets a, region j)`;
within each period `J` prices clear (the bond rate `r` and wages `w_2,…,w_J`, with
`w_1` the numeraire), while populations `L_j` and CES price indices `P_j` follow
from the distribution.

| path | description |
|---|---|
| `Spatial/SpatialHuggett.jl` | Shared model module: parameters/grids, CES trade block, EGM household problem with logit migration, the `(a,j)` transition operator, and the nested steady-state fixed point. |
| `Spatial/spatial_fame.jl` | FAME for the spatial model: static/household/continuation Jacobians, the price-impact operator `Q = dp/dg`, the GE kernel `G` and law of motion `T'+G`, and the coupled fixed point `fame_full`. |
| `Spatial/spatial_ssj.jl` | Sequence-space Jacobian (fake-news) for the spatial model, the GE system over `(r_t, w_jt, L_jt)`, and a nonlinear perfect-foresight transition. |
| `Spatial/spatial_irf_compare.jl` | Overlays the SSJ and FAME IRFs of **every region** to a productivity shock (colour = region, solid = SSJ, dashed+markers = FAME). → `figure/spatial_irf_compare.pdf`. |
| `Spatial/spatial_timing.jl` | Times FAME vs SSJ linearization as the number of regions `J` grows, appending results to `figure/spatial_timing.csv` as they are computed. |
| `Spatial/spatial_timing_plot.jl` | Reads `figure/spatial_timing.csv` and draws the timing comparison (linear + log panels) → `figure/spatial_timing.pdf`, so the figure can be restyled without re-running the benchmark. |

```bash
julia Spatial/spatial_irf_compare.jl   # per-region IRFs, FAME vs SSJ
julia Spatial/spatial_timing.jl        # benchmark -> figure/spatial_timing.{csv,pdf}
julia Spatial/spatial_timing_plot.jl   # redraw figure from the stored CSV
```

**Scaling.** Both methods linearize the same model around the same steady state and
agree to truncation/numerical error. FAME's cost grows much faster in `J`: it forms
`J` dense `n×n` migration Jacobians and does `J` `n×n` matmuls per fixed-point
iteration (`n = n_a·J`), whereas SSJ solves a `(1+2J)·T` sequence-space system. On
the benchmark grid (`n_a = 80`, `T = 150`) the FAME/SSJ linearization times were:

| J | n | SSJ | FAME |
|---:|---:|---:|---:|
| 2 | 160 | 0.07 s | 0.36 s |
| 5 | 400 | 0.74 s | 3.3 s |
| 10 | 800 | 5.2 s | 20.5 s |
| 20 | 1600 | 38.5 s | 141.6 s |

(`spatial_timing.jl` has a wall-clock budget that stops *attempting* FAME at larger
`J` once a run exceeds it, recording `NaN`; SSJ is always timed.)

## Notes
Claude Opus 4.8 was used in designing and implementing the code.