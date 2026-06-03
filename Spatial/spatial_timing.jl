# =============================================================================
#  Timing benchmark: FAME vs SSJ as the number of locations J grows.
#
#  Both methods share the same steady state, so we time only the LINEARIZATION
#  step that differs:
#    * SSJ  : fake-news household Jacobians + GE assembly + sequence-space solve
#             (cost driven by the (1+2J)·T square system),
#    * FAME : household + continuation Jacobians + the coupled master-equation
#             fixed point (cost driven by n×n operations, n = n_a · J).
#  The horizon T is fixed; the asset grid n_a is fixed; only J varies.
#
#  Run:  julia spatial_timing.jl   ->  figure/spatial_timing.pdf
# =============================================================================

if !isdefined(Main, :SpatialHuggett)
    include("SpatialHuggett.jl")
end
using .SpatialHuggett
include("spatial_ssj.jl")
include("spatial_fame.jl")
using Plots, Printf

const TH = 150          # SSJ horizon
const NA = 80           # asset grid points (fixed across J)
const JS = [2, 3, 4, 5, 6]

time_ssj(ss, p) = @elapsed solve_ssj(ss, p; T = TH, mshock = 1, z0 = -0.01, ρz = 0.8)

function time_fame(ss, p)
    @elapsed begin
        sp   = static_partials(ss, p)
        sens = household_sensitivities(ss, p)
        pim  = price_impact(ss, p, sens, sp)
        cj   = continuation_jacobians(ss, p)
        fame_full(ss, p, sens, cj, pim, sp; mshock = 1, ρz = 0.8)
    end
end

# ---- warm up the JIT (compile everything once, at the smallest J) ----
println("warming up (JIT compilation) ...")
let p = SParams(J = 2, na = NA)
    ss = solve_steady(p; verbose = false)
    time_ssj(ss, p); time_fame(ss, p)
end

# ---- benchmark ----
ns = Int[]; tssj = Float64[]; tfame = Float64[]
@printf("\n  %-3s %-6s %12s %12s\n", "J", "n", "SSJ (s)", "FAME (s)")
for J in JS
    p  = SParams(J = J, na = NA)
    ss = solve_steady(p; verbose = false)
    GC.gc()
    ts = minimum(time_ssj(ss, p)  for _ in 1:2)
    GC.gc()
    tf = minimum(time_fame(ss, p) for _ in 1:2)
    push!(ns, n_states(p)); push!(tssj, ts); push!(tfame, tf)
    @printf("  %-3d %-6d %12.3f %12.3f\n", J, n_states(p), ts, tf)
end

# ---- plot (linear + log scale) ----
p1 = plot(JS, tssj; marker = :circle, ms = 5, lw = 2.5, color = :steelblue, label = "SSJ",
          xlabel = "number of locations J", ylabel = "time (s)",
          title = "Linearization time (linear scale)", legend = :topleft)
plot!(p1, JS, tfame; marker = :square, ms = 5, lw = 2.5, color = :crimson, label = "FAME")

p2 = plot(JS, tssj; marker = :circle, ms = 5, lw = 2.5, color = :steelblue, label = "SSJ",
          xlabel = "number of locations J", ylabel = "time (s, log scale)", yscale = :log10,
          title = "Linearization time (log scale)", legend = :topleft)
plot!(p2, JS, tfame; marker = :square, ms = 5, lw = 2.5, color = :crimson, label = "FAME")

plt = plot(p1, p2; layout = (1, 2), size = (1050, 420),
           plot_title = @sprintf("FAME vs SSJ timing  (n_a=%d, T=%d, n = n_a·J)", NA, TH))
outpdf = joinpath(@__DIR__, "..", "figure", "spatial_timing.pdf")
savefig(plt, outpdf)
@printf("\n  saved timing plot to %s\n", normpath(outpdf))
