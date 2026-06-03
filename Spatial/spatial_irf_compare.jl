# =============================================================================
#  Overlay the SSJ and FAME impulse responses to a productivity shock in
#  region 1, for the multi-region Huggett economy.
#
#  Both methods linearize the SAME model around the SAME steady state:
#    * SSJ  (spatial_ssj.jl)  solves the GE transition in sequence space,
#    * FAME (spatial_fame.jl) iterates the state-space law of motion DΦ with the
#      shock's direct loadings (Q_Z, Φ_Z).
#
#  Run:  julia spatial_irf_compare.jl    ->  figure/spatial_irf_compare.pdf
# =============================================================================

if !isdefined(Main, :SpatialHuggett)
    include("SpatialHuggett.jl")
end
using .SpatialHuggett
include("spatial_ssj.jl")
include("spatial_fame.jl")
using Plots, Printf, LinearAlgebra

const MSHOCK = 1
const Z0     = -0.01      # -1% productivity in region 1
const RHOZ   = 0.8
const TH     = 150
const TSHOW  = 30

p  = SParams()
println("solving steady state ...")
ss = solve_steady(p; verbose = false)
@printf("  r=%.5f  w=%s  L=%s\n", ss.r,
        string(round.(ss.w, digits=3)), string(round.(ss.L, digits=3)))

# ---- SSJ ----
println("solving SSJ ...")
ssj = solve_ssj(ss, p; T = TH, mshock = MSHOCK, z0 = Z0, ρz = RHOZ)

# ---- FAME (full master equation, with anticipation channel) ----
println("solving FAME (continuation Jacobians + coupled fixed point) ...")
sp   = static_partials(ss, p)
sens = household_sensitivities(ss, p)
pim  = price_impact(ss, p, sens, sp)
cj   = continuation_jacobians(ss, p)
ff   = fame_full(ss, p, sens, cj, pim, sp; mshock = MSHOCK, ρz = RHOZ)
fame = fame_full_irf(ss, p, ff; z0 = Z0, ρz = RHOZ, T = TH)

# ---- report a few periods ----
@printf("\n  %-4s %12s %12s %12s %12s\n", "t", "dr_SSJ(bp)", "dr_FAME(bp)",
        "dL1_SSJ(%)", "dL1_FAME(%)")
for t in 1:8
    @printf("  %-4d %12.4f %12.4f %12.4f %12.4f\n", t,
            1e4*ssj.dr[t], 1e4*fame.dr[t],
            100*ssj.dL[1][t]/ss.L[1], 100*fame.dL[1][t]/ss.L[1])
end

# ---- overlay plot ----
tt = 1:TSHOW
p1 = plot(tt, 1e4 .* ssj.dr[tt]; lw = 2.5, color = :steelblue, label = "SSJ",
          xlabel = "period t", ylabel = "Δr (bp)", title = "Bond rate")
plot!(p1, tt, 1e4 .* fame.dr[tt]; lw = 2, ls = :dash, color = :crimson,
      marker = :circle, ms = 2.5, label = "FAME")

p2 = plot(; xlabel = "period t", ylabel = "% deviation", title = "Population L₁ (shocked region)")
plot!(p2, tt, 100 .* ssj.dL[1][tt] ./ ss.L[1]; lw = 2.5, color = :steelblue, label = "SSJ")
plot!(p2, tt, 100 .* fame.dL[1][tt] ./ ss.L[1]; lw = 2, ls = :dash, color = :crimson,
      marker = :circle, ms = 2.5, label = "FAME")

p3 = plot(; xlabel = "period t", ylabel = "% deviation", title = "Wage w₂")
plot!(p3, tt, 100 .* ssj.dw[2][tt] ./ ss.w[2]; lw = 2.5, color = :steelblue, label = "SSJ")
plot!(p3, tt, 100 .* fame.dw[2][tt] ./ ss.w[2]; lw = 2, ls = :dash, color = :crimson,
      marker = :circle, ms = 2.5, label = "FAME")

p4 = plot(tt, 100 .* ssj.dZ[tt]; lw = 2, ls = :dot, color = :black, legend = false,
          xlabel = "period t", ylabel = "%", title = "Productivity shock (region 1)")

plt = plot(p1, p2, p3, p4; layout = (2, 2), size = (1000, 720),
           plot_title = "SSJ vs FAME: IRF to a -1% productivity shock in region 1")
outpdf = joinpath(@__DIR__, "..", "figure", "spatial_irf_compare.pdf")
savefig(plt, outpdf)
@printf("\n  saved overlay to %s\n", normpath(outpdf))
