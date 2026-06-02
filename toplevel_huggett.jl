# =============================================================================
#  Toplevel driver for the discrete-time Huggett economy.
#
#  Loads every function library, then for a single wealth-redistribution shock
#  computes the interest-rate impulse response three ways and overlays them:
#
#     FAME  -- First-order Approximation to the Master Equation   (huggett_fame.jl)
#     SSJ   -- Sequence-Space Jacobian / fake-news algorithm      (huggett_ssj.jl)
#     MIT   -- fully nonlinear perfect-foresight transition       (huggett_mit.jl)
#
#  The combined figure is saved as figure/huggett_r_response.pdf.
#
#  Run with:   julia run_huggett.jl
# =============================================================================

include("huggett_fame.jl")   # model, steady state, EGM, FAME, redistribution_shock, fame_irf
include("huggett_mit.jl")    # nonlinear MIT transition, mit_irf_bps
include("huggett_ssj.jl")    # sequence-space Jacobian, ssj_irf_bps

function run_huggett(p::Huggett = Huggett(); Tshow::Int = 25, Tssj::Int = 300)
    PARAMS[] = p

    # ---- Steady state ----
    println("="^70)
    println(" Discrete-time Huggett: steady state")
    println("="^70)
    ss = solve_steady(p)
    n  = p.na * p.ny
    frac_constr = sum(ss.g[i] for i in 1:n if vec(ss.ap)[i] <= ss.agrid[1] + 1e-8)
    @printf("  bond price   q* = %.6f\n", ss.q)
    @printf("  interest     r* = %.6f   (annual)\n", ss.r)
    @printf("  aggregate bonds A = %.3e   (target B = %.3f)\n",
            aggregate_assets(ss.g, p, ss.agrid), p.B)
    @printf("  fraction at borrowing limit = %.3f\n", frac_constr)

    # ---- FAME ----
    println("\n", "="^70)
    println(" FAME: Impulse Value v(x,ξ) = ∂V/∂g  and law of motion (T'+G)")
    println("="^70)
    fame = solve_fame(ss)
    DΦ = Matrix(ss.T') .+ fame.G              # linearized law of motion (T'+G)
    λ = sort(abs.(eigvals(DΦ)); rev = true)
    @printf("\n  eigenvalues of (T'+G): largest = %.6f (mass conservation), 2nd = %.6f %s\n",
            λ[1], λ[2], λ[2] < 1 ? "(stable)" : "(UNSTABLE)")

    # ---- Same impulse for all three methods ----
    κshock = 0.5                                # compress assets 1% toward the mean
    h0 = redistribution_shock(ss, p; κ = κshock)

    dr_fame = fame_irf(ss, fame, h0; Tshow = Tshow)            # master equation
    dr_ssj  = ssj_irf_bps(ss, h0; Tshow = Tshow, T = Tssj)     # sequence-space Jacobian
    dr_mit  = mit_irf_bps(ss, h0; Tshow = Tshow)               # nonlinear transition

    # ---- Comparison table ----
    println("\n  Interest-rate IRF Δr_t (bps, annual) to a wealth-redistribution shock")
    @printf("    t        FAME           SSJ            MIT\n")
    for t in 1:8
        @printf("   %2d     %+10.4f     %+10.4f     %+10.4f\n",
                t-1, dr_fame[t], dr_ssj[t], dr_mit[t])
    end
    @printf("    max|FAME-SSJ| = %.2e   max|FAME-MIT| = %.2e  (bps, t<%d)\n",
            maximum(abs.(dr_fame .- dr_ssj)),
            maximum(abs.(dr_fame .- dr_mit)), Tshow)

    # ---- One combined plot (interest rate in percentage points) ----
    tg  = 0:(Tshow - 1)
    pp  = 1 / 100                                   # bps -> percentage points
    plt = plot(tg, dr_fame .* pp;
               label = "FAME (master equation)", lw = 4, marker = :circle, ms = 6,
               xlabel = "period \$t\$", ylabel = "\$ \\Delta r_t \$  (pp, annual)",
               title = "Interest-rate response to a wealth redistribution (\$\\kappa\$ = $(κshock))",
               legend = :topright, framestyle = :box, size = (900, 520),
               fontfamily = "Computer Modern",
               guidefontsize = 15, tickfontsize = 13, legendfontsize = 13, titlefontsize = 13,
               left_margin = 8Plots.mm, bottom_margin = 8Plots.mm,
               right_margin = 6Plots.mm, top_margin = 4Plots.mm, grid=:y)
    plot!(plt, tg, dr_ssj .* pp;
          label = "SSJ (sequence-space Jacobian)", lw = 4, ls = :dash, marker = :diamond, ms = 6)
    plot!(plt, tg, dr_mit .* pp;
          label = "MIT (nonlinear transition)", lw = 4, ls = :dot, marker = :utriangle, ms = 6)
    hline!(plt, [0.0]; label = "", lc = :black, lw = 1, alpha = 0.5)

    figdir = joinpath(@__DIR__, "figure")
    mkpath(figdir)
    outpdf = joinpath(figdir, "huggett_r_response_kappa$(κshock).pdf")
    savefig(plt, outpdf)
    @printf("\n  saved combined interest-rate IRF to %s\n", outpdf)

    return (; ss, fame, h0, dr_fame, dr_ssj, dr_mit)
end

run_huggett()
