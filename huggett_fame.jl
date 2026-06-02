# =============================================================================
#  FAME for the discrete-time Huggett economy
#
#  Companion code to KrusellSmith_DiscreteTime.tex (Section "The Huggett variant").
#
#  Solves
#    (1) the steady state of a discrete-time Huggett model (EGM + bond-market
#        clearing on the bond price q, with 1/q = 1 + r), and
#    (2) the First-order Approximation to the Master Equation (FAME): the
#        Impulse Value v(x, xi) = dV(x)/dg(xi), solved as the coupled fixed point
#
#            V = Dtilde(V) + beta * T * V * (T' + G(V)),
#
#        where T is the steady-state transition operator, T' its push-forward,
#        and G the general-equilibrium kernel.  For Huggett everything runs
#        through the single implicit bond price q = Q(g), so both Dtilde and G
#        depend on V (the price impact is part of the fixed point).
#
#  Only the Julia standard library is used (LinearAlgebra, Printf).
#  Tested on Julia 1.12.
# =============================================================================

using LinearAlgebra
using Printf
using Random
using Plots
using QuantEcon: rouwenhorst

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
Base.@kwdef struct Huggett
    β::Float64   = 0.96        # discount factor
    γ::Float64   = 2.0         # CRRA coefficient
    B::Float64   = 0.0         # net bond supply (0 = pure Huggett)
    amin::Float64 = -2.0       # (ad hoc) borrowing limit
    amax::Float64 = 40.0       # top of the asset grid
    na::Int      = 100         # number of asset grid points
    # income process: log y_t = ρ log y_{t-1} + σ ε_t, discretized by Rouwenhorst
    ρ::Float64   = 0.90
    σ::Float64   = 0.20
    ny::Int      = 3
end

# -----------------------------------------------------------------------------
# Income process: Rouwenhorst discretization (via QuantEcon) of the log-AR(1)
#   log y_t = ρ log y_{t-1} + σ ε_t.
# QuantEcon.rouwenhorst(n, ρ, σ) returns a MarkovChain whose state_values are
# the log-income nodes and whose `p` is the row-stochastic transition matrix.
# We exponentiate and rescale so that mean income equals 1.
# -----------------------------------------------------------------------------
function income_process(ρ, σ, n)
    mc = rouwenhorst(n, ρ, σ)          # QuantEcon MarkovChain
    Π  = Matrix(mc.p)                  # row-stochastic
    y  = exp.(collect(mc.state_values))
    π  = stationary(Matrix(Π'))        # normalize mean income to 1
    y ./= dot(π, y)
    return y, Π
end

# stationary distribution of a column-stochastic matrix P (P*g = g).
# g spans the null space of (I - P); since that matrix is singular we replace
# one (redundant) equation with the normalization Σ g = 1 and solve once.
function stationary(P)
    n = size(P, 1)
    A = Matrix(I - P)
    A[end, :] .= 1.0
    b = zeros(n); b[end] = 1.0
    return A \ b
end

# -----------------------------------------------------------------------------
# Grids and indexing
# -----------------------------------------------------------------------------
make_agrid(p::Huggett) = p.amin .+ (p.amax - p.amin) .* (range(0, 1; length = p.na) .^ 2)

idx(ia, iy, p::Huggett) = (iy - 1) * p.na + ia      # flatten (a,y) -> 1..na*ny

# linear interpolation bracket on a sorted grid xs for value x:
# returns (k, w) so that x ≈ xs[k] + w*(xs[k+1]-xs[k]),  k in 1..length-1
function bracket(xs, x)
    n = length(xs)
    k = searchsortedlast(xs, x)
    k = clamp(k, 1, n - 1)
    w = (x - xs[k]) / (xs[k+1] - xs[k])
    return k, w
end

# -----------------------------------------------------------------------------
# One backward EGM step.
#   Given the marginal continuation value EWa(a',y) = β E[u'(c_next)|y] on the
#   a'-grid (held FIXED), and a current bond price q, return the policies
#   c(a,y), a'(a,y) on the current-a grid.
#   This is exactly the asset-demand function A(x; q, g): the continuation is
#   fixed while the spot price q enters the current budget c + q a' = a + y.
# -----------------------------------------------------------------------------
function egm_policy(EWa, q, p::Huggett, agrid)
    na, ny = p.na, p.ny
    c  = zeros(na, ny)
    ap = zeros(na, ny)
    for iy in 1:ny
        # q u'(c) = EWa(a',y)  =>  c*(a') = (EWa/q)^(-1/γ)   on the a'-grid
        cstar  = (EWa[:, iy] ./ q) .^ (-1 / p.γ)
        aendog = cstar .+ q .* agrid .- yval(p, agrid, iy)     # a = c + q a' - y
        for ia in 1:na
            a = agrid[ia]
            if a <= aendog[1]                # borrowing constraint binds: a' = amin
                apol = agrid[1]
                c[ia, iy]  = a + yval(p, agrid, iy) - q * apol
                ap[ia, iy] = apol
            else
                k, w = bracket(aendog, a)
                apol = agrid[k] + w * (agrid[k+1] - agrid[k])
                apol = clamp(apol, agrid[1], agrid[end])
                c[ia, iy]  = a + yval(p, agrid, iy) - q * apol
                ap[ia, iy] = apol
            end
        end
    end
    return c, ap
end

# income level helper (kept as a function so the grid object stays simple)
yval(p::Huggett, agrid, iy) = YGRID[][iy]
const YGRID = Ref{Vector{Float64}}()      # set once in solve_steady

# marginal continuation value from a consumption policy
function continuation_EWa(c, Π, p::Huggett)
    na, ny = p.na, p.ny
    muc = c .^ (-p.γ)
    EWa = zeros(na, ny)
    for iy in 1:ny, ia in 1:na
        s = 0.0
        for jy in 1:ny
            s += Π[iy, jy] * muc[ia, jy]
        end
        EWa[ia, iy] = p.β * s
    end
    return EWa
end

# -----------------------------------------------------------------------------
# Solve the household problem at a fixed price q (iterate EGM to convergence)
# -----------------------------------------------------------------------------
function solve_household(q, Π, p::Huggett, agrid)
    na, ny = p.na, p.ny
    c = repeat(max.(agrid .* 0.0 .+ 0.5, agrid .+ 1.0), 1, ny)   # crude guess
    ap = similar(c)
    for it in 1:5_000
        EWa = continuation_EWa(c, Π, p)
        cnew, ap = egm_policy(EWa, q, p, agrid)
        d = maximum(abs.(cnew .- c))
        c = cnew
        if d < 1e-11
            break
        end
    end
    return c, ap
end

# -----------------------------------------------------------------------------
# Transition matrix T (row-stochastic, n×n) implied by policy ap, via the
# standard lottery (linear interpolation of a' onto the asset grid).
# -----------------------------------------------------------------------------
function make_T(ap, Π, p::Huggett, agrid)
    na, ny = p.na, p.ny
    n = na * ny
    T = zeros(n, n)
    for iy in 1:ny, ia in 1:na
        i = idx(ia, iy, p)
        k, w = bracket(agrid, ap[ia, iy])
        w = clamp(w, 0.0, 1.0)
        for jy in 1:ny
            pr = Π[iy, jy]
            T[i, idx(k,   jy, p)] += pr * (1 - w)
            T[i, idx(k+1, jy, p)] += pr * w
        end
    end
    return T
end

aggregate_assets(g, p::Huggett, agrid) =
    sum(g[idx(ia, iy, p)] * agrid[ia] for ia in 1:p.na, iy in 1:p.ny)

# -----------------------------------------------------------------------------
# Steady state: bisection on the bond price q to clear the bond market.
#   A(q) is decreasing in q (a higher bond price = lower return = less saving).
# -----------------------------------------------------------------------------
function solve_steady(p::Huggett)
    agrid = make_agrid(p)
    y, Π = income_process(p.ρ, p.σ, p.ny)
    YGRID[] = y
    excess(q) = begin
        c, ap = solve_household(q, Π, p, agrid)
        T = make_T(ap, Π, p, agrid)
        g = stationary(Matrix(T'))
        aggregate_assets(g, p, agrid) - p.B
    end
    qlo, qhi = p.β * (1 + 1e-4), 1.30          # q>β needed for a stationary dist
    flo, fhi = excess(qlo), excess(qhi)
    @assert flo > 0 && fhi < 0 "market clearing not bracketed: f(qlo)=$flo f(qhi)=$fhi"
    q = 0.5 * (qlo + qhi)
    for _ in 1:80
        f = excess(q)
        if abs(f) < 1e-9
            break
        end
        f > 0 ? (qlo = q) : (qhi = q)
        q = 0.5 * (qlo + qhi)
    end
    c, ap = solve_household(q, Π, p, agrid)
    T = make_T(ap, Π, p, agrid)
    g = stationary(Matrix(T'))
    return (; q, r = 1/q - 1, c, ap, T, g, Π, agrid, y)
end

# -----------------------------------------------------------------------------
# Push-forward derivative operator M (n×n):  M[m,i] = dΦ_m / d a'_i  (incl. g_i).
#   Increasing the saving target a'_i shifts the lottery mass g_i from the lower
#   to the upper bracketing grid node at rate 1/Δa.
# -----------------------------------------------------------------------------
function make_M(ap, g, Π, p::Huggett, agrid)
    na, ny = p.na, p.ny
    n = na * ny
    M = zeros(n, n)
    for iy in 1:ny, ia in 1:na
        i = idx(ia, iy, p)
        k, _ = bracket(agrid, ap[ia, iy])
        Δ = agrid[k+1] - agrid[k]
        gi = g[i]
        for jy in 1:ny
            pr = Π[iy, jy] * gi / Δ
            M[idx(k,   jy, p), i] -= pr
            M[idx(k+1, jy, p), i] += pr
        end
    end
    return M
end

# derivative of v along the asset dimension of its FIRST argument (central diff)
function dv_da(v, p::Huggett, agrid)
    na, ny = p.na, p.ny
    n = size(v, 2)
    va = zeros(size(v))
    for iy in 1:ny
        for ia in 1:na
            i = idx(ia, iy, p)
            ip = idx(min(ia+1, na), iy, p)
            im = idx(max(ia-1, 1),  iy, p)
            da = agrid[min(ia+1, na)] - agrid[max(ia-1, 1)]
            @views va[i, :] .= (v[ip, :] .- v[im, :]) ./ da
        end
    end
    return va
end

# -----------------------------------------------------------------------------
# FAME: solve the coupled fixed point for the Impulse Value v (n×n) and the
# linearized law of motion DΦ = T' + G.
# -----------------------------------------------------------------------------
function solve_fame(ss; tol = 1e-9, maxit = 2_000, verbose = true)
    p_local = PARAMS[]
    agrid = ss.agrid
    na, ny = p_local.na, p_local.ny
    n = na * ny
    β, γ, q = p_local.β, p_local.γ, ss.q

    cflat  = vec(ss.c)
    apflat = vec(ss.ap)
    g      = ss.g
    T      = ss.T

    # --- policy sensitivity to the CURRENT price, continuation fixed: ψ = ∂a'/∂q
    EWa_ss = continuation_EWa(ss.c, ss.Π, p_local)
    dq = 1e-5
    _, ap_p = egm_policy(EWa_ss, q + dq, p_local, agrid)
    _, ap_m = egm_policy(EWa_ss, q - dq, p_local, agrid)
    dap_dq = vec((ap_p .- ap_m) ./ (2dq))                 # ψ_i < 0 (demand slopes down)

    # constrained households: policy pinned at the borrowing limit
    constrained = vec(ss.ap) .<= (agrid[1] + 1e-8)
    dap_dq[constrained] .= 0.0

    # curvature denominator  denom_i = -(u'(c)-q u''(c)a') / ψ_i  (>0, unconstrained)
    uc  = cflat .^ (-γ)
    ucc = -γ .* cflat .^ (-γ - 1)
    num = uc .- q .* ucc .* apflat                   # = u'(c) - q u''(c) a'
    denom = fill(Inf, n)
    @inbounds for i in 1:n
        if !constrained[i] && dap_dq[i] != 0.0
            denom[i] = -num[i] / dap_dq[i]
        end
    end

    M  = make_M(ss.ap, g, ss.Π, p_local, agrid)
    bD = -(uc .* apflat)                             # b_i = -u'(c_i) a'_i   (for Dtilde)
    g_dap_dq = dot(g, dap_dq)                                   # aggregate demand slope (<0)

    v   = zeros(n, n)
    DΦ  = Matrix(T')
    Qp  = -(apflat) ./ g_dap_dq                           # initial price impact (B_future=0)

    local err
    for it in 1:maxit
        va = dv_da(v, p_local, agrid)

        # response of policy to the distribution through FUTURE prices, q fixed
        Bfut = β .* (T * va * DΦ)
        @inbounds for i in 1:n
            if isfinite(denom[i])
                @views Bfut[i, :] ./= denom[i]
            else
                @views Bfut[i, :] .= 0.0
            end
        end

        # price impact of mass at ξ:  Q'_j = -(a'_j + Σ_i g_i Bfut[i,j]) / (g·dap_dq)
        Qp = -(apflat .+ (Bfut' * g)) ./ g_dap_dq

        # total policy response to the impulse and the GE kernel
        dadg = dap_dq * Qp' .+ Bfut                       # da'_i/dg_j  (n×n)
        G    = M * dadg
        DΦ   = Matrix(T') .+ G

        # direct price impact (rank one) and the FAME update
        Dtil = bD * Qp'                              # Dtilde[i,j] = -u'(c_i)a'_i Q'_j
        vnew = Dtil .+ β .* (T * v * DΦ)

        err = maximum(abs.(vnew .- v))
        v = vnew
        if verbose && (it % 25 == 0 || it == 1)
            @printf("  FAME iter %4d   ‖Δv‖=%.3e\n", it, err)
        end
        err < tol && break
    end
    return (; v, DΦ, Qp, dap_dq, G = DΦ .- Matrix(T'))
end

# global handle so helper closures can see the parameters/grids
const PARAMS = Ref{Huggett}()

# Nonlinear perfect-foresight (MIT-shock) transition `mit_transition`, used only
# to validate the FAME.  Kept in a separate file; it shares this module scope
# and relies on PARAMS, egm_policy, continuation_EWa, make_T defined above.
include("huggett_mit.jl")

# -----------------------------------------------------------------------------
# Validation of the contemporaneous price impact.
# Temporary equilibrium with the continuation FIXED at steady state: solve for
# the bond price that clears the market at g = g^ss + ε h, then compare the
# numerical dq/dε to the analytic Q'_simple · h  (= Q' without the future term).
# -----------------------------------------------------------------------------
function validate_priceimpact(ss, fame)
    p = PARAMS[]; agrid = ss.agrid; n = p.na * p.ny
    EWa_ss = continuation_EWa(ss.c, ss.Π, p)
    # market-clearing price for distribution dist, continuation fixed
    function excess_q(q, dist)
        _, ap = egm_policy(EWa_ss, q, p, agrid)
        return dot(vec(ap), dist) - p.B
    end
    function clear_q(dist)
        qlo, qhi = p.β * (1 + 1e-4), 1.30
        q = 0.5 * (qlo + qhi)
        for _ in 1:100
            fq = excess_q(q, dist)
            abs(fq) < 1e-12 && break
            fq > 0 ? (qlo = q) : (qhi = q)
            q = 0.5 * (qlo + qhi)
        end
        return q
    end
    # a (reproducible) random mass-preserving direction, supported on interior
    # states (the policy is kinked at the borrowing limit, which would otherwise
    # add an O(ε) truncation error to the central finite difference).
    Random.seed!(1)
    constrained = vec(ss.ap) .<= (agrid[1] + 1e-8)
    h = randn(n); h[constrained] .= 0.0; h .-= sum(h) / n
    g_dap_dq = dot(ss.g, fame.dap_dq)
    Qsimple = -(vec(ss.ap)) ./ g_dap_dq
    ε = 1e-6
    dq_num = (clear_q(ss.g .+ ε .* h) - clear_q(ss.g .- ε .* h)) / (2ε)
    dq_an  = dot(Qsimple, h)
    @printf("\n  price-impact check (continuation fixed):\n")
    @printf("    finite-difference dq/dε = %+.6e\n", dq_num)
    @printf("    analytic  Q'_simple·h   = %+.6e   (rel.err %.2e)\n",
            dq_an, abs(dq_num - dq_an) / max(abs(dq_num), 1e-14))
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
function main(p::Huggett = Huggett(); run_mit = true)
    PARAMS[] = p
    println("="^70)
    println(" Discrete-time Huggett: steady state")
    println("="^70)
    ss = solve_steady(p)
    n = p.na * p.ny
    frac_constr = sum(ss.g[i] for i in 1:n if vec(ss.ap)[i] <= ss.agrid[1] + 1e-8)
    @printf("  bond price   q* = %.6f\n", ss.q)
    @printf("  interest     r* = %.6f   (annual)\n", ss.r)
    @printf("  aggregate bonds A = %.3e   (target B = %.3f)\n",
            aggregate_assets(ss.g, p, ss.agrid), p.B)
    @printf("  fraction at borrowing limit = %.3f\n", frac_constr)

    println("\n", "="^70)
    println(" FAME: solving for the Impulse Value v(x,ξ) = ∂V/∂g")
    println("="^70)
    fame = solve_fame(ss)

    # Stability: (T'+G) always has a trivial unit eigenvalue (mass conservation,
    # left-eigenvector = ones).  Mass-preserving deviations (Σh=0) are governed by
    # the remaining spectrum, so the relevant object is the 2nd-largest modulus.
    λ = sort(abs.(eigvals(fame.DΦ)); rev = true)
    @printf("\n  largest eigenvalue of (T'+G)      = %.6f  (mass conservation)\n", λ[1])
    @printf("  2nd-largest |eigenvalue|          = %.6f  %s\n", λ[2],
            λ[2] < 1 ? "(stable; sets the decay rate)" : "(UNSTABLE)")

    # ----- Validation: price impact Q' against a direct finite difference -----
    # Hold the continuation at steady state and recompute the market-clearing
    # bond price for g = g^ss + ε h.  d q/dε should equal Q'_simple · h, where
    # Q'_simple = -a'/(g·dap_dq) is Q' without the future-feedback term.
    validate_priceimpact(ss, fame)

    # --- Impulse response to a distributional shock --------------------------
    # h0: move a small mass ε from a low-asset state to a high-asset state
    # (mass-preserving:  Σ h0 = 0).
    agrid = ss.agrid
    iy0 = 1
    i_lo = idx(searchsortedfirst(agrid, 0.0),            iy0, p)
    i_hi = idx(searchsortedfirst(agrid, 5.0),            iy0, p)
    ε = 1e-3
    h0 = zeros(n); h0[i_hi] += ε; h0[i_lo] -= ε

    Tmax = 40
    dq = zeros(Tmax); dr = zeros(Tmax)
    h = copy(h0)
    for t in 1:Tmax
        dq[t] = dot(fame.Qp, h)            # bond-price deviation (FAME, linear)
        dr[t] = -dq[t] / ss.q^2            # since r = 1/q - 1,  dr = -dq/q²
        h = fame.DΦ * h
    end

    println("\n  Impulse response to a one-time wealth-redistribution shock")
    println("  (mass ε from a≈0 to a≈5 among low-income households):")
    println("    t=0 is the contemporaneous impact; t≥1 is propagation via (T'+G).")
    dr_mit = Float64[]
    if run_mit
        # Nonlinear MIT-shock transition for the SAME impulse (validation)
        dq_nl = mit_transition(ss, h0; Tmax = 120)
        dr_mit = -1e4 .* dq_nl[1:Tmax] ./ ss.q^2     # MIT interest response (bps)
        println("    FAME = linear Impulse-Value prediction; MIT = nonlinear transition.")
        @printf("    t      d r_t  FAME (bps)     d r_t  MIT (bps)\n")
        for t in 1:8
            @printf("   %2d        %+9.4f          %+9.4f\n",
                    t-1, 1e4 * dr[t], -1e4 * dq_nl[t] / ss.q^2)
        end
        @printf("    max|FAME-MIT| over t≤40 = %.3e bps\n",
                1e4 * maximum(abs.(dr .+ dq_nl[1:Tmax] ./ ss.q^2)))
    else
        @printf("    t      d r_t  FAME (bps)\n")
        for t in 1:8
            @printf("   %2d        %+9.4f\n", t-1, 1e4 * dr[t])
        end
    end

    # --- Plot the interest-rate impulse response (FAME vs MIT) ---------------
    # Two panels: the full response (dominated by the t=0 impact) and a zoom on
    # the propagation tail (t>=1), where the FAME-vs-MIT comparison is sharpest.
    tgrid = 0:(Tmax - 1)
    p1 = plot(tgrid, 1e4 .* dr;
              label = "FAME (linear)", lw = 2, marker = :circle, ms = 3,
              xlabel = "period t", ylabel = "Δr_t  (bps, annual)",
              title = "Full response (incl. t=0 impact)",
              legend = :bottomright, framestyle = :box)
    p2 = plot(tgrid[2:end], 1e4 .* dr[2:end];
              label = "FAME (linear)", lw = 2, marker = :circle, ms = 3,
              xlabel = "period t", ylabel = "Δr_t  (bps, annual)",
              title = "Propagation tail (t ≥ 1)",
              legend = :topright, framestyle = :box)
    if run_mit
        plot!(p1, tgrid, dr_mit;
              label = "MIT (nonlinear)", lw = 2, ls = :dash, marker = :diamond, ms = 3)
        plot!(p2, tgrid[2:end], dr_mit[2:end];
              label = "MIT (nonlinear)", lw = 2, ls = :dash, marker = :diamond, ms = 3)
    end
    hline!(p1, [0.0]; label = "", lc = :black, lw = 0.5, alpha = 0.5)
    hline!(p2, [0.0]; label = "", lc = :black, lw = 0.5, alpha = 0.5)
    plt = plot(p1, p2; layout = (1, 2), size = (1000, 420),
               plot_title = "Interest-rate response to a wealth-redistribution shock")
    outpng = joinpath(@__DIR__, "huggett_r_response.png")
    savefig(plt, outpng)
    @printf("\n  saved interest-rate impulse-response plot to %s\n", outpng)
    # persistence is governed by the 2nd-largest eigenvalue modulus of (T'+G)
    λ2 = sort(abs.(eigvals(fame.DΦ)); rev = true)[2]
    @printf("    propagation decay rate (per period) ≈ %.4f  (half-life ≈ %.1f periods)\n",
            λ2, log(0.5) / log(λ2))

    # --- Welfare value of the impulse:  ΔV(x) = (v h0)(x) --------------------
    ΔV = fame.v * h0
    @printf("\n  Welfare effect ΔV = v·h0:  min %.3e   max %.3e\n",
            minimum(ΔV), maximum(ΔV))

    return ss, fame
end

main()


p = Huggett()

ss = solve_steady(p)

using BenchmarkTools
@time fame = solve_fame(ss)