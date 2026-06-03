# =============================================================================
#  Multi-region Huggett economy with migration, savings, and trade.
#
#  Status-quo economy of Donald-Fukui-Miyauchi (2025, "Optimal Dynamic Spatial
#  Policy"), specialized to infinitely-lived households (omega = 1) and no
#  government transfers.  See document/MultiRegionHuggett.tex.
#
#  This file provides the SHARED steady-state engine used by both the FAME
#  (spatial_fame.jl) and the sequence-space-Jacobian (spatial_ssj.jl) codes:
#    * household problem by EGM with logit (type-I EV) migration,
#    * the one-period transition operator T over the state (a, j),
#    * the static trade/price block (CES + iso-elastic agglomeration),
#    * the nested fixed point that clears the bond, labor, and goods markets.
#
#  Only the Julia standard library is used.  Tested on Julia 1.10+.
# =============================================================================

module SpatialHuggett

using LinearAlgebra
using Printf

export SParams, make_fundamentals, agrid, idx, n_states,
       price_index, trade_shares,
       backward_step, migration_probs, solve_household,
       build_T, stationary, aggregates,
       Steady, solve_steady

# -----------------------------------------------------------------------------
# Parameters and grids
# -----------------------------------------------------------------------------
Base.@kwdef struct SParams
    J::Int       = 3         # number of regions
    β::Float64   = 0.92      # discount factor (5-year period)
    γ::Float64   = 1.0       # CRRA (γ=1 ⇒ log)
    σ::Float64   = 5.0       # trade elasticity (>1)
    α::Float64   = 0.02      # agglomeration elasticity
    θ::Float64   = 2.5       # migration elasticity (logit scale)
    amin::Float64 = -0.08    # borrowing limit
    amax::Float64 = 15.0     # top of asset grid
    na::Int      = 100       # asset grid points
end

n_states(p::SParams) = p.na * p.J

# curved asset grid (denser near the borrowing limit)
agrid(p::SParams) = p.amin .+ (p.amax - p.amin) .* (range(0, 1; length = p.na) .^ 2)

# flatten (asset index ia, region j) -> 1..na*J
@inline idx(ia::Int, j::Int, p::SParams) = (j - 1) * p.na + ia

# CRRA helpers
@inline uprime(c, γ)     = c^(-γ)
@inline uprime_inv(x, γ) = x^(-1 / γ)
@inline util(c, γ)       = γ == 1 ? log(c) : (c^(1 - γ) - 1) / (1 - γ)

# -----------------------------------------------------------------------------
# Fundamentals.  A[k,j] = productivity of labor shipped from origin k to
# destination j (embeds origin productivity T[k] and iceberg trade cost d[k,j]).
# χ[j,k] = bilateral migration-cost shifter from j to k (χ[j,j]=1).
# -----------------------------------------------------------------------------
function make_fundamentals(p::SParams; T = nothing, dcost = 1.5, χoff = 0.3)
    J = p.J
    Tprod = T === nothing ?
        (J == 1 ? [1.0] : collect(range(0.9, 1.1; length = J))) : collect(T)
    A = zeros(J, J)
    for k in 1:J, j in 1:J
        A[k, j] = Tprod[k] / (k == j ? 1.0 : dcost)
    end
    χ = fill(χoff, J, J)
    for j in 1:J
        χ[j, j] = 1.0
    end
    return A, χ
end

# -----------------------------------------------------------------------------
# Static trade / price block
# -----------------------------------------------------------------------------
# CES price index  P_j = [ Σ_k (w_k / (A_kj L_k^α))^{1-σ} ]^{1/(1-σ)}
function price_index(w, L, A, p::SParams)
    J, σ, α = p.J, p.σ, p.α
    P = zeros(J)
    @inbounds for j in 1:J
        s = 0.0
        for k in 1:J
            s += (w[k] / (A[k, j] * L[k]^α))^(1 - σ)
        end
        P[j] = s^(1 / (1 - σ))
    end
    return P
end

# expenditure (trade) shares  π[i,j] = (w_i/(A_ij L_i^α))^{1-σ} / Σ_l (...)
function trade_shares(w, L, A, p::SParams)
    J, σ, α = p.J, p.σ, p.α
    π = zeros(J, J)
    @inbounds for j in 1:J
        d = 0.0
        for l in 1:J
            d += (w[l] / (A[l, j] * L[l]^α))^(1 - σ)
        end
        for i in 1:J
            π[i, j] = (w[i] / (A[i, j] * L[i]^α))^(1 - σ) / d
        end
    end
    return π
end

# -----------------------------------------------------------------------------
# linear interpolation of y on sorted grid xs at point x (with linear extrap)
# -----------------------------------------------------------------------------
@inline function interp1(xs, y, x)
    n = length(xs)
    if x <= xs[1]
        return y[1] + (y[2] - y[1]) * (x - xs[1]) / (xs[2] - xs[1])
    elseif x >= xs[n]
        return y[n] + (y[n] - y[n-1]) * (x - xs[n]) / (xs[n] - xs[n-1])
    end
    k = searchsortedlast(xs, x)
    k = clamp(k, 1, n - 1)
    w = (x - xs[k]) / (xs[k+1] - xs[k])
    return y[k] + w * (y[k+1] - y[k])
end

# -----------------------------------------------------------------------------
# ONE backward EGM step (the linearization building block).
#   State carried between dates is the pair (Vnext, Cnext), both na×J:
#     * Cnext gives the marginal continuation value via the envelope,
#       v_k'(ã) = (1+r) u'(C_k(ã))/P_k, used in the Euler equation;
#     * Vnext gives the level used by the logit migration shares.
#   Given current prices (r, w, P), returns (Vcurr, C, ap).
# -----------------------------------------------------------------------------
function backward_step(Vnext, Cnext, r, w, P, χ, p::SParams)
    na, J, β, γ, θ = p.na, p.J, p.β, p.γ, p.θ
    ag = agrid(p)
    aendog = zeros(na, J); Cstar = zeros(na, J)
    ap = zeros(na, J); C = zeros(na, J); Vcurr = zeros(na, J)

    mc = Cnext .^ (-γ)                                   # u'(C_k(ã)) on the grid

    @inbounds for j in 1:J
        for m in 1:na
            maxv = -Inf
            for k in 1:J
                maxv = max(maxv, θ * Vnext[m, k])
            end
            denom = 0.0
            for k in 1:J
                denom += χ[j, k] * exp(θ * Vnext[m, k] - maxv)
            end
            Mcont = 0.0                                  # β Σ_k μ_jk v_k'(ã)
            for k in 1:J
                μjk = χ[j, k] * exp(θ * Vnext[m, k] - maxv) / denom
                Mcont += μjk * (1 + r) * mc[m, k] / P[k]
            end
            Mcont *= β
            Cstar[m, j]  = uprime_inv(P[j] * Mcont, γ)
            aendog[m, j] = (ag[m] + P[j] * Cstar[m, j] - w[j]) / (1 + r)
        end
    end

    @inbounds for j in 1:J
        ae = @view aendog[:, j]
        for ia in 1:na
            a = ag[ia]
            if a <= ae[1]
                apol = ag[1]
            else
                apol = interp1(ae, ag, a)
            end
            # feasibility guards: keep a' on grid and consumption strictly positive
            coh  = (1 + r) * a + w[j]                    # cash on hand (gross)
            apol = clamp(apol, ag[1], min(ag[end], coh - 1e-8 * P[j]))
            ap[ia, j] = apol
            C[ia, j]  = max((coh - apol) / P[j], 1e-10)
            maxv = -Inf
            for k in 1:J
                vk = interp1(ag, @view(Vnext[:, k]), apol)
                maxv = max(maxv, θ * vk)
            end
            s = 0.0
            for k in 1:J
                vk = interp1(ag, @view(Vnext[:, k]), apol)
                s += χ[j, k] * exp(θ * vk - maxv)
            end
            Vcurr[ia, j] = util(C[ia, j], γ) + (β / θ) * (maxv + log(s))
        end
    end
    return (; Vcurr, C, ap)
end

# migration probabilities on the fixed grid given value V and saving policy ap
function migration_probs(V, ap, χ, p::SParams)
    na, J, θ = p.na, p.J, p.θ
    ag = agrid(p)
    μ = zeros(na, J, J)
    @inbounds for j in 1:J, ia in 1:na
        apol = ap[ia, j]
        maxv = -Inf
        for k in 1:J
            vk = interp1(ag, @view(V[:, k]), apol)
            maxv = max(maxv, θ * vk)
        end
        denom = 0.0
        for k in 1:J
            vk = interp1(ag, @view(V[:, k]), apol)
            denom += χ[j, k] * exp(θ * vk - maxv)
        end
        for k in 1:J
            vk = interp1(ag, @view(V[:, k]), apol)
            μ[ia, j, k] = χ[j, k] * exp(θ * vk - maxv) / denom
        end
    end
    return μ
end

# -----------------------------------------------------------------------------
# Household problem: iterate the backward step to convergence.
# -----------------------------------------------------------------------------
function solve_household(r, w, P, χ, p::SParams; tol = 1e-10, maxit = 5_000)
    na, J, β, γ = p.na, p.J, p.β, p.γ
    ag = agrid(p)
    V = zeros(na, J); C = zeros(na, J)
    @inbounds for j in 1:J, ia in 1:na
        C[ia, j] = max(1e-4, r * ag[ia] + w[j])
        V[ia, j] = util(C[ia, j], γ) / (1 - β)
    end
    local ap
    for it in 1:maxit
        bw = backward_step(V, C, r, w, P, χ, p)
        err = max(maximum(abs.(bw.Vcurr .- V)), maximum(abs.(bw.C .- C)))
        V = bw.Vcurr; C = bw.C; ap = bw.ap
        err < tol && break
    end
    μ = migration_probs(V, ap, χ, p)
    return (; V, C, ap, μ)
end

# -----------------------------------------------------------------------------
# One-period transition operator T (row-stochastic, n×n).
#   Households at (a,j) save ap[ia,j] (split by a lottery onto the asset grid)
#   and migrate to k with probability μ[ia,j,k].  g_{t+1} = T' g_t.
# -----------------------------------------------------------------------------
function build_T(ap, μ, p::SParams)
    na, J = p.na, p.J
    n = na * J
    ag = agrid(p)
    T = zeros(n, n)
    @inbounds for j in 1:J, ia in 1:na
        i = idx(ia, j, p)
        # asset lottery
        apol = ap[ia, j]
        ka = searchsortedlast(ag, apol)
        ka = clamp(ka, 1, na - 1)
        wa = (apol - ag[ka]) / (ag[ka+1] - ag[ka])
        wa = clamp(wa, 0.0, 1.0)
        for k in 1:J
            pr = μ[ia, j, k]
            T[i, idx(ka,   k, p)] += pr * (1 - wa)
            T[i, idx(ka+1, k, p)] += pr * wa
        end
    end
    return T
end

# stationary distribution of column-stochastic P=T' (single linear solve)
function stationary(T)
    n = size(T, 1)
    Pc = Matrix(T')                  # column-stochastic
    A = Pc - I
    A[end, :] .= 1.0
    b = zeros(n); b[end] = 1.0
    g = A \ b
    g ./= sum(g)
    return g
end

# -----------------------------------------------------------------------------
# Aggregates from the stationary distribution g (over flattened (a,j)).
# -----------------------------------------------------------------------------
function aggregates(g, C, P, p::SParams)
    na, J = p.na, p.J
    ag = agrid(p)
    L  = zeros(J)                    # populations
    E  = zeros(J)                    # nominal expenditure P_j Y_j
    B  = 0.0                         # aggregate bond holdings
    @inbounds for j in 1:J, ia in 1:na
        gij = g[idx(ia, j, p)]
        L[j] += gij
        E[j] += gij * P[j] * C[ia, j]
        B    += gij * ag[ia]
    end
    return (; L, E, B)
end

# -----------------------------------------------------------------------------
# Steady state: nested fixed point.
#   outer: wages w and populations L (w[1] = 1 numeraire)
#   inner: bond rate r clears the integrated asset market (B = 0)
# -----------------------------------------------------------------------------
struct Steady
    p::SParams
    A::Matrix{Float64}
    χ::Matrix{Float64}
    r::Float64
    w::Vector{Float64}
    L::Vector{Float64}
    P::Vector{Float64}
    π::Matrix{Float64}
    V::Matrix{Float64}
    C::Matrix{Float64}
    ap::Matrix{Float64}
    μ::Array{Float64,3}
    g::Vector{Float64}
    T::Matrix{Float64}
end

# bond-market excess demand at rate r, given w, L, P (returns B and a solved hh)
function _bond_excess(r, w, P, χ, p::SParams)
    hh = solve_household(r, w, P, χ, p)
    T  = build_T(hh.ap, hh.μ, p)
    g  = stationary(T)
    agg = aggregates(g, hh.C, P, p)
    return agg.B, hh, T, g, agg
end

function solve_steady(p::SParams = SParams(); A = nothing, χ = nothing,
                      verbose = true, outer_tol = 1e-7, maxout = 200,
                      ξw = 0.3, ξL = 0.5)
    if A === nothing || χ === nothing
        A, χ = make_fundamentals(p)
    end
    J = p.J
    w = ones(J)
    L = fill(1.0 / J, J)

    local r, P, π, hh, T, g, agg
    for outit in 1:maxout
        P = price_index(w, L, A, p)

        # --- inner: bisection on r to clear the bond market (β(1+r) < 1) ---
        rhi = 1 / p.β - 1 - 1e-4
        rlo = -0.05
        Blo, _, _, _, _ = _bond_excess(rlo, w, P, χ, p)
        Bhi, _, _, _, _ = _bond_excess(rhi, w, P, χ, p)
        # B is increasing in r; ensure a bracket
        r = 0.5 * (rlo + rhi)
        for _ in 1:60
            B, hh, T, g, agg = _bond_excess(r, w, P, χ, p)
            if abs(B) < 1e-9
                break
            end
            B > 0 ? (rhi = r) : (rlo = r)
            r = 0.5 * (rlo + rhi)
        end
        B, hh, T, g, agg = _bond_excess(r, w, P, χ, p)

        # --- outer: update wages and populations ---
        # Labor-market clearing is  w_i L_i = Σ_j π_ij E_j  with
        # π_ij ∝ w_i^{1-σ}, so the contraction update divides out w_i^{1-σ}
        # (ODSP eq. F.21):  w_i^new = [ (Σ_j π_ij E_j / w_i^{1-σ}) / L_i ]^{1/σ}.
        π = trade_shares(w, L, A, p)
        wnew = similar(w); Lnew = similar(L)
        for i in 1:J
            inc = 0.0
            for j in 1:J
                inc += π[i, j] * agg.E[j]
            end
            inc /= w[i]^(1 - p.σ)                # remove the w_i^{1-σ} loading
            wnew[i] = (inc / max(agg.L[i], 1e-10))^(1 / p.σ)
        end
        Lnew .= agg.L
        wnew ./= wnew[1]                       # numeraire
        Lnew ./= sum(Lnew)

        err = max(maximum(abs.(wnew .- w)), maximum(abs.(Lnew .- L)))
        w .= ξw .* wnew .+ (1 - ξw) .* w
        L .= ξL .* Lnew .+ (1 - ξL) .* L
        w ./= w[1]
        if verbose && (outit % 10 == 0 || outit == 1)
            @printf("  outer %3d  ‖Δ(w,L)‖=%.3e  r=%.5f  B=%.2e\n", outit, err, r, B)
        end
        if err < outer_tol
            P = price_index(w, L, A, p)
            B, hh, T, g, agg = _bond_excess(r, w, P, χ, p)
            π = trade_shares(w, L, A, p)
            verbose && @printf("  converged after %d outer iters (‖Δ‖=%.2e)\n", outit, err)
            return Steady(p, A, χ, r, copy(w), copy(L), P, π,
                          hh.V, hh.C, hh.ap, hh.μ, g, T)
        end
    end
    error("steady state did not converge")
end

end # module
