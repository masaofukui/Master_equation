# =============================================================================
#  FAME (First-order Approximation to the Master Equation) for the multi-region
#  Huggett economy with migration, savings, and trade.
#
#  The aggregate state is the distribution g over (assets a, region j).  Given g,
#  the WITHIN-PERIOD equilibrium reduces to J prices — the bond rate r and the
#  wages w_2,…,w_J (w_1 is the numeraire) — that clear
#       * the integrated bond market:  Σ_{a,j} a'(a,j) g(a,j) = 0,
#       * the labor markets (regions 2..J): w_i L_i = Σ_j π_ij E_j,
#  while L_j = Σ_a g(a,j) and P_j = price_index(w,L,A) follow from (g, w).
#
#  FAME linearizes this map around the steady state.  Its central object is the
#  price-impact operator  Q = dp/dg  (J×n), obtained from the static equilibrium
#  residual F(p, g) = 0 by the implicit function theorem,
#       Q = -(∂F/∂p)^{-1} (∂F/∂g),
#  here computed with the household continuation held at the steady state (the
#  "temporary equilibrium" used to validate Q).  The induced linear law of motion
#  of the distribution is  DΦ = T' + G,  with the general-equilibrium kernel
#       G = (M_a + M_μ) · da_dg,
#  where da_dg is the policy response to a marginal mass and M_a, M_μ are the
#  push-forward derivatives through the saving lottery and the migration shares.
#
#  We (i) validate Q against an independent nonlinear temporary-equilibrium solve
#  and (ii) trace the interest-rate / population impulse response to a one-time
#  wealth-redistribution shock.  Cross-checks with the SSJ code are in
#  spatial_ssj.jl (same steady state and linearization).
#
#  Run:  julia spatial_fame.jl
# =============================================================================

if !isdefined(Main, :SpatialHuggett)
    include("SpatialHuggett.jl")
end
using .SpatialHuggett
using LinearAlgebra, Printf, Random, Plots

const SH = SpatialHuggett
Ashock(A, m, z) = (B = copy(A); B[m, :] .*= (1 + z); B)

# -----------------------------------------------------------------------------
# Static partials of the trade/price block w.r.t. wages and populations.
# -----------------------------------------------------------------------------
function static_partials(ss, p; h = 1e-6)
    J = p.J; A = ss.A; w = ss.w; L = ss.L
    dPdw = zeros(J, J); dPdL = zeros(J, J)
    dπdw = zeros(J, J, J); dπdL = zeros(J, J, J)
    for k in 1:J
        wp = copy(w); wp[k] += h; wm = copy(w); wm[k] -= h
        dPdw[:, k]    = (price_index(wp, L, A, p) .- price_index(wm, L, A, p)) ./ (2h)
        dπdw[:, :, k] = (trade_shares(wp, L, A, p) .- trade_shares(wm, L, A, p)) ./ (2h)
        Lp = copy(L); Lp[k] += h; Lm = copy(L); Lm[k] -= h
        dPdL[:, k]    = (price_index(w, Lp, A, p) .- price_index(w, Lm, A, p)) ./ (2h)
        dπdL[:, :, k] = (trade_shares(w, Lp, A, p) .- trade_shares(w, Lm, A, p)) ./ (2h)
    end
    return (; dPdw, dPdL, dπdw, dπdL)
end

# -----------------------------------------------------------------------------
# Household contemporaneous sensitivities (continuation fixed at steady state):
# ∂(a', C)/∂(direct input) for inputs (r, w_1..w_J, P_1..P_J).  Returned as
# n×(1+2J) matrices Sap, SC (flattened states), plus ∂μ/∂a' per state (n×J).
# -----------------------------------------------------------------------------
function household_sensitivities(ss, p; hr = 1e-6)
    na, J = p.na, p.J; n = SH.n_states(p)
    Vss, Css = ss.V, ss.C
    bw0 = SH.backward_step(Vss, Css, ss.r, ss.w, ss.P, ss.χ, p)
    ap0 = bw0.ap

    ninp = 1 + 2J
    Sap = zeros(n, ninp); SC = zeros(n, ninp); SV = zeros(n, ninp)
    function fill_col!(col, dr, dw, dP)
        bp = SH.backward_step(Vss, Css, ss.r + dr, ss.w .+ dw, ss.P .+ dP, ss.χ, p)
        bm = SH.backward_step(Vss, Css, ss.r - dr, ss.w .- dw, ss.P .- dP, ss.χ, p)
        Sap[:, col] = vec((bp.ap    .- bm.ap)    ./ (2hr))
        SC[:, col]  = vec((bp.C     .- bm.C )    ./ (2hr))
        SV[:, col]  = vec((bp.Vcurr .- bm.Vcurr) ./ (2hr))
    end
    fill_col!(1, hr, zeros(J), zeros(J))                       # r
    for k in 1:J
        dw = zeros(J); dw[k] = hr; fill_col!(1 + k, 0.0, dw, zeros(J))      # w_k
        dP = zeros(J); dP[k] = hr; fill_col!(1 + J + k, 0.0, zeros(J), dP)  # P_k
    end

    # ∂μ_{i,k}/∂a'_i : migration shares depend on the chosen a' (continuation fixed)
    dμda = zeros(n, J)
    ag = agrid(p); hap = 1e-6
    for j in 1:J, ia in 1:na
        i = idx(ia, j, p)
        ap_p = copy(ap0); ap_p[ia, j] += hap
        ap_m = copy(ap0); ap_m[ia, j] -= hap
        μp = SH.migration_probs(Vss, ap_p, ss.χ, p)
        μm = SH.migration_probs(Vss, ap_m, ss.χ, p)
        for k in 1:J
            dμda[i, k] = (μp[ia, j, k] - μm[ia, j, k]) / (2hap)
        end
    end
    return (; Sap, SC, SV, dμda, ap0)
end

# -----------------------------------------------------------------------------
# Continuation Jacobians of the backward step (the anticipation channel).
# Perturb the continuation value Vnext and policy Cnext state by state and read
# off the response of (a', C, Vcurr) and the migration shares μ.
#   JaV = ∂a'/∂Vnext, JCV = ∂C/∂Vnext, JVV = ∂Vcurr/∂Vnext   (n×n)
#   JaC = ∂a'/∂Cnext, JCC = ∂C/∂Cnext, JVC = ∂Vcurr/∂Cnext   (n×n)
#   JmuV[k] = ∂μ_{·,k}/∂Vnext                                  (n×n, per region k)
# All flattenings use vec (column-major), consistent with idx(ia,j)=(j-1)na+ia.
# -----------------------------------------------------------------------------
function continuation_jacobians(ss, p; δ = 1e-6)
    na, J = p.na, p.J; n = SH.n_states(p)
    Vss, Css = ss.V, ss.C
    JaV = zeros(n, n); JCV = zeros(n, n); JVV = zeros(n, n)
    JaC = zeros(n, n); JCC = zeros(n, n); JVC = zeros(n, n)
    JmuV = [zeros(n, n) for _ in 1:J]
    for s in 1:n
        Vp = copy(Vss); Vp[s] += δ; Vm = copy(Vss); Vm[s] -= δ
        bp = SH.backward_step(Vp, Css, ss.r, ss.w, ss.P, ss.χ, p)
        bm = SH.backward_step(Vm, Css, ss.r, ss.w, ss.P, ss.χ, p)
        JaV[:, s] = vec((bp.ap    .- bm.ap)    ./ (2δ))
        JCV[:, s] = vec((bp.C     .- bm.C )    ./ (2δ))
        JVV[:, s] = vec((bp.Vcurr .- bm.Vcurr) ./ (2δ))
        μp = SH.migration_probs(Vp, bp.ap, ss.χ, p)
        μm = SH.migration_probs(Vm, bm.ap, ss.χ, p)
        for k in 1:J
            JmuV[k][:, s] = vec((μp[:, :, k] .- μm[:, :, k]) ./ (2δ))
        end
        Cp = copy(Css); Cp[s] += δ; Cm = copy(Css); Cm[s] -= δ
        bp2 = SH.backward_step(Vss, Cp, ss.r, ss.w, ss.P, ss.χ, p)
        bm2 = SH.backward_step(Vss, Cm, ss.r, ss.w, ss.P, ss.χ, p)
        JaC[:, s] = vec((bp2.ap    .- bm2.ap)    ./ (2δ))
        JCC[:, s] = vec((bp2.C     .- bm2.C )    ./ (2δ))
        JVC[:, s] = vec((bp2.Vcurr .- bm2.Vcurr) ./ (2δ))
    end
    return (; JaV, JCV, JVV, JaC, JCC, JVC, JmuV)
end

# input column indices in Sap/SC
ci_r() = 1
ci_w(k, J) = 1 + k
ci_P(k, J) = 1 + J + k

# reduced sensitivities of a flattened policy S (n×ninp) w.r.t. (r, w_k, L_k),
# composing the P-channel  dP_m/dw_k, dP_m/dL_k.
red_r(S) = S[:, ci_r()]
red_w(S, k, sp, J) = S[:, ci_w(k, J)] .+ sum(sp.dPdw[m, k] .* S[:, ci_P(m, J)] for m in 1:J)
red_L(S, k, sp, J) = sum(sp.dPdL[m, k] .* S[:, ci_P(m, J)] for m in 1:J)

# -----------------------------------------------------------------------------
# Price-impact operator Q = dp/dg  (J×n), p = (r, w_2,…,w_J).
# Assembles ∂F/∂p (J×J) and ∂F/∂g (J×n) of the static equilibrium residual and
# inverts.  Continuation is held at the steady state.
# -----------------------------------------------------------------------------
function price_impact(ss, p, sens, sp)
    na, J = p.na, p.J; n = SH.n_states(p); ag = agrid(p)
    g = ss.g; Pss = ss.P; πss = ss.π; wss = ss.w; Lss = ss.L
    Sap, SC = sens.Sap, sens.SC
    region(ξ) = (ξ - 1) ÷ na + 1
    asset(ξ)  = (ξ - 1) % na + 1

    Cagg = [sum(g[idx(ia, j, p)] * ss.C[ia, j] for ia in 1:na) for j in 1:J]
    Ess  = [Pss[j] * Cagg[j] for j in 1:J]

    # reduced policy responses
    rap_r = red_r(Sap);  rC_r = red_r(SC)
    rap_w = [red_w(Sap, k, sp, J) for k in 1:J]
    rap_L = [red_L(Sap, k, sp, J) for k in 1:J]
    rC_w  = [red_w(SC, k, sp, J) for k in 1:J]
    rC_L  = [red_L(SC, k, sp, J) for k in 1:J]

    # region-summed consumption responses  gCj(resp) = Σ_{a∈j} g(a,j) resp(a,j)
    gC(resp) = [sum(g[idx(ia, j, p)] * resp[idx(ia, j, p)] for ia in 1:na) for j in 1:J]
    gCr = gC(rC_r)
    gCw = [gC(rC_w[k]) for k in 1:J]
    gCL = [gC(rC_L[k]) for k in 1:J]

    # ---- ∂F/∂p  (J×J): rows = (bond, labor 2..J); cols = (r, w_2..w_J) ----
    App = zeros(J, J)
    # bond row
    App[1, 1] = dot(g, rap_r)
    for k in 2:J
        App[1, k] = dot(g, rap_w[k])
    end
    # labor rows i = 2..J  (row index = i)
    for i in 2:J
        # d/dr
        App[i, 1] = -sum(πss[i, j] * Pss[j] * gCr[j] for j in 1:J)
        # d/dw_k
        for k in 2:J
            val = (i == k ? Lss[i] : 0.0)
            for j in 1:J
                val -= sp.dπdw[i, j, k] * Ess[j]
                val -= πss[i, j] * sp.dPdw[j, k] * Cagg[j]
                val -= πss[i, j] * Pss[j] * gCw[k][j]
            end
            App[i, k] = val
        end
    end

    # ---- ∂F/∂g  (J×n) ----
    Apg = zeros(J, n)
    for ξ in 1:n
        rξ = region(ξ); aξ = asset(ξ)
        # bond row: a'_ξ + (direct L→P channel on everyone's saving)
        Apg[1, ξ] = ss.ap[aξ, rξ] + dot(g, rap_L[rξ])
        # labor rows
        for i in 2:J
            val = (rξ == i ? wss[i] : 0.0)
            val -= πss[i, rξ] * Pss[rξ] * ss.C[aξ, rξ]            # direct mass in region rξ
            for j in 1:J
                val -= sp.dπdL[i, j, rξ] * Ess[j]
                val -= πss[i, j] * sp.dPdL[j, rξ] * Cagg[j]
                val -= πss[i, j] * Pss[j] * gCL[rξ][j]
            end
            Apg[i, ξ] = val
        end
    end

    Q = -(App \ Apg)                  # J×n; row 1 = dr/dg, rows 2..J = dw/dg
    return (; Q, App, Apg, Cagg, Ess, rap_r, rap_w, rap_L)
end

# -----------------------------------------------------------------------------
# Independent nonlinear temporary equilibrium at a distribution `dist`
# (continuation fixed at steady state).  Nested fixed point: bisection on r for
# the bond market inside a wage/Pop update for the labor markets.  Used to
# validate the price-impact operator Q.
# -----------------------------------------------------------------------------
function temp_equilibrium(ss, p, dist; ξw = 0.5, tol = 1e-11, maxit = 400)
    na, J = p.na, p.J; ag = agrid(p)
    L = [sum(dist[idx(ia, j, p)] for ia in 1:na) for j in 1:J]
    w = copy(ss.w)
    local r, P, π
    for it in 1:maxit
        P = price_index(w, L, ss.A, p)
        # bond market: bisection on r
        rlo, rhi = -0.04, 1/p.β - 1 - 1e-4
        bond(rr) = begin
            bw = SH.backward_step(ss.V, ss.C, rr, w, P, ss.χ, p)
            sum(dist[idx(ia, j, p)] * bw.ap[ia, j] for j in 1:J, ia in 1:na)
        end
        r = 0.5 * (rlo + rhi)
        for _ in 1:80
            b = bond(r); abs(b) < 1e-12 && break
            b > 0 ? (rhi = r) : (rlo = r); r = 0.5 * (rlo + rhi)
        end
        bw = SH.backward_step(ss.V, ss.C, r, w, P, ss.χ, p)
        E = [sum(dist[idx(ia, j, p)] * P[j] * bw.C[ia, j] for ia in 1:na) for j in 1:J]
        π = trade_shares(w, L, ss.A, p)
        wnew = copy(w)
        for i in 2:J
            inc = sum(π[i, j] * E[j] for j in 1:J) / w[i]^(1 - p.σ)
            wnew[i] = (inc / max(L[i], 1e-10))^(1 / p.σ)
        end
        err = maximum(abs.(wnew .- w))
        w .= ξw .* wnew .+ (1 - ξw) .* w
        err < tol && break
    end
    return vcat(r, w[2:J])           # p = (r, w_2..w_J)
end

# -----------------------------------------------------------------------------
# General-equilibrium kernel G and law of motion DΦ = T' + G.
# -----------------------------------------------------------------------------
function law_of_motion(ss, p, sens, pim, sp)
    na, J = p.na, p.J; n = SH.n_states(p); ag = agrid(p)
    g = ss.g; Q = pim.Q
    region(ξ) = (ξ - 1) ÷ na + 1

    # total policy response to a marginal mass: price channel + direct L→P channel
    # da_dg[i, ξ] = Σ_k rap_price[i,k] Q[k,ξ] + rap_L[reg ξ][i]
    rap_price = hcat(pim.rap_r, [pim.rap_w[k] for k in 2:J]...)    # n × J  (cols: r,w_2..w_J)
    da_dg = rap_price * Q                                          # n × n  (price channel)
    for ξ in 1:n
        @views da_dg[:, ξ] .+= pim.rap_L[region(ξ)]               # direct L channel
    end

    # push-forward derivatives.  For state i with mass g_i, saving lands in the
    # bracket (k,k+1) with weight w on the asset grid; migration share μ_{i,kk}.
    μss = ss.μ
    Ma   = zeros(n, n)     # d g'/d a'_i   weighted by migration
    Mμap = zeros(n, n)     # d g'/d a'_i   through the migration-share response ∂μ/∂a'
    for j in 1:J, ia in 1:na
        i = idx(ia, j, p)
        ka = searchsortedlast(ag, ss.ap[ia, j]); ka = clamp(ka, 1, na - 1)
        Δ  = ag[ka+1] - ag[ka]
        ww = clamp((ss.ap[ia, j] - ag[ka]) / Δ, 0.0, 1.0)
        gi = g[i]
        for kk in 1:J
            # saving-lottery channel (μ fixed)
            Ma[idx(ka,   kk, p), i] -= gi * μss[ia, j, kk] / Δ
            Ma[idx(ka+1, kk, p), i] += gi * μss[ia, j, kk] / Δ
            # migration-share channel: dμ_{i,kk} = (∂μ/∂a') da'_i, mass placed by lottery
            c = gi * sens.dμda[i, kk]
            Mμap[idx(ka,   kk, p), i] += c * (1 - ww)
            Mμap[idx(ka+1, kk, p), i] += c * ww
        end
    end
    Mtot = Ma .+ Mμap
    G  = Mtot * da_dg
    DΦ = Matrix(ss.T') .+ G
    return (; DΦ, G, da_dg, Mtot)
end

# -----------------------------------------------------------------------------
# FAME impulse response to an aggregate productivity shock in region `mshock`.
# Adds the shock's DIRECT loadings onto prices (Q_Z = dp/dZ) and onto the
# distribution transition (Φ_Z = dg'/dZ), then iterates the state-space recursion
#       dp_t = Q dg_t + Q_Z dZ_t,     dg_{t+1} = DΦ dg_t + Φ_Z dZ_t,
# with dg_1 = 0 (assets/population predetermined at impact) and dZ an AR(1).
# -----------------------------------------------------------------------------
function fame_productivity_irf(ss, p, sens, sp, pim, lom; mshock = 1,
                               z0 = -0.01, ρz = 0.8, T = 150, h = 1e-6)
    na, J = p.na, p.J; n = SH.n_states(p)
    g = ss.g; Q = pim.Q

    # static partials of P, π w.r.t. the shock Z (scales A[mshock,:])
    dPdZ = (price_index(ss.w, ss.L, Ashock(ss.A, mshock, h), p) .-
            price_index(ss.w, ss.L, Ashock(ss.A, mshock, -h), p)) ./ (2h)
    dπdZ = (trade_shares(ss.w, ss.L, Ashock(ss.A, mshock, h), p) .-
            trade_shares(ss.w, ss.L, Ashock(ss.A, mshock, -h), p)) ./ (2h)

    Sap, SC = sens.Sap, sens.SC
    # direct effect of Z on saving/consumption through the price index P
    S_ap_dPZ = sum(dPdZ[m] .* Sap[:, ci_P(m, J)] for m in 1:J)
    S_C_dPZ  = sum(dPdZ[m] .* SC[:,  ci_P(m, J)] for m in 1:J)

    # ∂F/∂Z (J-vector): bond market + labor markets (regions 2..J)
    F_Z = zeros(J)
    F_Z[1] = dot(g, S_ap_dPZ)
    for i in 2:J
        val = 0.0
        for j in 1:J
            gCj = sum(g[idx(ia, j, p)] * S_C_dPZ[idx(ia, j, p)] for ia in 1:na)
            val -= dπdZ[i, j] * pim.Ess[j]
            val -= ss.π[i, j] * dPdZ[j] * pim.Cagg[j]
            val -= ss.π[i, j] * ss.P[j] * gCj
        end
        F_Z[i] = val
    end
    Q_Z = -(pim.App \ F_Z)                                   # J-vector dp/dZ

    # direct effect of the shock on the transition (through the policy response)
    rap_price = hcat(pim.rap_r, [pim.rap_w[k] for k in 2:J]...)
    dap_Z = S_ap_dPZ .+ rap_price * Q_Z                      # total saving response
    Φ_Z   = lom.Mtot * dap_Z                                 # n-vector dg'/dZ

    dZ = [z0 * ρz^(t-1) for t in 1:T]
    dr = zeros(T); dw = [zeros(T) for _ in 1:J]; dL = [zeros(T) for _ in 1:J]
    dg = zeros(n)
    for t in 1:T
        dp = Q * dg .+ Q_Z .* dZ[t]                          # (r, w_2..w_J)
        dr[t] = dp[1]
        for k in 2:J
            dw[k][t] = dp[k]
        end
        for j in 1:J
            dL[j][t] = sum(dg[idx(ia, j, p)] for ia in 1:na)
        end
        dg = lom.DΦ * dg .+ Φ_Z .* dZ[t]
    end
    return (; dr, dw, dL, dZ, Q_Z, Φ_Z)
end

# -----------------------------------------------------------------------------
# Wealth-redistribution shock: within each region nudge assets a fraction κ
# toward the region's mean (spread back with the model lottery).  Conserves
# population in every region (so L is unchanged) and perturbs only the asset
# margin → moves the bond market.
# -----------------------------------------------------------------------------
function redistribution_shock(ss, p; κ = 1e-2)
    na, J = p.na, p.J; ag = agrid(p); g = ss.g
    h = zeros(SH.n_states(p))
    for j in 1:J
        mass = sum(g[idx(ia, j, p)] for ia in 1:na)
        mass < 1e-14 && continue
        abar = sum(g[idx(ia, j, p)] * ag[ia] for ia in 1:na) / mass
        for ia in 1:na
            i = idx(ia, j, p)
            atgt = clamp(ag[ia] + κ * (abar - ag[ia]), ag[1], ag[end])
            k = searchsortedlast(ag, atgt); k = clamp(k, 1, na - 1)
            ww = clamp((atgt - ag[k]) / (ag[k+1] - ag[k]), 0.0, 1.0)
            h[idx(k,   j, p)] += g[i] * (1 - ww)
            h[idx(k+1, j, p)] += g[i] * ww
            h[i] -= g[i]
        end
    end
    return h
end

# -----------------------------------------------------------------------------
# FULL FAME with the anticipation (continuation) channel.
#
# Households' saving and (especially) migration choices respond to EXPECTED
# future values, which move with the aggregate state.  Writing the value/policy
# functions as v·dg + v_Z·dZ etc., the master equation linearizes to a coupled
# fixed point for the impulse value v=dV/dg, the impulse policy cg=dC/dg, the
# price-impact Q=dp/dg and the law of motion DΦ=T'+G, plus the analogous shock
# loadings (v_Z, c_Z, Q_Z, Φ_Z).  The interest-rate impulse response then solves
#       dp_t = Q dg_t + Q_Z dZ_t,   dg_{t+1} = DΦ dg_t + Φ_Z dZ_t,  dg_1 = 0.
# -----------------------------------------------------------------------------
function fame_full(ss, p, sens, cj, pim, sp; mshock = 1, ρz = 0.8,
                   tol = 1e-9, maxit = 4000, ξ = 1.0)
    na, J = p.na, p.J; n = SH.n_states(p); ag = agrid(p)
    g = ss.g; β = p.β
    Sap, SC, SV, dμda = sens.Sap, sens.SC, sens.SV, sens.dμda
    JaV, JCV, JVV = cj.JaV, cj.JCV, cj.JVV
    JaC, JCC, JVC = cj.JaC, cj.JCC, cj.JVC
    JmuV = cj.JmuV
    region(ξidx) = (ξidx - 1) ÷ na + 1

    # ---- input maps: dInput = Dp·dp + DL·dg + DZ·dZ  (input order r,w_1..w_J,P_1..P_J)
    ninp = 1 + 2J
    Dp = zeros(ninp, J)               # dp = (dr, dw_2,…,dw_J)
    Dp[ci_r(), 1] = 1.0
    for k in 2:J; Dp[ci_w(k, J), k] = 1.0; end
    for m in 1:J, k in 2:J; Dp[ci_P(m, J), k] = sp.dPdw[m, k]; end
    DL = zeros(ninp, n)
    for m in 1:J, ξidx in 1:n; DL[ci_P(m, J), ξidx] = sp.dPdL[m, region(ξidx)]; end

    # shock partials of P, π
    hZ = 1e-6
    dPdZ = (price_index(ss.w, ss.L, Ashock(ss.A, mshock, hZ), p) .-
            price_index(ss.w, ss.L, Ashock(ss.A, mshock, -hZ), p)) ./ (2hZ)
    dπdZ = (trade_shares(ss.w, ss.L, Ashock(ss.A, mshock, hZ), p) .-
            trade_shares(ss.w, ss.L, Ashock(ss.A, mshock, -hZ), p)) ./ (2hZ)
    DZ = zeros(ninp); for m in 1:J; DZ[ci_P(m, J)] = dPdZ[m]; end

    # ---- push-forward pieces: Ma (saving, μ fixed) and P_asset (asset lottery) ----
    Ma = zeros(n, n); Pas = zeros(na, n)
    for j in 1:J, ia in 1:na
        i  = idx(ia, j, p)
        ka = clamp(searchsortedlast(ag, ss.ap[ia, j]), 1, na - 1)
        Δ  = ag[ka+1] - ag[ka]
        ww = clamp((ss.ap[ia, j] - ag[ka]) / Δ, 0.0, 1.0)
        gi = g[i]
        Pas[ka,   i] += gi * (1 - ww)
        Pas[ka+1, i] += gi * ww
        for kk in 1:J
            Ma[idx(ka,   kk, p), i] -= gi * ss.μ[ia, j, kk] / Δ
            Ma[idx(ka+1, kk, p), i] += gi * ss.μ[ia, j, kk] / Δ
        end
    end
    # region-masked steady-state distribution (for g-weighted region sums)
    gj = [[(region(i) == j ? g[i] : 0.0) for i in 1:n] for j in 1:J]

    Tp = Matrix(ss.T')
    πss, Pss = ss.π, ss.P

    # assemble G = Ma·ag + Σ_k place(P_asset·μ_g[k]) into region-k asset rows
    function build_G(ag, μg)
        G = Ma * ag
        for r in 1:J
            blk = Pas * μg[r]                              # na × n
            @inbounds for a in 1:na
                G[idx(a, r, p), :] .+= @view blk[a, :]
            end
        end
        return G
    end

    # ===================== dg-block fixed point =====================
    v = zeros(n, n); cg = zeros(n, n); DΦ = copy(Tp); Q = copy(pim.Q)
    local err
    for it in 1:maxit
        CV = v * DΦ; CC = cg * DΦ
        SaveCont = JaV * CV .+ JaC * CC
        ConsCont = JCV * CV .+ JCC * CC
        ValCont  = JVV * CV .+ JVC * CC
        # continuation correction to market clearing
        Acont = zeros(J, n)
        Acont[1, :] = g' * SaveCont
        for i in 2:J, j in 1:J
            Acont[i, :] .-= (πss[i, j] * Pss[j]) .* (gj[j]' * ConsCont)[:]
        end
        Qn  = -(pim.App \ (pim.Apg .+ Acont))
        Min = Dp * Qn .+ DL
        ag  = Sap * Min .+ SaveCont
        cgn = SC  * Min .+ ConsCont
        vn  = SV  * Min .+ ValCont
        μg  = [JmuV[r] * CV .+ dμda[:, r] .* ag for r in 1:J]
        DΦn = Tp .+ build_G(ag, μg)
        err = max(maximum(abs.(vn .- v)), maximum(abs.(DΦn .- DΦ)))
        v .= ξ .* vn .+ (1-ξ) .* v
        cg .= ξ .* cgn .+ (1-ξ) .* cg
        DΦ .= ξ .* DΦn .+ (1-ξ) .* DΦ
        Q  = Qn
        err < tol && break
    end

    # ===================== Z-block fixed point =====================
    # contemporaneous F_Z (continuation fixed)
    S_ap_dPZ = Sap * DZ; S_C_dPZ = SC * DZ
    Cagg, Ess = pim.Cagg, pim.Ess
    F_Z = zeros(J)
    F_Z[1] = dot(g, S_ap_dPZ)
    for i in 2:J, j in 1:J
        gCj = sum(g[idx(ia, j, p)] * S_C_dPZ[idx(ia, j, p)] for ia in 1:na)
        F_Z[i] += -dπdZ[i, j]*Ess[j] - πss[i,j]*dPdZ[j]*Cagg[j] - πss[i,j]*Pss[j]*gCj
    end

    ΦZ = zeros(n); vZ = zeros(n); cZ = zeros(n); QZ = -(pim.App \ F_Z)
    for it in 1:maxit
        KV = v * ΦZ .+ ρz .* vZ
        KC = cg * ΦZ .+ ρz .* cZ
        SaveContZ = JaV * KV .+ JaC * KC
        ConsContZ = JCV * KV .+ JCC * KC
        ValContZ  = JVV * KV .+ JVC * KC
        AcontZ = zeros(J)
        AcontZ[1] = dot(g, SaveContZ)
        for i in 2:J, j in 1:J
            AcontZ[i] -= πss[i, j] * Pss[j] * dot(gj[j], ConsContZ)
        end
        QZn  = -(pim.App \ (F_Z .+ AcontZ))
        MinZ = Dp * QZn .+ DZ
        aZ   = Sap * MinZ .+ SaveContZ
        cZn  = SC  * MinZ .+ ConsContZ
        vZn  = SV  * MinZ .+ ValContZ
        μZ   = [JmuV[r] * KV .+ dμda[:, r] .* aZ for r in 1:J]
        ΦZn  = Ma * aZ
        for r in 1:J
            blk = Pas * μZ[r]
            @inbounds for a in 1:na; ΦZn[idx(a, r, p)] += blk[a]; end
        end
        err = max(maximum(abs.(vZn .- vZ)), maximum(abs.(ΦZn .- ΦZ)))
        vZ .= vZn; cZ .= cZn; ΦZ .= ΦZn; QZ = QZn
        err < tol && break
    end

    return (; v, cg, Q, DΦ, vZ, cZ, QZ, ΦZ)
end

# trace the FAME IRF (full) to an AR(1) productivity shock
function fame_full_irf(ss, p, ff; z0 = -0.01, ρz = 0.8, T = 150)
    na, J = p.na, p.J; n = SH.n_states(p)
    dZ = [z0 * ρz^(t-1) for t in 1:T]
    dr = zeros(T); dw = [zeros(T) for _ in 1:J]; dL = [zeros(T) for _ in 1:J]
    dg = zeros(n)
    for t in 1:T
        dp = ff.Q * dg .+ ff.QZ .* dZ[t]
        dr[t] = dp[1]
        for k in 2:J; dw[k][t] = dp[k]; end
        for j in 1:J; dL[j][t] = sum(dg[idx(ia, j, p)] for ia in 1:na); end
        dg = ff.DΦ * dg .+ ff.ΦZ .* dZ[t]
    end
    return (; dr, dw, dL, dZ)
end

# =============================================================================
# Driver
# =============================================================================
function main()
    p = SParams()
    println("="^70); println(" Spatial Huggett: steady state"); println("="^70)
    ss = solve_steady(p; verbose = false)
    @printf("  r=%.5f  w=%s  L=%s\n", ss.r,
            string(round.(ss.w, digits=3)), string(round.(ss.L, digits=3)))

    sp   = static_partials(ss, p)
    sens = household_sensitivities(ss, p)
    pim  = price_impact(ss, p, sens, sp)

    println("\n", "="^70)
    println(" Validate price impact  Q=dp/dg  vs nonlinear temporary equilibrium")
    println("="^70)
    Random.seed!(1)
    n = SH.n_states(p)
    # random mass-preserving direction supported on interior (unconstrained) states
    constrained = vec(ss.ap) .<= (agrid(p)[1] + 1e-8)
    h = randn(n); h[constrained] .= 0.0
    # keep populations fixed (mass preserving within each region) so L is unchanged
    for j in 1:p.J
        rng = ((j-1)*p.na+1):(j*p.na)
        h[rng] .-= sum(h[rng]) / p.na
    end
    ε = 1e-6
    p_p = temp_equilibrium(ss, p, ss.g .+ ε .* h)
    p_m = temp_equilibrium(ss, p, ss.g .- ε .* h)
    dp_num = (p_p .- p_m) ./ (2ε)
    dp_an  = pim.Q * h
    @printf("  %-6s %14s %14s %12s\n", "price", "finite-diff", "FAME Q·h", "rel.err")
    names = vcat("r", ["w$k" for k in 2:p.J])
    for k in 1:p.J
        @printf("  %-6s %+14.6e %+14.6e %12.2e\n", names[k], dp_num[k], dp_an[k],
                abs(dp_num[k]-dp_an[k])/max(abs(dp_num[k]),1e-12))
    end

    println("\n", "="^70)
    println(" FAME law of motion and interest-rate impulse response")
    println("="^70)
    lom = law_of_motion(ss, p, sens, pim, sp)
    @printf("  max|colsum G| = %.2e  (mass conservation)\n",
            maximum(abs.(vec(sum(lom.G, dims=1)))))
    @printf("  spectral radius of DΦ = %.4f\n", maximum(abs.(eigvals(lom.DΦ))))

    # trace the IRF to a one-time wealth redistribution
    Tshow = 40
    h0 = redistribution_shock(ss, p; κ = 1e-2)
    dr = zeros(Tshow); dL = zeros(p.J, Tshow)
    hh = copy(h0)
    for t in 1:Tshow
        dr[t] = (pim.Q*hh)[1]
        for j in 1:p.J
            dL[j, t] = sum(hh[idx(ia, j, p)] for ia in 1:p.na)
        end
        hh = lom.DΦ * hh
    end
    @printf("  %-4s %12s\n", "t", "dr (bp)")
    for t in 1:8
        @printf("  %-4d %12.4f\n", t, 1e4*dr[t])
    end

    p1 = plot(1:Tshow, 1e4 .* dr; lw = 2, marker = :circle, ms = 2, legend = false,
              xlabel = "period t", ylabel = "Δr (bp)",
              title = "FAME: bond-rate response to wealth redistribution")
    p2 = plot(; xlabel = "period t", ylabel = "Δ population", title = "Population")
    for j in 1:p.J
        plot!(p2, 1:Tshow, dL[j, :]; lw = 2, label = "region $j")
    end
    plt = plot(p1, p2; layout = (1, 2), size = (1000, 380))
    outpng = joinpath(@__DIR__, "..", "figure", "spatial_fame_irf.png")
    savefig(plt, outpng)
    @printf("\n  saved IRF plot to %s\n", normpath(outpng))
    return ss, pim, lom
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
