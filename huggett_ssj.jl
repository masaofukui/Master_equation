# =============================================================================
#  Sequence-Space Jacobian (SSJ) solution of the discrete-time Huggett economy.
#
#  Companion to huggett_fame.jl.  This file recomputes the SAME impulse response
#  as the master-equation (FAME) code, but with the method of
#
#      Auclert, Bardóczy, Rognlie & Straub (2021),
#      "Using the Sequence-Space Jacobian to Solve and Estimate
#       Heterogeneous-Agent Models", Econometrica.   (literature/sequence_space_jacobian.pdf)
#
#  --------------------------------------------------------------------------
#  Mapping of the paper's objects to this Huggett model
#  --------------------------------------------------------------------------
#  * Aggregate input  : the bond price q_t          (1/q = 1+r).
#  * Aggregate output : asset demand  A_t = <a'_t, D_t>  (bonds carried to t+1).
#  * Market clearing  : A_t = B  for all t  =>  dA_t = 0.
#  * Steady state     : policy a'_ss = ss.ap, transition Λ_ss = ss.T (row-stoch.,
#                       distributions evolve as D_{t+1} = T' D_t), D_ss = ss.g.
#
#  Household (heterogeneous-agent) Jacobian   J[t,s] = dA_t / dq_s   is built with
#  the paper's "fake news" algorithm (Proposition 1 and the 4-step recipe, p.15):
#     1. one backward iteration -> curlyY_s (date-0 demand response to a q-shock
#        s periods ahead) and curlyD_s (resulting change in the date-1 distribution);
#     2. expectation vectors  E_t = (Λ_ss)^t a'_ss   via  E_t = T E_{t-1};
#     3. fake-news matrix  F[0,s]=curlyY_s,  F[t,s]=E_{t-1}' curlyD_s  (t>=1);
#     4. Jacobian recursion  J[t,s] = J[t-1,s-1] + F[t,s].
#
#  General equilibrium.  A one-time perturbation h0 of the initial distribution
#  (the same shock the FAME code feeds in) enters only through a "ghost" run with
#  prices held at the steady state:
#        dA^ghost_t = <a'_ss, (T')^t h0> = E_t' h0.
#  Clearing dA_t = J dq + dA^ghost = 0 then gives the equilibrium price path
#        dq = -(J)^{-1} dA^ghost,        dr_t = -dq_t / q_ss^2.
#
#  This is exactly the linear equilibrium the FAME solves, so the two IRFs agree
#  up to truncation (horizon T) and numerical-differentiation error.
# =============================================================================

include("huggett_fame.jl")   # steady state, EGM, make_T, make_M, solve_fame, ...

# -----------------------------------------------------------------------------
# Step 1.  Fake-news primitives via one backward iteration.
#   Returns
#     curlyY :: Vector{Float64}      length T, curlyY[s+1] = <da'_0, D_ss> for a
#                                    unit q-shock s periods ahead;
#     curlyD :: Matrix{Float64}      n×T, column s+1 = dD_1 for that same shock.
#   Central differences in the price/continuation perturbation give a clean
#   first-order operator.
# -----------------------------------------------------------------------------
function ssj_fake_news(ss; T::Int, dq::Float64 = 1e-4)
    p = PARAMS[]; agrid = ss.agrid; Π = ss.Π
    n = p.na * p.ny
    css, apss = ss.c, ss.ap
    g = ss.g
    M = make_M(apss, g, Π, p, agrid)          # push-forward derivative (constant)

    curlyY = zeros(T)
    curlyD = zeros(n, T)

    dc_prev = zeros(p.na, p.ny)               # date-0 consumption response, distance k-1
    for k in 0:T-1
        if k == 0
            # distance 0: shock the SPOT price, continuation fixed at steady state
            EWa = continuation_EWa(css, Π, p)
            cP, apP = egm_policy(EWa, ss.q + dq, p, agrid)
            cM, apM = egm_policy(EWa, ss.q - dq, p, agrid)
        else
            # distance k>=1: spot price steady, continuation perturbed by dc_prev
            EWaP = continuation_EWa(css .+ dq .* dc_prev, Π, p)
            EWaM = continuation_EWa(css .- dq .* dc_prev, Π, p)
            cP, apP = egm_policy(EWaP, ss.q, p, agrid)
            cM, apM = egm_policy(EWaM, ss.q, p, agrid)
        end
        dc_k  = (cP  .- cM)  ./ (2dq)          # da'/.. and dc/.. (date-0, distance k)
        dap_k = (apP .- apM) ./ (2dq)

        curlyY[k+1]    = dot(vec(dap_k), g)    # <da'_0, D_ss>
        curlyD[:, k+1] = M * vec(dap_k)        # dD_1 = (dΛ_0)' D_ss
        dc_prev = dc_k
    end
    return curlyY, curlyD
end

# -----------------------------------------------------------------------------
# Step 2.  Expectation vectors  E_t = (Λ_ss)^t a'_ss,  t = 0,...,T-1.
#   Column t+1 of the returned matrix is E_t (E_0 = a'_ss).  Recursion E_t = T E_{t-1}.
# -----------------------------------------------------------------------------
function ssj_expectations(ss; T::Int)
    n = size(ss.T, 1)
    E = zeros(n, T)
    e = vec(ss.ap)
    for t in 1:T
        @views E[:, t] .= e
        e = ss.T * e
    end
    return E
end

# -----------------------------------------------------------------------------
# Steps 3-4.  Fake-news matrix F and the household Jacobian J (both T×T).
# -----------------------------------------------------------------------------
function ssj_jacobian(curlyY, curlyD, E; T::Int)
    F = zeros(T, T)
    @views F[1, :] .= curlyY                          # t=0 row: date-0 demand response
    @views F[2:T, :] .= E[:, 1:T-1]' * curlyD         # F[t,s] = E_{t-1}' curlyD_s, t>=1

    J = copy(F)                                       # J[0,s]=F[0,s], J[t,0]=F[t,0]
    @inbounds for s in 2:T, t in 2:T
        J[t, s] += J[t-1, s-1]                        # Jt,s = Jt-1,s-1 + Ft,s
    end
    return J, F
end

# -----------------------------------------------------------------------------
# Full SSJ impulse response of the bond price to a one-time distribution shock h0.
#   Returns (dq, J, F, E) with dq[t+1] = q_t - q_ss for t = 0,...,T-1.
# -----------------------------------------------------------------------------
function ssj_irf(ss, h0; T::Int = 300, dq::Float64 = 1e-4)
    curlyY, curlyD = ssj_fake_news(ss; T = T, dq = dq)
    E = ssj_expectations(ss; T = T)
    J, F = ssj_jacobian(curlyY, curlyD, E; T = T)

    dAghost = E' * h0                 # dAghost[t+1] = E_t' h0 = <a'_ss, (T')^t h0>
    dqpath  = -(J \ dAghost)          # market clearing: J dq + dAghost = 0
    return dqpath, J, F, E
end

# -----------------------------------------------------------------------------
# Driver: solve the SSJ IRF, the FAME IRF, and compare them on the same shock.
# -----------------------------------------------------------------------------
function main_ssj(p::Huggett = Huggett(); T::Int = 300, Tshow::Int = 40)
    PARAMS[] = p
    println("="^70)
    println(" Huggett via the Sequence-Space Jacobian (fake-news algorithm)")
    println("="^70)
    ss = solve_steady(p)
    n  = p.na * p.ny
    @printf("  steady state: q* = %.6f   r* = %.6f\n", ss.q, ss.r)

    # same wealth-redistribution impulse as in huggett_fame.jl's main()
    agrid = ss.agrid
    iy0   = 1
    i_lo  = idx(searchsortedfirst(agrid, 0.0), iy0, p)
    i_hi  = idx(searchsortedfirst(agrid, 5.0), iy0, p)
    ε     = 1e-3
    h0    = zeros(n); h0[i_hi] += ε; h0[i_lo] -= ε

    # ---- SSJ ----
    dq_ssj, J, F, E = ssj_irf(ss, h0; T = T)
    dr_ssj = -dq_ssj ./ ss.q^2

    # ---- FAME (master equation), same impulse ----
    fame = solve_fame(ss; verbose = false)
    dq_fame = zeros(T); h = copy(h0)
    for t in 1:T
        dq_fame[t] = dot(fame.Qp, h)
        h = fame.DΦ * h
    end
    dr_fame = -dq_fame ./ ss.q^2

    println("\n  Interest-rate IRF (bps, annual):  SSJ vs. master equation (FAME)")
    @printf("    t        dr SSJ (bps)      dr FAME (bps)        diff (bps)\n")
    for t in 1:8
        @printf("   %2d        %+11.4f       %+11.4f       %+.2e\n",
                t-1, 1e4*dr_ssj[t], 1e4*dr_fame[t], 1e4*(dr_ssj[t]-dr_fame[t]))
    end
    maxdiff = maximum(abs.(dr_ssj[1:Tshow] .- dr_fame[1:Tshow]))
    @printf("    max|SSJ-FAME| over t<%d = %.3e bps\n", Tshow, 1e4*maxdiff)

    # ---- plot both IRFs (propagation tail) ----
    tg = 1:(Tshow-1)
    plt = plot(tg, 1e4 .* dr_ssj[2:Tshow];
               label = "SSJ", lw = 2, marker = :circle, ms = 3,
               xlabel = "period t", ylabel = "Δr_t (bps, annual)",
               title = "Huggett interest-rate IRF: SSJ vs master equation",
               legend = :topright, framestyle = :box)
    plot!(plt, tg, 1e4 .* dr_fame[2:Tshow];
          label = "FAME", lw = 2, ls = :dash, marker = :diamond, ms = 3)
    hline!(plt, [0.0]; label = "", lc = :black, lw = 0.5, alpha = 0.5)
    out = joinpath(@__DIR__, "huggett_ssj_vs_fame.png")
    savefig(plt, out)
    @printf("\n  saved comparison plot to %s\n", out)

    return (; ss, dq_ssj, dq_fame, J, F, E)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_ssj()
end
