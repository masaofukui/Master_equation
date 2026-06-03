# =============================================================================
#  Sequence-space Jacobian (SSJ) for the multi-region Huggett economy.
#
#  Implements the fake-news algorithm of Auclert, Bardóczy, Rognlie & Straub
#  (2021), following ODSP Appendix F.5, to compute the first-order transition of
#  the status-quo economy to an aggregate productivity shock.
#
#  Household block inputs  : price paths (r_t, w_{jt}, P_{jt})
#  Household block outputs : aggregate bond demand A_t, regional populations
#                            L_{jt}, and real consumption Cagg_{jt}.
#  GE unknowns             : (r_t, w_{jt}, L_{jt}); P_{jt} static from (w,L,A).
#  GE targets              : bond market, labor markets, population consistency.
#
#  Includes a nonlinear perfect-foresight transition (`td_solve`) and a check of
#  the household Jacobians against direct numerical differentiation.
#
#  Run:  julia spatial_ssj.jl
# =============================================================================

if !isdefined(Main, :SpatialHuggett)
    include("SpatialHuggett.jl")
end
using .SpatialHuggett
using LinearAlgebra, Printf, Plots

const SH = SpatialHuggett

# -----------------------------------------------------------------------------
# Output reward vectors y_o (length n) with steady-state aggregate O = y_o · g.
#   :A        bond holdings (asset of the state)        -> depends on dist only
#   (:L, j)   indicator of region j                      -> dist only
#   (:C, j)   consumption in region j (real)             -> policy × dist
# -----------------------------------------------------------------------------
function reward_vectors(ss, p)
    n = SH.n_states(p); ag = agrid(p)
    yA = zeros(n)
    yL = [zeros(n) for _ in 1:p.J]
    yC = [zeros(n) for _ in 1:p.J]
    for j in 1:p.J, ia in 1:p.na
        i = idx(ia, j, p)
        yA[i]      = ag[ia]
        yL[j][i]   = 1.0
        yC[j][i]   = ss.C[ia, j]
    end
    return (; yA, yL, yC)
end

# -----------------------------------------------------------------------------
# Fake-news Jacobians of household outputs w.r.t. household inputs (r, w_k, P_k).
# Returns a Dict keyed by (output, input) -> T×T matrix J with J[t,s]=∂O_t/∂I_s.
#   outputs: :A, (:L,j), (:C,j);   inputs: :r, (:w,k), (:P,k)
# -----------------------------------------------------------------------------
function household_jacobians(ss, p; T = 150, ε = 1e-4)
    na, J = p.na, p.J
    n = SH.n_states(p)
    ag = agrid(p)
    Vss, Css, apss = ss.V, ss.C, ss.ap
    μss = ss.μ
    gss = ss.g
    Tmat = ss.T                      # row-stochastic; Λ = Tmat', Λ' = Tmat
    rew = reward_vectors(ss, p)

    # one linearized backward step (ε-level perturbations); returns level policies
    function dstep(dVnext, dCnext, dr, dw, dP)
        bw = SH.backward_step(Vss .+ dVnext, Css .+ dCnext,
                              ss.r + dr, ss.w .+ dw, ss.P .+ dP, ss.χ, p)
        μcur = SH.migration_probs(Vss .+ dVnext, bw.ap, ss.χ, p)
        return bw.ap, μcur, (bw.Vcurr .- Vss), (bw.C .- Css)
    end

    # expectation vectors curlyE[τ+1] = (Λ')^τ y = Tmat^τ y, τ=0..T-1
    function expectation_vectors(y)
        E = Vector{Vector{Float64}}(undef, T)
        E[1] = copy(y)
        for τ in 2:T
            E[τ] = Tmat * E[τ-1]
        end
        return E
    end
    EA  = expectation_vectors(rew.yA)
    EL  = [expectation_vectors(rew.yL[j]) for j in 1:J]
    EC  = [expectation_vectors(rew.yC[j]) for j in 1:J]

    inputs = vcat([:r], [(:w, k) for k in 1:J], [(:P, k) for k in 1:J])
    Jac = Dict{Tuple{Any,Any},Matrix{Float64}}()

    for inp in inputs
        # set the contemporaneous price perturbation for this input
        dr0 = 0.0; dw0 = zeros(J); dP0 = zeros(J)
        if inp == :r
            dr0 = ε
        elseif inp[1] == :w
            dw0[inp[2]] = ε
        else
            dP0[inp[2]] = ε
        end

        # curlyY[o][s] (date-1 output response) and curlyD[s] (date-2 dist change)
        curlyY_A = zeros(T)
        curlyY_L = [zeros(T) for _ in 1:J]
        curlyY_C = [zeros(T) for _ in 1:J]
        curlyD   = [zeros(n) for _ in 1:T]

        dV = zeros(na, J); dC = zeros(na, J)
        for s in 1:T
            if s == 1
                ap_s, μ_s, dV, dC = dstep(zeros(na,J), zeros(na,J), dr0, dw0, dP0)
            else
                ap_s, μ_s, dV, dC = dstep(dV, dC, 0.0, zeros(J), zeros(J))
            end
            # date-1 output responses (g fixed at gss): only Cagg has a policy term
            for j in 1:J, ia in 1:na
                curlyY_C[j][s] += dC[ia, j] * gss[idx(ia, j, p)] / ε
            end
            # date-2 distribution change from the date-1 policy response
            Tp = SH.build_T(ap_s, μ_s, p)
            curlyD[s] .= (Tp' * gss .- gss) ./ ε
        end

        # assemble fake-news matrix F and Jacobian J[t,s]=F[t,s]+J[t-1,s-1]
        function assemble(curlyY, E)
            F = zeros(T, T)
            for s in 1:T
                F[1, s] = curlyY[s]
                for t in 2:T
                    F[t, s] = dot(E[t-1], curlyD[s])     # E[t-1]=(Λ')^{t-2} y
                end
            end
            Jm = zeros(T, T)
            for s in 1:T, t in 1:T
                Jm[t, s] = F[t, s] + (t > 1 && s > 1 ? Jm[t-1, s-1] : 0.0)
            end
            return Jm
        end

        Jac[(:A, inp)] = assemble(zeros(T), EA)
        for j in 1:J
            Jac[((:L, j), inp)] = assemble(zeros(T), EL[j])
            Jac[((:C, j), inp)] = assemble(curlyY_C[j], EC[j])
        end
    end
    return Jac, inputs
end

# -----------------------------------------------------------------------------
# Static partials of the trade/price block w.r.t. (w_k, L_k, Z), where Z scales
# origin `mshock`'s productivity:  A(z)[k,j] = A_ss[k,j]·(1+z if k=mshock).
# -----------------------------------------------------------------------------
Ashock(A, mshock, z) = (B = copy(A); B[mshock, :] .*= (1 + z); B)

function static_partials(ss, p, mshock; h = 1e-6)
    J = p.J; A = ss.A; w = ss.w; L = ss.L
    P0(w_, L_, z) = price_index(w_, L_, Ashock(A, mshock, z), p)
    π0(w_, L_, z) = trade_shares(w_, L_, Ashock(A, mshock, z), p)

    dPdw = zeros(J, J); dPdL = zeros(J, J); dPdZ = zeros(J)
    dπdw = zeros(J, J, J); dπdL = zeros(J, J, J); dπdZ = zeros(J, J)

    for k in 1:J
        wp = copy(w); wp[k] += h; wm = copy(w); wm[k] -= h
        dPdw[:, k] = (P0(wp, L, 0) .- P0(wm, L, 0)) ./ (2h)
        dπdw[:, :, k] = (π0(wp, L, 0) .- π0(wm, L, 0)) ./ (2h)
        Lp = copy(L); Lp[k] += h; Lm = copy(L); Lm[k] -= h
        dPdL[:, k] = (P0(w, Lp, 0) .- P0(w, Lm, 0)) ./ (2h)
        dπdL[:, :, k] = (π0(w, Lp, 0) .- π0(w, Lm, 0)) ./ (2h)
    end
    dPdZ = (P0(w, L, h) .- P0(w, L, -h)) ./ (2h)
    dπdZ = (π0(w, L, h) .- π0(w, L, -h)) ./ (2h)
    return (; dPdw, dPdL, dPdZ, dπdw, dπdL, dπdZ)
end

# -----------------------------------------------------------------------------
# Nonlinear perfect-foresight household transition along given price paths.
#   rpath (T), wpath (J×T), Ppath (J×T).  Backward induction from terminal ss,
#   then forward simulation of the distribution from g_0 = g_ss.
#   Returns aggregate paths A_t, L_{jt}, Cagg_{jt}.
# -----------------------------------------------------------------------------
function td_household(ss, p, rpath, wpath, Ppath; T = length(rpath))
    na, J = p.na, p.J; ag = agrid(p)
    # backward pass
    apts = Vector{Matrix{Float64}}(undef, T)
    μts  = Vector{Array{Float64,3}}(undef, T)
    Vnext = copy(ss.V); Cnext = copy(ss.C)
    for t in T:-1:1
        bw = SH.backward_step(Vnext, Cnext, rpath[t], wpath[:, t], Ppath[:, t], ss.χ, p)
        μts[t]  = SH.migration_probs(Vnext, bw.ap, ss.χ, p)
        apts[t] = bw.ap
        Vnext = bw.Vcurr; Cnext = bw.C
        # store consumption policy for output (recompute from budget)
    end
    # forward pass
    Apath = zeros(T); Lpath = zeros(J, T); Cpath = zeros(J, T)
    g = copy(ss.g)
    for t in 1:T
        # consumption policy at t from budget given apol
        ap = apts[t]
        for j in 1:J, ia in 1:na
            i = idx(ia, j, p)
            c = ((1 + rpath[t]) * ag[ia] + wpath[j, t] - ap[ia, j]) / Ppath[j, t]
            Apath[t]    += g[i] * ag[ia]
            Lpath[j, t] += g[i]
            Cpath[j, t] += g[i] * c
        end
        Tt = SH.build_T(apts[t], μts[t], p)
        g = Tt' * g
    end
    return (; Apath, Lpath, Cpath)
end

# -----------------------------------------------------------------------------
# Assemble and solve the GE system for the impulse response to dZ (length T).
#   Unknowns U = (dr, dw_j, dL_j) stacked over t; P static; targets bond/labor/pop.
# -----------------------------------------------------------------------------
function solve_ssj(ss, p; T = 150, mshock = 1, z0 = -0.01, ρz = 0.8)
    J = p.J
    Jac, inputs = household_jacobians(ss, p; T = T)
    sp = static_partials(ss, p, mshock)
    rew = reward_vectors(ss, p)

    I_T = Matrix{Float64}(I, T, T)

    # reduced output Jacobians w.r.t. (r, w_k, L_k, Z): compose P-channel
    # J^{o,w_k}_red = J^{o,w_k} + Σ_m (∂P_m/∂w_k) J^{o,P_m}, etc.
    function Jred(o, var, k)
        Z = zeros(T, T)
        if var == :r
            Z .+= Jac[(o, :r)]
        elseif var == :w
            Z .+= Jac[(o, (:w, k))]
            for m in 1:J
                Z .+= sp.dPdw[m, k] .* Jac[(o, (:P, m))]
            end
        elseif var == :L
            for m in 1:J
                Z .+= sp.dPdL[m, k] .* Jac[(o, (:P, m))]
            end
        elseif var == :Z
            for m in 1:J
                Z .+= sp.dPdZ[m] .* Jac[(o, (:P, m))]
            end
        end
        return Z
    end

    ndim = (1 + 2J) * T
    H = zeros(ndim, ndim)
    rhs = zeros(ndim)

    # column index ranges
    colr()      = 1:T
    colw(j)     = (T + (j-1)*T) .+ (1:T)
    colL(j)     = (T + J*T + (j-1)*T) .+ (1:T)
    # row index ranges
    rowB()      = 1:T
    rowLab(i)   = (T + (i-1)*T) .+ (1:T)
    rowPop(i)   = (T + J*T + (i-1)*T) .+ (1:T)

    addB!(rows, cols, blk) = (@views H[rows, cols] .+= blk)

    # steady-state levels
    Pss, πss, wss, Lss = ss.P, ss.π, ss.w, ss.L
    Cagg = [sum(ss.g[idx(ia, j, p)] * ss.C[ia, j] for ia in 1:p.na) for j in 1:J]
    Ess  = [Pss[j] * Cagg[j] for j in 1:J]

    dZ = [z0 * ρz^(t-1) for t in 1:T]

    # ---- (1) bond market:  dA_t = 0 ----
    addB!(rowB(), colr(), Jred(:A, :r, 0))
    for k in 1:J
        addB!(rowB(), colw(k), Jred(:A, :w, k))
        addB!(rowB(), colL(k), Jred(:A, :L, k))
    end
    rhs[rowB()] .= .-(Jred(:A, :Z, 0) * dZ)
    # The date-1 bond market is vacuous: A_1 = Σ a g_ss = 0 is predetermined, so
    # that row is identically zero.  Replace it with the terminal condition
    # dr_T = 0 (the rate has returned to steady state by the truncation horizon).
    @views H[rowB()[1], :] .= 0.0
    H[rowB()[1], colr()[T]] = 1.0
    rhs[rowB()[1]] = 0.0

    # ---- (3) population consistency:  dL_i - dLpop_i = 0 ----
    for i in 1:J
        addB!(rowPop(i), colL(i), I_T)                       # dL_i
        addB!(rowPop(i), colr(),  .-Jred((:L, i), :r, 0))
        for k in 1:J
            addB!(rowPop(i), colw(k), .-Jred((:L, i), :w, k))
            addB!(rowPop(i), colL(k), .-Jred((:L, i), :L, k))
        end
        rhs[rowPop(i)] .= Jred((:L, i), :Z, 0) * dZ
    end

    # ---- (2) labor market:  L_i dw_i + w_i dL_i - Σ_j d(π_ij P_j Cagg_j) = 0 ----
    # Region 1's wage is the numeraire (dw_1 = 0); by Walras' law its labor-market
    # equation is redundant, so we replace row block rowLab(1) with dw_1 = 0.
    addB!(rowLab(1), colw(1), I_T)
    for i in 2:J
        addB!(rowLab(i), colw(i), Lss[i] .* I_T)
        addB!(rowLab(i), colL(i), wss[i] .* I_T)
        for k in 1:J
            # dπ_ij and dP_j static parts (diagonal in time) and dCagg_j (dynamic)
            blkw = zeros(T, T); blkL = zeros(T, T)
            for j in 1:J
                blkw .-= (Ess[j] * sp.dπdw[i, j, k]) .* I_T
                blkw .-= (πss[i, j] * Cagg[j] * sp.dPdw[j, k]) .* I_T
                blkw .-= (πss[i, j] * Pss[j]) .* Jred((:C, j), :w, k)
                blkL .-= (Ess[j] * sp.dπdL[i, j, k]) .* I_T
                blkL .-= (πss[i, j] * Cagg[j] * sp.dPdL[j, k]) .* I_T
                blkL .-= (πss[i, j] * Pss[j]) .* Jred((:C, j), :L, k)
            end
            addB!(rowLab(i), colw(k), blkw)
            addB!(rowLab(i), colL(k), blkL)
        end
        blkr = zeros(T, T)
        for j in 1:J
            blkr .-= (πss[i, j] * Pss[j]) .* Jred((:C, j), :r, 0)
        end
        addB!(rowLab(i), colr(), blkr)
        # RHS: shock terms
        rj = zeros(T)
        for j in 1:J
            rj .+= (Ess[j] * sp.dπdZ[i, j]) .* dZ
            rj .+= (πss[i, j] * Cagg[j] * sp.dPdZ[j]) .* dZ
            rj .+= (πss[i, j] * Pss[j]) .* (Jred((:C, j), :Z, 0) * dZ)
        end
        rhs[rowLab(i)] .= rj
    end

    if get(ENV, "SSJ_DIAG", "0") == "1"
        F = svd(H)
        nz = count(<(1e-9 * F.S[1]), F.S)
        @printf("  [diag] ndim=%d  cond=%.2e  #tiny singular vals=%d\n",
                ndim, F.S[1]/F.S[end], nz)
        # which columns are ~empty
        cnorm = [norm(@view H[:, c]) for c in 1:ndim]
        rnorm = [norm(@view H[r, :]) for r in 1:ndim]
        @printf("  [diag] min col norm=%.2e (col %d)  min row norm=%.2e (row %d)\n",
                minimum(cnorm), argmin(cnorm), minimum(rnorm), argmin(rnorm))
        return (; H, rhs, ndim)
    end
    U = H \ rhs

    dr = U[colr()]
    dw = [U[colw(j)] for j in 1:J]
    dL = [U[colL(j)] for j in 1:J]
    return (; dr, dw, dL, dZ, Jac, sp, T, mshock)
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

    T = 150
    println("\n", "="^70)
    println(" Validate household Jacobians vs direct numerical differentiation")
    println("="^70)
    Jac, _ = household_jacobians(ss, p; T = T)
    # direct: shock r at date s0; compare dA_t, dCagg_1,t to Jacobian columns
    s0 = 10; ε = 1e-5
    rp = fill(ss.r, T); rp[s0] += ε
    wp = repeat(ss.w, 1, T); Pp = repeat(ss.P, 1, T)
    td  = td_household(ss, p, rp, wp, Pp; T = T)
    td0 = td_household(ss, p, fill(ss.r, T), wp, Pp; T = T)
    dA_num = (td.Apath .- td0.Apath) ./ ε
    dC1_num = (td.Cpath[1, :] .- td0.Cpath[1, :]) ./ ε
    eA = maximum(abs.(dA_num .- Jac[(:A, :r)][:, s0]))
    eC = maximum(abs.(dC1_num .- Jac[((:C, 1), :r)][:, s0]))
    @printf("  max|dA  fake-news - direct|     = %.3e\n", eA)
    @printf("  max|dCagg1 fake-news - direct|  = %.3e\n", eC)

    println("\n", "="^70)
    println(" SSJ general-equilibrium impulse response")
    println("="^70)
    out = solve_ssj(ss, p; T = T, mshock = 1, z0 = -0.01, ρz = 0.8)
    @printf("  shock: -1%% productivity in region %d (AR1 ρ=0.8)\n", out.mshock)
    @printf("  %-4s %10s %10s %10s %10s\n", "t", "dr(bp)", "dw1(%)", "dL1(%)", "dZ(%)")
    for t in 1:8
        @printf("  %-4d %10.4f %10.4f %10.4f %10.4f\n", t,
                1e4*out.dr[t], 100*out.dw[1][t]/ss.w[1],
                100*out.dL[1][t]/ss.L[1], 100*out.dZ[t])
    end

    # --- nonlinear consistency check: feed the LINEAR solution through the exact
    #     household transition; market residuals should be O(shock^2) ---------
    println("\n  nonlinear consistency check (residuals should be ~shock^2):")
    wp = repeat(ss.w, 1, T) .+ permutedims(hcat(out.dw...))      # J×T
    Lp = repeat(ss.L, 1, T) .+ permutedims(hcat(out.dL...))
    rp = ss.r .+ out.dr
    Pp = similar(wp)
    for t in 1:T
        Pp[:, t] = price_index(wp[:, t], Lp[:, t], Ashock(ss.A, out.mshock, out.dZ[t]), p)
    end
    td = td_household(ss, p, rp, wp, Pp; T = T)
    bond_res = maximum(abs.(td.Apath))
    pop_res  = maximum(abs.(td.Lpath .- Lp))
    lab_res  = maximum(abs.([wp[i,t]*Lp[i,t] -
                 sum(trade_shares(wp[:,t],Lp[:,t],Ashock(ss.A,out.mshock,out.dZ[t]),p)[i,j]*
                     Pp[j,t]*td.Cpath[j,t] for j in 1:p.J) for i in 1:p.J, t in 1:T]))
    @printf("    max|bond market|  = %.2e\n", bond_res)
    @printf("    max|pop consist.| = %.2e\n", pop_res)
    @printf("    max|labor market| = %.2e\n", lab_res)

    # --- plot the IRFs -------------------------------------------------------
    tt = 1:40
    p1 = plot(tt, 1e4 .* out.dr[tt]; lw = 2, marker = :circle, ms = 2,
              xlabel = "period t", ylabel = "Δr (bp)", title = "Bond rate", legend = false)
    p2 = plot(; xlabel = "period t", ylabel = "% deviation", title = "Population L_j")
    for j in 1:p.J
        plot!(p2, tt, 100 .* out.dL[j][tt] ./ ss.L[j]; lw = 2, label = "region $j")
    end
    p3 = plot(; xlabel = "period t", ylabel = "% deviation", title = "Wage w_j")
    for j in 1:p.J
        plot!(p3, tt, 100 .* out.dw[j][tt] ./ ss.w[j]; lw = 2, label = "region $j")
    end
    p4 = plot(tt, 100 .* out.dZ[tt]; lw = 2, ls = :dash, lc = :black,
              xlabel = "period t", ylabel = "%", title = "Productivity shock (region $(out.mshock))", legend = false)
    plt = plot(p1, p2, p3, p4; layout = (2, 2), size = (950, 680),
               plot_title = "SSJ impulse response to a -1% regional productivity shock")
    outpng = joinpath(@__DIR__, "..", "figure", "spatial_ssj_irf.png")
    savefig(plt, outpng)
    @printf("\n  saved IRF plot to %s\n", normpath(outpng))
    return ss, out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
