# =============================================================================
#  Cross-check tying the FAME and SSJ linearizations together.
#
#  Both methods linearize the SAME model around the SAME steady state.  A clean
#  shared object is the contemporaneous aggregate saving response to the bond
#  rate, ∂A_2/∂r_1 (a one-time rate shock at date 1, continuation at the steady
#  state).  This equals
#     * SSJ : the household Jacobian entry J^{A,r}[2,1] (fake-news), and
#     * FAME: the push-forward of the policy sensitivity,
#             aᵀ (M_a + M_μ) · (∂a'/∂r),
#  computed here three independent ways (direct nonlinear FD, SSJ, FAME).
#
#  Run:  julia spatial_crosscheck.jl
# =============================================================================

include("SpatialHuggett.jl")
using .SpatialHuggett
using LinearAlgebra, Printf
const SH = SpatialHuggett

p  = SParams()
ss = solve_steady(p; verbose = false)
@printf("steady state:  r=%.6f  w=%s  L=%s\n", ss.r,
        string(round.(ss.w, digits=4)), string(round.(ss.L, digits=4)))

na, J = p.na, p.J; n = SH.n_states(p); ag = agrid(p)
avec = [ag[(i-1)%na + 1] for i in 1:n]                    # asset value of each state
g = ss.g; Vss, Css = ss.V, ss.C
ε = 1e-6

# policy at r ± ε (continuation fixed at steady state) ------------------------
bwp = SH.backward_step(Vss, Css, ss.r + ε, ss.w, ss.P, ss.χ, p)
bwm = SH.backward_step(Vss, Css, ss.r - ε, ss.w, ss.P, ss.χ, p)
μp  = SH.migration_probs(Vss, bwp.ap, ss.χ, p)
μm  = SH.migration_probs(Vss, bwm.ap, ss.χ, p)
dap_r = vec((bwp.ap .- bwm.ap) ./ (2ε))                   # ∂a'/∂r  (per state)

# (1) DIRECT nonlinear finite difference of A_2 = Σ a g_2,  g_2 = T(r)' g ------
g2p = SH.build_T(bwp.ap, μp, p)' * g
g2m = SH.build_T(bwm.ap, μm, p)' * g
dA2_direct = (dot(avec, g2p) - dot(avec, g2m)) / (2ε)

# (2) FAME: analytic push-forward of dap_r through (M_a + M_μ) -----------------
Ma = zeros(n, n); Mμap = zeros(n, n)
dμda = zeros(n, J); hap = 1e-6
for j in 1:J, ia in 1:na
    i = idx(ia, j, p)
    ap_p = copy(ss.ap); ap_p[ia, j] += hap
    ap_m = copy(ss.ap); ap_m[ia, j] -= hap
    mp = SH.migration_probs(Vss, ap_p, ss.χ, p)
    mm = SH.migration_probs(Vss, ap_m, ss.χ, p)
    for k in 1:J
        dμda[i, k] = (mp[ia, j, k] - mm[ia, j, k]) / (2hap)
    end
end
for j in 1:J, ia in 1:na
    i  = idx(ia, j, p)
    ka = clamp(searchsortedlast(ag, ss.ap[ia, j]), 1, na - 1)
    Δ  = ag[ka+1] - ag[ka]
    ww = clamp((ss.ap[ia, j] - ag[ka]) / Δ, 0.0, 1.0)
    gi = g[i]
    for kk in 1:J
        Ma[idx(ka,   kk, p), i] -= gi * ss.μ[ia, j, kk] / Δ
        Ma[idx(ka+1, kk, p), i] += gi * ss.μ[ia, j, kk] / Δ
        c = gi * dμda[i, kk]
        Mμap[idx(ka,   kk, p), i] += c * (1 - ww)
        Mμap[idx(ka+1, kk, p), i] += c * ww
    end
end
dA2_fame = dot(avec, (Ma .+ Mμap) * dap_r)

# (3) SSJ: same object via the fake-news household Jacobian J^{A,r}[2,1] -------
# (reproduce the minimal fake-news increment for output A, input r, horizon 1)
Tmat = ss.T
yA = avec
# curlyD[1]: date-2 distribution change from the date-1 (contemporaneous) policy
curlyD1 = (SH.build_T(bwp.ap, μp, p)' * g .- SH.build_T(bwm.ap, μm, p)' * g) ./ (2ε)
dA2_ssj = dot(yA, curlyD1)            # F[2,1] = curlyE_0 · curlyD[1] = yA · curlyD[1]

println("\ncontemporaneous aggregate saving response  ∂A₂/∂r₁:")
@printf("  (1) direct nonlinear FD       = %+.8e\n", dA2_direct)
@printf("  (2) FAME push-forward         = %+.8e   (rel.err %.2e)\n",
        dA2_fame, abs(dA2_fame - dA2_direct)/abs(dA2_direct))
@printf("  (3) SSJ fake-news increment   = %+.8e   (rel.err %.2e)\n",
        dA2_ssj, abs(dA2_ssj - dA2_direct)/abs(dA2_direct))
