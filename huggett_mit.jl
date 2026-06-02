# =============================================================================
#  Nonlinear perfect-foresight (MIT-shock) transition for the Huggett economy.
#
#  FUNCTIONS ONLY.  Shares module scope with huggett_fame.jl (loaded together by
#  the toplevel driver run_huggett.jl), so it relies on:
#     - the `Huggett` parameter struct and the global `PARAMS::Ref{Huggett}`,
#     - the steady-state object `ss` (fields q, c, g, ap, agrid, Π),
#     - the household routines `egm_policy`, `continuation_EWa`, `make_T`.
#
#  It provides the fully nonlinear transition used to VALIDATE the linear FAME
#  and SSJ solutions; it is not needed to solve either of them.
# =============================================================================

# -----------------------------------------------------------------------------
# Nonlinear perfect-foresight (MIT-shock) transition, used to validate the FAME.
#   Start from g_0 = g^ss + h0 and compute the equilibrium price path {q_t} that
#   clears the bond market every period, with the economy returning to steady
#   state.  For small h0 the price path should match the linear FAME prediction
#   q_t - q^ss ≈ Q'·h_t,  h_{t+1} = (T'+G) h_t.
# -----------------------------------------------------------------------------
function mit_transition(ss, h0; Tmax = 120, damp = 0.5, maxit = 400)
    p = PARAMS[]; agrid = ss.agrid; Π = ss.Π; qss = ss.q
    css = ss.c
    q = fill(qss, Tmax + 1)
    cpath = [copy(css) for _ in 1:Tmax+1]
    clear(EWa, g) = begin
        qlo, qhi = p.β * (1 + 1e-4), 1.30
        qq = 0.5 * (qlo + qhi)
        for _ in 1:100
            _, ap = egm_policy(EWa, qq, p, agrid)
            fq = dot(vec(ap), g) - p.B
            abs(fq) < 1e-12 && break
            fq > 0 ? (qlo = qq) : (qhi = qq); qq = 0.5 * (qlo + qhi)
        end
        return qq
    end
    for it in 1:maxit
        # backward pass: consumption policy along the (current) price path
        cpath[Tmax+1] = css
        for t in Tmax:-1:1
            EWa = continuation_EWa(cpath[t+1], Π, p)
            ct, _ = egm_policy(EWa, q[t], p, agrid)
            cpath[t] = ct
        end
        # forward pass: clear the market each period given the predetermined g_t
        g = ss.g .+ h0
        qnew = copy(q)
        for t in 1:Tmax
            EWa = continuation_EWa(cpath[t+1], Π, p)
            qnew[t] = clear(EWa, g)
            _, ap = egm_policy(EWa, qnew[t], p, agrid)
            g = make_T(ap, Π, p, agrid)' * g
        end
        qnew[Tmax+1] = qss
        err = maximum(abs.(qnew .- q))
        q .= damp .* qnew .+ (1 - damp) .* q
        err < 1e-11 && break
    end
    return q[1:Tmax] .- qss          # dq_t for t = 0, 1, ..., Tmax-1
end

# -----------------------------------------------------------------------------
# Convenience: nonlinear MIT interest-rate impulse response (bps, annual) to a
# one-time distributional shock h0, for t = 0,...,Tshow-1.
# -----------------------------------------------------------------------------
function mit_irf_bps(ss, h0; Tshow::Int = 40, Tmax::Int = 120)
    dq = mit_transition(ss, h0; Tmax = max(Tmax, Tshow))
    return -1e4 .* dq[1:Tshow] ./ ss.q^2
end
