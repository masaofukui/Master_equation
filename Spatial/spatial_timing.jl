# =============================================================================
#  Timing benchmark: FAME vs SSJ as the number of locations J grows.
#
#  Both methods share the same steady state, so we time only the LINEARIZATION
#  step that differs:
#    * SSJ  : fake-news household Jacobians + GE assembly + sequence-space solve
#             (cost driven by the (1+2J)·T square system),
#    * FAME : household + continuation Jacobians + the coupled master-equation
#             fixed point (cost driven by n×n operations, n = n_a · J, and by the
#             J dense n×n migration Jacobians).
#  The horizon T is fixed; the asset grid n_a is fixed; only J varies.
#
#  Results are appended to figure/spatial_timing.csv AS THEY ARE COMPUTED, so a
#  long or interrupted run still leaves usable data.  Restyle the figure from the
#  stored CSV with spatial_timing_plot.jl (no need to re-run the benchmark).
#
#  FAME's cost grows steeply in J (it forms J dense n×n matrices and does J
#  n×n matmuls per fixed-point iteration).  Once a FAME timing exceeds
#  FAME_BUDGET seconds we stop ATTEMPTING FAME for larger J (recording NaN),
#  while still timing SSJ everywhere — this bounds the total run time and avoids
#  the multi-GB allocations FAME would need at large J.
#
#  Run:  julia spatial_timing.jl   ->  figure/spatial_timing.{csv,pdf}
# =============================================================================

if !isdefined(Main, :SpatialHuggett)
    include("SpatialHuggett.jl")
end
using .SpatialHuggett
include("spatial_ssj.jl")
include("spatial_fame.jl")
include("spatial_timing_plot.jl")
using Printf

const TH = 150                       # SSJ / IRF horizon
const NA = 80                        # asset grid points (fixed across J)
const JS = [2, 5, 10, 20]            # numbers of locations to compare
const FAME_BUDGET = 300.0            # s: stop attempting FAME for larger J past this
const SSJ_BUDGET  = 600.0            # s: stop attempting SSJ for larger J past this

const CSV = joinpath(@__DIR__, "..", "figure", "spatial_timing.csv")

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

# minimum over `reps` runs; NaN if the method throws (e.g. out of memory)
function best_time(f, ss, p; reps)
    t = Inf
    for _ in 1:reps
        GC.gc()
        try
            t = min(t, f(ss, p))
        catch err
            @warn "timing failed" exception = (err, catch_backtrace())
            return NaN
        end
    end
    return t
end

function run_timing()
    # ---- warm up the JIT (compile everything once, at a small J) ----
    println("warming up (JIT compilation) ...")
    let p = SParams(J = 2, na = NA)
        ss = solve_steady(p; verbose = false)
        time_ssj(ss, p); time_fame(ss, p)
    end

    mkpath(dirname(CSV))
    open(CSV, "w") do io
        println(io, "J,n,t_ssj,t_fame")
    end

    @printf("\n  %-4s %-7s %12s %12s\n", "J", "n", "SSJ (s)", "FAME (s)")
    skip_ssj = false; skip_fame = false
    for J in JS
        p  = SParams(J = J, na = NA)
        ss = solve_steady(p; verbose = false)
        n  = n_states(p)
        reps = J <= 10 ? 2 : 1

        ts = skip_ssj ? NaN : best_time(time_ssj, ss, p; reps = reps)
        (!isfinite(ts) || ts > SSJ_BUDGET) && (skip_ssj = true)

        # FAME is timed once (expensive); skip for larger J once over budget
        tf = skip_fame ? NaN : best_time(time_fame, ss, p; reps = 1)
        (!isfinite(tf) || tf > FAME_BUDGET) && (skip_fame = true)

        @printf("  %-4d %-7d %12s %12s\n", J, n,
                isfinite(ts) ? @sprintf("%.3f", ts) : "skip",
                isfinite(tf) ? @sprintf("%.3f", tf) : "skip")
        open(CSV, "a") do io
            @printf(io, "%d,%d,%.6f,%.6f\n", J, n, ts, tf)
        end
    end
    @printf("\n  saved timing data to %s\n", normpath(CSV))
end

run_timing()
plot_spatial_timing(CSV; na = NA, T = TH)
