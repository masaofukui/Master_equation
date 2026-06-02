# =============================================================================
#  Performance comparison of the three solution methods for the Huggett IRF:
#     FAME -- First-order Approximation to the Master Equation   (huggett_fame.jl)
#     SSJ  -- Sequence-Space Jacobian / fake-news algorithm      (huggett_ssj.jl)
#     MIT  -- nonlinear perfect-foresight transition             (huggett_mit.jl)
#
#  Run with:  julia bench_ssj_fame.jl
# =============================================================================

include("huggett_fame.jl")
include("huggett_mit.jl")
include("huggett_ssj.jl")
using BenchmarkTools, Printf

const Tssj = 300     # SSJ truncation horizon
const Tmit = 120     # MIT transition horizon

p  = Huggett()
PARAMS[] = p
ss = solve_steady(p)
n  = p.na * p.ny

h0 = redistribution_shock(ss, p; κ = 1e-2)   # same shock as toplevel_huggett.jl

# ---- warm up (force compilation of each path) ----
solve_fame(ss; verbose = false)
ssj_irf(ss, h0; T = Tssj)
mit_transition(ss, h0; Tmax = Tmit)

# ---- benchmark each method ----
bf = @benchmark solve_fame($ss; verbose = false)
bs = @benchmark ssj_irf($ss, $h0; T = $Tssj)
bm = @benchmark mit_transition($ss, $h0; Tmax = $Tmit)

# ---- summary table ----
row(b) = (min = minimum(b).time / 1e6,        # ms
          med = median(b).time / 1e6,         # ms
          mem = minimum(b).memory / 2^20,     # MiB
          alloc = minimum(b).allocs)

methods = [("FAME (master equation)", row(bf)),
           ("SSJ  (sequence-space J)", row(bs)),
           ("MIT  (nonlinear trans.)", row(bm))]

slowest = maximum(m.med for (_, m) in methods)

println("\n", "="^78)
println(" Performance: Huggett interest-rate IRF   (n = $n states, SSJ T = $Tssj, MIT T = $Tmit)")
println("="^78)
@printf(" %-24s %10s %12s %12s %10s %9s\n",
        "method", "min (ms)", "median (ms)", "memory (MiB)", "allocs", "speedup")
println("-"^78)
for (name, m) in methods
    @printf(" %-24s %10.3f %12.3f %12.2f %10d %8.1f×\n",
            name, m.min, m.med, m.mem, m.alloc, slowest / m.med)
end
println("-"^78)
println(" speedup = (slowest median) / (method median);  larger is faster")
println("="^78)

# ---- render the same table to a PDF (Computer Modern, like the other figures) ----
commas(x) = replace(string(round(Int, x)), r"(?<=\d)(?=(\d{3})+$)" => ",")

function save_perf_table_pdf(methods, slowest; path,
        title = "Huggett interest-rate IRF: solver performance")
    headers = ["method", "min (ms)", "median (ms)", "memory (MiB)", "allocs", "speedup"]
    xcol    = [0.02, 0.40, 0.56, 0.74, 0.89, 0.99]
    halign  = [:left, :right, :right, :right, :right, :right]
    nrow    = length(methods)
    plt = plot(; xlim = (0, 1), ylim = (0, 1), framestyle = :none, legend = false,
               grid = false, ticks = nothing,
               size = (980, 110 + 40 * (nrow + 1)), fontfamily = "Computer Modern",
               title = title, titlefontsize = 16,
               left_margin = 4Plots.mm, right_margin = 4Plots.mm, top_margin = 3Plots.mm)
    ytop = 0.86
    dy   = 0.80 / (nrow + 1)
    for (x, h, ha) in zip(xcol, headers, halign)
        annotate!(plt, x, ytop, text(h, 14, ha, "Computer Modern"))
    end
    yline = ytop - 0.5dy
    plot!(plt, [0.0, 1.0], [yline, yline]; lc = :black, lw = 1.2)
    for (r, (name, m)) in enumerate(methods)
        y = ytop - r * dy
        vals = [name,
                @sprintf("%.1f", m.min),
                @sprintf("%.1f", m.med),
                @sprintf("%.0f", m.mem),
                commas(m.alloc),
                @sprintf("%.0f×", slowest / m.med)]
        for (x, v, ha) in zip(xcol, vals, halign)
            annotate!(plt, x, y, text(v, 13, ha, "Computer Modern"))
        end
    end
    savefig(plt, path)
    return path
end

tbl_pdf = joinpath(@__DIR__, "figure", "performance_table.pdf")
mkpath(dirname(tbl_pdf))
save_perf_table_pdf(methods, slowest; path = tbl_pdf)
@printf(" saved performance table to %s\n", tbl_pdf)
