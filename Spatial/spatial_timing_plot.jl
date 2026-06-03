# =============================================================================
#  Plot the FAME-vs-SSJ timing comparison from stored results.
#
#  Reads figure/spatial_timing.csv  (columns: J, n, t_ssj, t_fame; seconds, with
#  NaN where a method was skipped/failed) and writes figure/spatial_timing.pdf.
#
#  This is intentionally SEPARATE from the benchmark (spatial_timing.jl) so the
#  figure can be restyled without re-running the (expensive) timing.
#
#  Run:  julia spatial_timing_plot.jl
# =============================================================================

using DelimitedFiles, Plots, Printf

const CSV_DEFAULT = joinpath(@__DIR__, "..", "figure", "spatial_timing.csv")
const PDF_DEFAULT = joinpath(@__DIR__, "..", "figure", "spatial_timing.pdf")

"""
    plot_spatial_timing(csv=CSV_DEFAULT; outpdf=PDF_DEFAULT, na=nothing, T=nothing)

Read the timing CSV and save a two-panel comparison of FAME and SSJ linearization
time vs J: linear scale (left) and log scale (right).
`na`/`T` are only used to annotate the overall title.
"""
function plot_spatial_timing(csv = CSV_DEFAULT; outpdf = PDF_DEFAULT,
                             na = nothing, T = nothing)
    raw, _ = readdlm(csv, ','; header = true)
    J     = Float64.(raw[:, 1])
    tssj  = Float64.(raw[:, 3])
    tfame = Float64.(raw[:, 4])

    finite(x, y) = (m = isfinite.(y); (x[m], y[m]))
    Js, ts = finite(J, tssj)
    Jf, tf = finite(J, tfame)

    style = (xlabel = "number of locations  J", framestyle = :box,
             fontfamily = "Computer Modern", guidefontsize = 14, tickfontsize = 12,
             legendfontsize = 12, titlefontsize = 14, grid = :y,
             left_margin = 7Plots.mm, bottom_margin = 7Plots.mm)
    fame_kw = (label = "FAME (master equation)", lw = 4, marker = :square,
               ms = 7, color = :crimson)
    ssj_kw  = (label = "SSJ (sequence-space Jacobian)", lw = 4, marker = :circle,
               ms = 7, color = :steelblue)

    # left: linear scale
    pl = plot(Jf, tf; ylabel = "linearization time (s)", title = "Linear scale",
              legend = :topleft, fame_kw..., style...)
    plot!(pl, Js, ts; ssj_kw...)

    # right: log scale
    pr = plot(Jf, tf; ylabel = "linearization time (s, log scale)",
              title = "Log scale", yscale = :log10, legend = :topleft,
              fame_kw..., style...)
    plot!(pr, Js, ts; ssj_kw...)

    ttl = "FAME vs SSJ: linearization time"
    if na !== nothing && T !== nothing
        ttl *= "  (\$n = n_a \\cdot J\$,  \$n_a = $(na)\$,  \$T = $(T)\$)"
    end
    plt = plot(pl, pr; layout = (1, 2), size = (1100, 480), plot_title = ttl,
               plot_titlefontsize = 15)

    mkpath(dirname(outpdf))
    savefig(plt, outpdf)
    @printf("  saved timing figure to %s\n", normpath(outpdf))
    return plt
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_spatial_timing(; na = 80, T = 150)   # na/T only annotate the title
end
