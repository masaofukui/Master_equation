include("huggett_ssj.jl")   # pulls in huggett_fame.jl too (guarded)

p  = Huggett()
PARAMS[] = p
ss = solve_steady(p)
n  = p.na * p.ny

agrid = ss.agrid
i_lo  = idx(searchsortedfirst(agrid, 0.0), 1, p)
i_hi  = idx(searchsortedfirst(agrid, 5.0), 1, p)
h0 = zeros(n); h0[i_hi] += 1e-3; h0[i_lo] -= 1e-3

# warmup (compile)
solve_fame(ss; verbose=false)
ssj_irf(ss, h0; T=300)

println("n = ", n, ",  SSJ horizon T = 300")
println("\n--- FAME: solve_fame(ss) ---")
display(@benchmark solve_fame($ss; verbose=false))
println("\n\n--- SSJ: ssj_irf(ss,h0;T=300) ---")
display(@benchmark ssj_irf($ss, $h0; T=300))

println("\n\n--- SSJ pieces ---")
print("  fake_news:    "); display(@benchmark ssj_fake_news($ss; T=300)); println()
print("  expectations: "); display(@benchmark ssj_expectations($ss; T=300)); println()
cy, cd = ssj_fake_news(ss; T=300); E = ssj_expectations(ss; T=300)
print("  jacobian:     "); display(@benchmark ssj_jacobian($cy,$cd,$E; T=300)); println()
