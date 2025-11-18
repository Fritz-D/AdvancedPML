using BenchmarkTools

# priors
N, μ1, μ2, σ1, σ2, β = 20, 8, 12, 2, 2, 3

# factors
f1(s1)          = (2*π*σ1^2)^-0.5   * exp(-(s1-μ1)^2/(2*σ1^2))
f2(s1, p1)      = (2*π*β^2)^-0.5    * exp(-(p1-s1)^2/(2*β^2))
f3(s2)          = (2*π*σ2^2)^-0.5   * exp(-(s2-μ2)^2/(2*σ2^2))
f4(s2, p2)      = (2*π*β^2)^-0.5    * exp(-(p2-s2)^2/(2*β^2))
f5(p1, p2, d)   = Int(d == p1 - p2)
f6(d)           = Int(d > 0)

# (a) marginals
pm(s1,s2,p1,p2,d) = f1(s1) * f2(s1, p1) * f3(s2) * f4(s2, p2) * f5(p1, p2, d) * f6(d)
pm_s1(s1) = sum(pm(s1,s2,p1,p2,d) for (s2,p1,p2,d)  in Iterators.product(1:N, 1:N, 1:N, -N:N))
pm_s2(s2) = sum(pm(s1,s2,p1,p2,d) for (s1,p1,p2,d)  in Iterators.product(1:N, 1:N, 1:N, -N:N))
pm_p1(p1) = sum(pm(s1,s2,p1,p2,d) for (s1,s2,p2,d)  in Iterators.product(1:N, 1:N, 1:N, -N:N))
pm_p2(p2) = sum(pm(s1,s2,p1,p2,d) for (s1,s2,p1,d)  in Iterators.product(1:N, 1:N, 1:N, -N:N))
pm_d(d)   = sum(pm(s1,s2,p1,p2,d) for (s1,s2,p1,p2) in Iterators.product(1:N, 1:N, 1:N,  1:N))

# (b) messages
m_f1s1(s1) = f1(s1)
m_f3s2(s2) = f3(s2)
p_s1(s1)   = m_f1s1(s1)
p_s2(s2)   = m_f3s2(s2)
m_f2p1(p1) = sum(f2(s1,p1) * p_s1(s1) for s1 in 1:N)
m_f4p2(p2) = sum(f4(s2,p2) * p_s2(s2) for s2 in 1:N)
p_p1(p1)   = m_f2p1(p1)
p_p2(p2)   = m_f4p2(p2)
m_p1f5(p1) = p_p1(p1)
m_p2f5(p2) = p_p2(p2)
m_f5d(d)   = sum(f5(p1, p2, d) * m_p1f5(p1) * m_p2f5(p2) for (p1, p2) in Iterators.product(1:N, 1:N))
p_d(d)     = m_f5d(d)
# setting y = 1
m_f6d(d)   = f6(d)
p_d(d)     = m_f5d(d) * m_f6(d)
m_df5(d)   = p_d(d) / m_f5d(d) * Int(m_f5d(d) > 0)
m_f5p1(p1) = sum(f5(p1,p2,d) * m_df5(d) * m_p2f5(p2) for (p2, d) in Iterators.product(1:N, -N:N))
m_f5p2(p2) = sum(f5(p1,p2,d) * m_df5(d) * m_p1f5(p1) for (p1, d) in Iterators.product(1:N, -N:N))
p_p1(p1)   = m_f2p1(p1) * m_f5p1(p1)
p_p2(p2)   = m_f4p2(p2) * m_f5p2(p2)
m_f2s1(s1) = sum(f2(s1,p1) * p_p1(p1) / m_f2p1(p1) for p1 in 1:N)
m_f4s2(s2) = sum(f4(s2,p2) * p_p2(p2) / m_f4p2(p2) for p2 in 1:N)
p_s1(s1)   = m_f1s1(s1) * m_f2s1(s1)
p_s2(s2)   = m_f3s2(s2) * m_f4s2(s2)

# test
function solveWithMarginals()
    return [
        [pm_s1(s1) for s1 in 1:N], 
        [pm_s2(s2) for s2 in 1:N], 
        [pm_p1(p1) for p1 in 1:N], 
        [pm_p2(p2) for p2 in 1:N], 
        [pm_d(d)   for d in -N:N]
    ]
end

function solveWithMessages()
    return [
        [p_s1(s1) for s1 in 1:N], 
        [p_s2(s2) for s2 in 1:N], 
        [p_p1(p1) for p1 in 1:N], 
        [p_p2(p2) for p2 in 1:N], 
        [p_d(d)   for d in -N:N]
    ]
end

@btime solveWithMarginals()
@btime solveWithMessages()
println(solveWithMarginals())
println(solveWithMessages())