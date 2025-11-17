# default values
N, μ1, μ2, σ1, σ2, β = 10, 4, 4, 2, 2, 3

# (a)
f1(s1)          = (2*π*σ1^2)^-0.5   * exp(-(s1-μ1)^2/(2σ1^2))
f2(s1, p1)      = (2*π*β^2)^-0.5    * exp(-(p1-s1)^2/(2β^2))
f3(s2)          = (2*π*σ2^2)^-0.5   * exp(-(s2-μ2)^2/(2σ2^2))
f4(s2, p2)      = (2*π*β^2)^-0.5    * exp(-(p2-s2)^2/(2β^2))
f5(p1, p2, d)   = Int(d == p1 - p2)
f6(d)           = Int(d > 0)

p(s1,s2,p1,p2,d) = f1(s1) * f2(s1, p1) * f3(s2) * f4(s2, p2) * f5(p1, p2, d) * f6(d)
p1(s1, N=20) = sum(p(s1,s2,p1,p2,d) for (s2,p1,p2,d)  in Iterators.product(1:N, 1:N, 1:N, -N:N))
p2(s2, N=20) = sum(p(s1,s2,p1,p2,d) for (s1,p1,p2,d)  in Iterators.product(1:N, 1:N, 1:N, -N:N))
p3(p1, N=20) = sum(p(s1,s2,p1,p2,d) for (s1,s2,p2,d)  in Iterators.product(1:N, 1:N, 1:N, -N:N))
p4(p2, N=20) = sum(p(s1,s2,p1,p2,d) for (s1,s2,p1,d)  in Iterators.product(1:N, 1:N, 1:N, -N:N))
p5(d,  N=20) = sum(p(s1,s2,p1,p2,d) for (s1,s2,p1,p2) in Iterators.product(1:N, 1:N, 1:N,  1:N))

# (b)
m_f1s1(s1) = f1(s1)
m_f3s2(s2) = f3(s2)


# (c)
N, μ1, μ2, σ1, σ2, β = 20, 8, 12, 2, 2, 3

println(sum(p5(d) for d in -N:0), ", ", sum(p5(d) for d in 1:N))
