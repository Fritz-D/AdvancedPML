# Filename: test_dirichlet.jl
#
# This script contains unit tests for dirichlet.jl
# To run:
# 1. Make sure 'dirichlet.jl' is in the same directory.
# 2. Run `julia test_dirichlet.jl` from your terminal.

using Test
using SpecialFunctions  # Required for digamma, loggamma
using Distributions     # Required for distribution() test

# Include the library file to be tested
include("dirichlet.jl")
# Bring the library's functions into scope
using .Dirichlet

# --- Helper Function ---
# This helper is copied from the library's implementation
# to verify the log-normalization of non-normalized operations.
function log_Beta(α::Vector{Float64})
    return sum(loggamma.(α)) - loggamma(sum(α))
end

# --- Test Sets ---

@testset "Constructors and Types" begin
    # Standard NormalizedDirichlet
    d_norm = NormalizedDirichlet([2.0, 3.0, 1.0])
    @test d_norm.α == [2.0, 3.0, 1.0]
    @test d_norm isa Dirichlet.DirichletDist

    # Edge Case: NormalizedDirichlet must have positive parameters
    @test_throws ErrorException NormalizedDirichlet([1.0, 0.0, 2.0])
    @test_throws ErrorException NormalizedDirichlet([1.0, -1.0, 2.0])

    # Standard NonNormalizedDirichlet
    d_non_norm = NonNormalizedDirichlet([2.0, 3.0], 0.5)
    @test d_non_norm.α == [2.0, 3.0]
    @test d_non_norm.log_norm == 0.5
    @test d_non_norm isa Dirichlet.DirichletDist

    # Edge Case: NonNormalizedDirichlet allows non-negative (>= 0) parameters
    d_zero = NonNormalizedDirichlet([1.0, 0.0, 2.0], 0.1) # This is valid
    @test d_zero.α == [1.0, 0.0, 2.0]
    @test_throws ErrorException NonNormalizedDirichlet([-1.0, 2.0], 0.5)

    # DirichletUniform
    d_unif = DirichletUniform(3)
    @test d_unif.α == [1.0, 1.0, 1.0]
    @test d_unif isa NormalizedDirichlet

    # Conversion: NonNormalized -> Normalized
    d_norm_conv = NormalizedDirichlet(NonNormalizedDirichlet([2.0, 3.0], 0.5))
    @test d_norm_conv.α == [2.0, 3.0]
    @test d_norm_conv isa NormalizedDirichlet

    # Conversion: Normalized -> NonNormalized
    d_non_norm_conv = NonNormalizedDirichlet(NormalizedDirichlet([4.0, 5.0]))
    @test d_non_norm_conv.α == [4.0, 5.0]
    @test d_non_norm_conv.log_norm == 0.0
    @test d_non_norm_conv isa NonNormalizedDirichlet
end

# -----------------------------------------------------------------

@testset "Basic Functions (size, is_uniform)" begin
    d1 = NormalizedDirichlet([1.0, 2.0, 3.0])
    d2 = NonNormalizedDirichlet([1.0, 2.0], 0.5)
    d_unif_norm = DirichletUniform(3)
    d_unif_non = NonNormalizedDirichlet([1.0, 1.0], 0.1)

    # size() - Must be prefixed with Dirichlet. to resolve conflict with Base.size
    @test Dirichlet.size(d1) == 3
    @test Dirichlet.size(d2) == 2
    @test Dirichlet.size(d_unif_norm) == 3

    # is_uniform()
    @test is_uniform(d_unif_norm) == true
    @test is_uniform(d_unif_non) == true
    @test is_uniform(NormalizedDirichlet([1.0, 1.0])) == true
    @test is_uniform(d1) == false
    @test is_uniform(d2) == false
end

# -----------------------------------------------------------------

@testset "Statistical Functions" begin
    d_norm = NormalizedDirichlet([2.0, 3.0, 5.0]) # sum(α) = 10
    d_non_norm = NonNormalizedDirichlet([2.0, 3.0, 5.0], 0.5) # Same params
    d_unif = DirichletUniform(3) # sum(α) = 3

    # means()
    @test means(d_norm) ≈ [0.2, 0.3, 0.5]
    @test means(d_non_norm) ≈ [0.2, 0.3, 0.5] # Non-norm should not affect mean
    @test means(d_unif) ≈ [1/3, 1/3, 1/3]

    # variances()
    α0_norm = 10.0
    denom_norm = α0_norm^2 * (α0_norm + 1) # 100 * 11 = 1100
    @test variances(d_norm) ≈ [(2*(10-2))/denom_norm, (3*(10-3))/denom_norm, (5*(10-5))/denom_norm]
    @test variances(d_norm) ≈ [16/1100, 21/1100, 25/1100]
    @test variances(d_non_norm) ≈ variances(d_norm) # Non-norm should not affect variance

    α0_unif = 3.0
    denom_unif = α0_unif^2 * (α0_unif + 1) # 9 * 4 = 36
    @test variances(d_unif) ≈ [2/36, 2/36, 2/36]

    # expected_logs()
    c_norm = digamma(sum(d_norm.α))
    @test expected_logs(d_norm) ≈ digamma.(d_norm.α) .- c_norm
    @test expected_logs(d_non_norm) ≈ expected_logs(d_norm) # Non-norm should not affect

    c_unif = digamma(sum(d_unif.α))
    @test expected_logs(d_unif) ≈ digamma.([1.0, 1.0, 1.0]) .- c_unif
end

# -----------------------------------------------------------------

@testset "Operators (* and /)" begin
    # --- Normalized ---
    d1 = NormalizedDirichlet([3.0, 4.0, 2.0])
    d2 = NormalizedDirichlet([1.0, 2.0, 1.0])
    d_unif = DirichletUniform(3)
    d_dim2 = NormalizedDirichlet([1.0, 1.0])

    # Normalized * Normalized
    d_prod = d1 * d2
    @test d_prod.α == [3.0+1.0-1.0, 4.0+2.0-1.0, 2.0+1.0-1.0]
    @test d_prod.α == [3.0, 5.0, 2.0]
    
    # Edge Case: Multiply by uniform
    @test (d1 * d_unif).α == d1.α
    
    # Edge Case: Dimension mismatch
    @test_throws ErrorException d1 * d_dim2

    # Edge Case: Invalid result (α <= 0)
    @test_throws ErrorException NormalizedDirichlet([0.5, 1.0]) * NormalizedDirichlet([0.2, 1.0])

    # Normalized / Normalized
    d_div = d1 / d2
    @test d_div.α == [3.0-1.0+1.0, 4.0-2.0+1.0, 2.0-1.0+1.0]
    @test d_div.α == [3.0, 3.0, 2.0]
    
    # Edge Case: Divide by uniform
    @test (d1 / d_unif).α == d1.α

    # Edge Case: Dimension mismatch
    @test_throws ErrorException d1 / d_dim2

    # Edge Case: Invalid result (α <= 0)
    @test_throws ErrorException NormalizedDirichlet([1.0, 2.0]) / NormalizedDirichlet([2.0, 1.0])

    # Round-trip property
    @test ((d1 * d2) / d2).α ≈ d1.α
    @test ((d1 / d2) * d2).α ≈ d1.α

    # --- NonNormalized ---
    d_non = NonNormalizedDirichlet([3.0, 4.0], 0.5)
    d_norm = NormalizedDirichlet([2.0, 1.0])

    # NonNormalized * Normalized
    d_prod_non = d_non * d_norm
    α_prod = [3.0+2.0-1.0, 4.0+1.0-1.0] # [4.0, 4.0]
    @test d_prod_non.α == α_prod
    log_norm_Δ_prod = log_Beta(α_prod) - log_Beta(d_non.α) - log_Beta(d_norm.α)
    @test d_prod_non.log_norm ≈ 0.5 + log_norm_Δ_prod

    # Normalized * NonNormalized (Commutative check)
    d_prod_non_2 = d_norm * d_non
    @test d_prod_non_2.α == d_prod_non.α
    @test d_prod_non_2.log_norm ≈ d_prod_non.log_norm

    # NonNormalized / Normalized
    d_div_non = d_non / d_norm
    α_div = [3.0-2.0+1.0, 4.0-1.0+1.0] # [2.0, 4.0]
    @test d_div_non.α == α_div
    log_norm_Δ_div = log_Beta(α_div) - log_Beta(d_non.α) + log_Beta(d_norm.α)
    @test d_div_non.log_norm ≈ 0.5 + log_norm_Δ_div
end

# -----------------------------------------------------------------

@testset "KL Divergence and Utilities" begin
    d1 = NormalizedDirichlet([2.0, 3.0])
    d2 = NormalizedDirichlet([1.0, 5.0])
    d3 = NormalizedDirichlet([2.0, 3.0]) # Same as d1
    d_dim3 = NormalizedDirichlet([1.0, 1.0, 1.0])

    d_non1 = NonNormalizedDirichlet([2.0, 3.0], 0.1)
    d_non2 = NonNormalizedDirichlet([1.0, 5.0], 0.2)
    d_non3 = NonNormalizedDirichlet([2.0, 3.0], 0.3) # Same α as d_non1

    # KL_divergence()
    @test KL_divergence(d1, d3) ≈ 0.0 # KL(d1 || d1) == 0
    @test KL_divergence(d1, d2) > 0.0  # KL(d1 || d2) > 0
    
    # Manual check of formula
    kl_val = log_Beta(d2.α) - log_Beta(d1.α) + sum((d1.α .- d2.α) .* (digamma.(d1.α) .- digamma(sum(d1.α))))
    @test KL_divergence(d1, d2) ≈ kl_val

    # KL for NonNormalized (log_norm should be ignored)
    @test KL_divergence(d_non1, d_non3) ≈ 0.0 # Same α
    @test KL_divergence(d_non1, d_non2) ≈ kl_val # Same α's as d1, d2

    # Edge Case: Dimension mismatch
    @test_throws ErrorException KL_divergence(d1, d_dim3)

    # Edge Case: Type mismatch (as per docstring)
    @test_throws MethodError KL_divergence(d1, d_non1)

    # distribution()
    dist_norm = distribution(d1)
    @test dist_norm isa Distributions.Dirichlet
    @test dist_norm.alpha == d1.α

    dist_non_norm = distribution(d_non1)
    @test dist_non_norm isa Distributions.Dirichlet
    @test dist_non_norm.alpha == d_non1.α # Ignores log_norm

    # show()
    @test sprint(show, d1) == "α = [2.0, 3.0]"
    @test sprint(show, DirichletUniform(2)) == "uniform"
    @test sprint(show, d_non1) == "α = [2.0, 3.0], Z = $(exp(0.1))"
    @test sprint(show, NonNormalizedDirichlet([1.0, 1.0], 0.2)) == "uniform (Z = $(exp(0.2)))"
end
