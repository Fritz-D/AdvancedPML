# A unit test script for the categorical.jl module.
# To run:
# 1. Make sure categorical.jl is in the same directory.
# 2. Make sure Distributions.jl is installed.


using Test

include("categorical.jl")
using .Categorical

@testset "Categorical Distribution Tests" begin

    @testset "NormalizedCategorical Construction" begin
        η = [0.0, -0.5, -1.0]
        dist = NormalizedCategorical(η)
        @test dist.η ≈ η
        @test length(dist.η) == 3
    end

    @testset "NonNormalizedCategorical Construction" begin
        η = [1.0, 0.5, 0.0]
        log_norm = -0.5
        dist = NonNormalizedCategorical(η, log_norm)
        @test dist.η ≈ η
        @test dist.log_norm ≈ log_norm
    end

    @testset "CategoricalUniform" begin
        u = CategoricalUniform(4)
        @test u isa NormalizedCategorical
        @test is_uniform(u)
        @test length(u.η) == 4
        @test_throws ArgumentError CategoricalUniform(0)
        @test_throws ArgumentError CategoricalUniform(-1)
    end

    @testset "CategoricalFromProbabilities" begin
        probs = [0.5, 0.3, 0.2]
        dist = CategoricalFromProbabilities(probs)
        @test dist isa NormalizedCategorical
        @test means(dist) ≈ probs
        @test_throws ArgumentError CategoricalFromProbabilities([-0.1, 0.5, 0.6])
        @test_throws ArgumentError CategoricalFromProbabilities([0.5, 0.4, 0.4])
    end

    @testset "size()" begin
        d = NormalizedCategorical([0.0, -0.5])
        @test Categorical.size(d) == 2
        d = CategoricalUniform(10)
        @test Categorical.size(d) == 10
    end

    @testset "is_uniform()" begin
        @test is_uniform(NormalizedCategorical([0.0, 0.0, 0.0]))
        @test !is_uniform(NormalizedCategorical([0.0, -1.0, -2.0]))
        @test is_uniform(NormalizedCategorical([1.0]))  # single category
    end

    @testset "means() and variances()" begin
        d = NormalizedCategorical([0.0, -0.5])
        p = means(d)
        @test abs(sum(p) - 1.0) < 1e-10
        v = variances(d)
        @test all(v .>= 0)
        @test length(v) == length(p)
    end

    @testset "Multiplication (Normalized * Normalized)" begin
        d1 = NormalizedCategorical([0.0, -0.5, -1.0])
        d2 = NormalizedCategorical([-0.5, -1.6, -1.6])
        result = d1 * d2
        expected = (means(d1) .* means(d2)) ./ sum(means(d1) .* means(d2))
        @test means(result) ≈ expected
    end

    @testset "Multiplication (NonNormalized * Normalized)" begin
        n1 = NonNormalizedCategorical([1.0, 0.5, 0.0], -0.3)
        n2 = NormalizedCategorical([-0.916, -0.916, -1.609])
        result = n1 * n2
        @test result isa NonNormalizedCategorical
        @test length(result.η) == 3
    end

    @testset "Division (Normalized / Normalized)" begin
        d1 = NormalizedCategorical([0.0, -0.511, -2.303])
        d2 = NormalizedCategorical([0.0, -1.609, -6.908])
        result = d1 / d2
        @test result isa NormalizedCategorical
        @test length(result.η) == 3
    end

    @testset "KL Divergence" begin
        uniform = CategoricalUniform(3)
        skewed = NormalizedCategorical(log.([0.1, 0.8, 0.1]))
        kl1 = KL_divergence(skewed, uniform)
        kl2 = KL_divergence(uniform, skewed)
        @test kl1 > 0 && kl2 > 0
        dist = NormalizedCategorical(log.([0.3, 0.4, 0.3]))
        @test KL_divergence(dist, dist) ≈ 0 atol=1e-10
    end

    @testset "Conversion functions" begin
        n = NonNormalizedCategorical([0.5, 0.3, 0.2], -1.0)
        norm = NormalizedCategorical(n)
        @test norm isa NormalizedCategorical
        @test means(norm) ≈ means(n)

        norm2 = NormalizedCategorical([0.4, 0.3, 0.3])
        unnorm = NonNormalizedCategorical(norm2)
        @test unnorm isa NonNormalizedCategorical
        @test unnorm.log_norm ≈ 0.0
    end

    @testset "Distributions.jl interop" begin
        using Distributions
        d = NormalizedCategorical(log.([0.3, 0.5, 0.2]))
        std = distribution(d)
        @test std isa Distributions.Categorical
        @test std.p ≈ [0.3, 0.5, 0.2]
    end

    @testset "Edge Cases" begin
        d_empty = NormalizedCategorical(Float64[])
        @test is_uniform(d_empty)
        d_inf = NormalizedCategorical([-Inf, 0.0])
        @test all(isfinite.(means(d_inf)))
    end

end

