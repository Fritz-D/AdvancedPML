# A unit test script for the gamma.jl module.
# To run:
# 1. Make sure gamma.jl is in the same directory.
# 2. Make sure necessary packages are installed: Distributions,jl, SpecialFunctions.jl

using Test
using Distributions
using SpecialFunctions

include("gamma.jl")

import .Gamma

@testset "Gamma Module Tests" begin

    g1 = Gamma.Gamma1D(1.0, 2.0)
    g2 = Gamma.Gamma1D(2.0, 3.0)
    g_uni = Gamma.Gamma1DUniform()
    nn1 = Gamma.NonNormalizedGamma1D(1.0, 2.0, 0.5)

    @testset "1. Constructors and Properties" begin
        # Test basic constructor
        @test g1.β == 1.0
        @test g1.λ == 2.0
        @test g1 isa Gamma.Gamma1D

        # Test uniform constructor
        @test g_uni.β == 0.0
        @test g_uni.λ == 0.0
        @test Gamma.is_uniform(g_uni) == true
        @test Gamma.is_uniform(g1) == false

        # Test constructor error conditions
        @test_throws ErrorException begin Gamma.Gamma1D(-1.1, 1.0) end # Negative beta exponent
        # Beta MAY be -1.0, see note. test has been adjusted
        @test_throws ErrorException begin Gamma.Gamma1D(1.0, -0.1) end # Negative lambda rate
    end

    @testset "2. Conversions" begin
        # Test NonNormalized -> Normalized
        g_conv_norm = Gamma.Gamma1D(nn1)
        @test g_conv_norm.β == nn1.β
        @test g_conv_norm.λ == nn1.λ
        @test g_conv_norm isa Gamma.Gamma1D

        # Test Normalized -> NonNormalized
        g_conv_nn = Gamma.NonNormalizedGamma1D(g1)
        @test g_conv_nn.β == g1.β
        @test g_conv_nn.λ == g1.λ
        @test g_conv_nn.log_norm == 0.0
        @test g_conv_nn isa Gamma.NonNormalizedGamma1D
    end

    @testset "3. Statistics" begin
        @test Gamma.mean(g1) ≈ 1.0
        @test Gamma.variance(g1) ≈ 0.5

        @test Gamma.mean(g2) ≈ 1.0
        @test Gamma.variance(g2) ≈ 1/3

        expected_log_g1 = digamma(g1.β + 1) - log(g1.λ)
        @test Gamma.expected_log(g1) ≈ expected_log_g1

        # Test non-normalized (should be identical)
        @test Gamma.mean(nn1) ≈ Gamma.mean(g1)
        @test Gamma.variance(nn1) ≈ Gamma.variance(g1)
        @test Gamma.expected_log(nn1) ≈ expected_log_g1

        # Test uniform edge case
        @test Gamma.mean(g_uni) == Inf
        @test Gamma.variance(g_uni) == Inf
    end

    @testset "4. Multiplication (*)" begin
        prod_g = g1 * g2
        @test prod_g.β ≈ g1.β + g2.β
        @test prod_g.λ ≈ g1.λ + g2.λ
        
        prod_uni = g1 * g_uni
        @test prod_uni.β == g1.β
        @test prod_uni.λ == g1.λ
        @test prod_uni isa Gamma.Gamma1D
        
        nn_ex = Gamma.NonNormalizedGamma1D(1.0, 1.0, 5.0) # β=1, λ=1, Z=e^5
        g_ex = Gamma.Gamma1D(0.5, 2.0)                  # β=0.5, λ=2
        prod_nn = nn_ex * g_ex
        
        @test prod_nn.β ≈ 1.5
        @test prod_nn.λ ≈ 3.0
    end

    @testset "5. Division (/)" begin
        div_g = g2 / g1
        @test div_g.β ≈ g2.β - g1.β # Expected: 1.0
        @test div_g.λ ≈ g2.λ - g1.λ # Expected: 1.0

        # Test uniform division (uniform / uniform)
        @test Gamma.is_uniform(g_uni / g_uni)

        # Test NonNormalized / Normalized
        nn_num = Gamma.NonNormalizedGamma1D(3.0, 5.0, 2.0) # β=3, λ=5, Z=e^2
        g_den_div = Gamma.Gamma1D(1.0, 2.0)                # β=1, λ=2
        div_nn = nn_num / g_den_div
        
        @test div_nn.β ≈ 2.0
        @test div_nn.λ ≈ 3.0
        
        # Test round-trip property for NonNormalized
        g_ex = Gamma.Gamma1D(0.8, 1.5)
        nn_ex = Gamma.NonNormalizedGamma1D(1.2, 2.3, 4.5)
        nn_round_trip = (nn_ex * g_ex) / g_ex
        @test nn_round_trip.β ≈ nn_ex.β
        @test nn_round_trip.λ ≈ nn_ex.λ
        @test nn_round_trip.log_norm ≈ nn_ex.log_norm

        # Test division error cases (negative result parameters)
        @test_throws ErrorException begin g1 / g2 end # β1 < β2
        @test_throws ErrorException begin g2 / Gamma.Gamma1D(1.0, 5.0) end # λ2 < λ_den
        @test_throws ErrorException begin g_uni / g1 end # Uniform cannot be divided
    end

    @testset "6. KL Divergence" begin
        # KL(g, g) = 0
        @test Gamma.KL_divergence(g1, g1) ≈ 0.0

        # Test standard case KL(g1, g2)
        kl_val = (g1.β - g2.β) * digamma(g1.β + 1) - loggamma(g1.β + 1) + loggamma(g2.β + 1) + 
                 (g2.β + 1) * (log(g1.λ) - log(g2.λ)) + (g1.β + 1) * (g2.λ - g1.λ) / g1.λ
        
        @test Gamma.KL_divergence(g1, g2) ≈ kl_val

        # Test with non-normalized (should be identical)
        nn_g1 = Gamma.NonNormalizedGamma1D(g1)
        nn_g2 = Gamma.NonNormalizedGamma1D(g2)
        @test Gamma.KL_divergence(nn_g1, nn_g2) ≈ kl_val

        @test Gamma.KL_divergence(g1, g_uni) ≈ 0.0 
        @test Gamma.KL_divergence(g_uni, g1) ≈ 0.0 
        @test Gamma.KL_divergence(g_uni, g_uni) ≈ 0.0
    end

    @testset "7. Distribution Utility" begin
        d = Gamma.distribution(g1)

        @test d isa Distributions.Gamma
        @test Distributions.shape(d) ≈ g1.β + 1.0 # The shape parameter α is β + 1
        @test Distributions.rate(d) ≈ g1.λ      # The rate parameter θ is λ
        @test Distributions.mean(d) ≈ Gamma.mean(g1)

        d_nn = Gamma.distribution(nn1)
        @test d_nn isa Distributions.Gamma
        @test Distributions.shape(d_nn) ≈ nn1.β + 1.0

        @test_throws DomainError begin Gamma.distribution(g_uni) end
    end

    @testset "8. Randomized (Fuzz) Tests" begin
        for i in 1:100
            # Generate random valid parameters
            β1 = 5.0 * rand() + 0.1
            λ1 = 5.0 * rand() + 0.1
            
            # To test division g2 / g1, we need β2 > β1 and λ2 > λ1
            β2 = β1 + 5.0 * rand() + 0.1
            λ2 = λ1 + 5.0 * rand() + 0.1
            
            g_rand1 = Gamma.Gamma1D(β1, λ1)
            g_rand2 = Gamma.Gamma1D(β2, λ2)

            # Test mean and variance
            expected_mean1 = (β1 + 1.0) / λ1
            expected_variance1 = (β1 + 1.0) / (λ1 * λ1)

            @test Gamma.mean(g_rand1) ≈ expected_mean1
            @test Gamma.variance(g_rand1) ≈ expected_variance1

            # Test expected_log
            expected_log1 = digamma(β1 + 1.0) - log(λ1)
            @test Gamma.expected_log(g_rand1) ≈ expected_log1

            # Test multiplication
            prod_g = g_rand1 * g_rand2
            @test prod_g.β ≈ β1 + β2
            @test prod_g.λ ≈ λ1 + λ2
            @test Gamma.mean(prod_g) ≈ (β1 + β2 + 1.0) / (λ1 + λ2)

            # Test division (g2 / g1)
            div_g = g_rand2 / g_rand1
            @test div_g.β ≈ β2 - β1
            @test div_g.λ ≈ λ2 - λ1
            @test div_g.β >= 0.0
            @test div_g.λ >= 0.0
            @test Gamma.mean(div_g) ≈ (β2 - β1 + 1.0) / (λ2 - λ1)

            # Test KL Divergence
            kl_val = (β1 - β2) * digamma(β1 + 1) - loggamma(β1 + 1) + loggamma(β2 + 1) + 
                     (β2 + 1) * (log(λ1) - log(λ2)) + (β1 + 1) * (λ2 - λ1) / λ1
            @test Gamma.KL_divergence(g_rand1, g_rand2) ≈ kl_val

            # Test NonNormalized Round-trip
            log_norm = 10 * (rand() - 0.5)
            nn_rand1 = Gamma.NonNormalizedGamma1D(g_rand1.β, g_rand1.λ, log_norm)
            
            # (nn_rand1 * g_rand2) / g_rand2 should be nn_rand1
            nn_round_trip = (nn_rand1 * g_rand2) / g_rand2
            @test nn_round_trip.β ≈ nn_rand1.β
            @test nn_round_trip.λ ≈ nn_rand1.λ
            @test nn_round_trip.log_norm ≈ nn_rand1.log_norm
        end
    end
end

