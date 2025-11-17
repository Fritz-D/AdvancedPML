# A unit test script for the gaussian.jl module.
# To run:
# 1. Make sure gaussian.jl is in the same directory.
# 2. Make sure Distributions.jl is installed.

using Test
using Distributions

include("gaussian.jl")

import .Gaussian

@testset "Gaussian Module Tests" begin

    # Define common distributions for testing
    g1 = Gaussian.Gaussian1D(2.0, 0.5)
    g2 = Gaussian.Gaussian1D(1.4, 3.5)
    g_uni = Gaussian.Gaussian1DUniform()
    nn1 = Gaussian.NonNormalizedGaussian1D(2.0, 0.5, 1.0)
    nn_uni = Gaussian.NonNormalizedGaussian1D(0.0, 0.0, 1.0)

    @testset "1. Constructors and Properties" begin
        @test g1.τ == 2.0
        @test g1.ρ == 0.5
        g_mv = Gaussian.Gaussian1DFromMeanVariance(4.0, 2.0)
        @test g_mv.τ ≈ 2.0
        @test g_mv.ρ ≈ 0.5
        g_std = Gaussian.Gaussian1D()
        @test g_std.τ == 0.0
        @test g_std.ρ == 1.0
        @test g_uni.τ == 0.0
        @test g_uni.ρ == 0.0
        @test Gaussian.is_uniform(g_uni) == true
        @test Gaussian.is_uniform(g1) == false
        @test Gaussian.is_uniform(nn_uni) == true
        @test Gaussian.is_uniform(nn1) == false
        @test_throws ErrorException begin Gaussian.Gaussian1D(1.0, -0.1) end
        @test_throws ErrorException begin Gaussian.NonNormalizedGaussian1D(1.0, -0.1, 0.0) end
        @test_throws ErrorException begin Gaussian.Gaussian1DFromMeanVariance(1.0, 0.0) end
    end

    @testset "2. Conversions" begin
        g_conv_norm = Gaussian.Gaussian1D(nn1)
        @test g_conv_norm.τ == 2.0
        @test g_conv_norm.ρ == 0.5
        @test g_conv_norm isa Gaussian.Gaussian1D
        g_conv_nn = Gaussian.NonNormalizedGaussian1D(g1)
        @test g_conv_nn.τ == 2.0
        @test g_conv_nn.ρ == 0.5
        @test g_conv_nn.log_norm == 0.0
        @test g_conv_nn isa Gaussian.NonNormalizedGaussian1D
    end

    @testset "3. Statistics (mean, variance)" begin
        @test Gaussian.mean(g1) ≈ 4.0
        @test Gaussian.variance(g1) ≈ 2.0
        @test Gaussian.mean(nn1) ≈ 4.0
        @test Gaussian.variance(nn1) ≈ 2.0
        @test Gaussian.mean(g_uni) == 0.0
        @test Gaussian.variance(g_uni) == +Inf
    end

    @testset "4. Multiplication (*)" begin
        prod_g = g1 * g2
        @test prod_g.τ ≈ 3.4
        @test prod_g.ρ ≈ 4.0
        prod_uni = g1 * g_uni
        @test prod_uni.τ == g1.τ
        @test prod_uni.ρ == g1.ρ
        @test Gaussian.is_uniform(g_uni * g_uni)
        nn_ex = Gaussian.NonNormalizedGaussian1D(1.0, 1.0, 2.0)
        g_ex = Gaussian.Gaussian1D(2.0, 3.0)
        prod_nn = nn_ex * g_ex
        @test prod_nn.τ ≈ 3.0
        @test prod_nn.ρ ≈ 4.0
        log_norm_add = -0.5 * (log(2.0 * π * (1.0 + 1.0/3.0)) + (1.0 - 2.0/3.0)^2 / (1.0 + 1.0/3.0))
        @test prod_nn.log_norm ≈ 2.0 + log_norm_add
        prod_nn_comm = g_ex * nn_ex
        @test prod_nn_comm.τ == prod_nn.τ
        @test prod_nn_comm.ρ == prod_nn.ρ
        @test prod_nn_comm.log_norm == prod_nn.log_norm
        prod_nn_uni = nn_ex * g_uni
        @test prod_nn_uni.τ == nn_ex.τ
        @test prod_nn_uni.ρ == nn_ex.ρ
        @test prod_nn_uni.log_norm == nn_ex.log_norm
    end

    @testset "5. Division (/)" begin
        g_num = Gaussian.Gaussian1D(7.0, 8.0)
        g_den = Gaussian.Gaussian1D(4.0, 3.0)
        div_g = g_num / g_den
        @test div_g.τ ≈ 3.0
        @test div_g.ρ ≈ 5.0
        @test Gaussian.is_uniform(g_uni / g_uni)
        @test_throws ErrorException begin g_uni / g_den end
        @test_throws ErrorException begin g_den / g_num end
        nn_num = Gaussian.NonNormalizedGaussian1D(2.0, 5.0, 1.0)
        g_den_div = Gaussian.Gaussian1D(1.0, 1.0)
        div_nn = nn_num / g_den_div
        @test div_nn.τ ≈ 1.0
        @test div_nn.ρ ≈ 4.0
        μ_diff = 0.4 - 1.0
        σ2_diff = 1.0 - 0.2
        log_norm_add = log(1.0) + 0.5 * (log(2 * π / σ2_diff) + μ_diff^2 / σ2_diff)
        @test div_nn.log_norm ≈ 1.0 + log_norm_add
        div_nn_uni = nn_num / g_uni
        @test div_nn_uni.τ == nn_num.τ
        @test div_nn_uni.ρ == nn_num.ρ
        @test div_nn_uni.log_norm == nn_num.log_norm
        g_round_trip = (nn1 * g_den_div) / g_den_div
        @test g_round_trip.τ ≈ nn1.τ
        @test g_round_trip.ρ ≈ nn1.ρ
        @test g_round_trip.log_norm ≈ nn1.log_norm
    end

    @testset "6. KL Divergence" begin
        h1 = Gaussian.Gaussian1D(1.0, 1.0)
        h2 = Gaussian.Gaussian1D(2.0, 3.0)
        @test Gaussian.KL_divergence(h1, h2) ≈ 0.617360522332612
        @test Gaussian.KL_divergence(h2, h1) ≈ 0.2715283665562771
        @test Gaussian.KL_divergence(h1, h1) == 0.0
        nn_h1 = Gaussian.NonNormalizedGaussian1D(h1)
        nn_h2 = Gaussian.NonNormalizedGaussian1D(h2)
        @test Gaussian.KL_divergence(nn_h1, nn_h2) ≈ 0.617360522332612
        @test Gaussian.KL_divergence(h1, g_uni) == Inf
        @test Gaussian.KL_divergence(g_uni, h1) == Inf
        @test Gaussian.KL_divergence(g_uni, g_uni) == 0.0
    end

    @testset "7. Distribution Utility" begin
        g_dist = Gaussian.Gaussian1DFromMeanVariance(2.0, 3.0)
        d = Gaussian.distribution(g_dist)
        @test d isa Distributions.Normal
        @test Distributions.mean(d) ≈ 2.0
        @test Distributions.var(d) ≈ 3.0
        @test Distributions.std(d) ≈ sqrt(3.0)
        d_nn = Gaussian.distribution(Gaussian.NonNormalizedGaussian1D(g_dist))
        @test d_nn isa Distributions.Normal
        @test Distributions.mean(d_nn) ≈ 2.0
        @test_throws DomainError begin Gaussian.distribution(g_uni) end
    end

    @testset "8. Randomized Tests" begin
        for _ in 1:100
            μ1 = 10 * (rand() - 0.5)
            μ2 = 10 * (rand() - 0.5)

            # To test division g2 / g1, we must ensure ρ2 >= ρ1.
            # This means σ2_2 <= σ1_2.
            # Let's generate σ2_2 first, then add to it for σ1_2.
            σ2_2 = 5 * rand() + 0.1
            σ1_2 = σ2_2 + 5 * rand() + 0.1 
            
            g1 = Gaussian.Gaussian1DFromMeanVariance(μ1, σ1_2)
            g2 = Gaussian.Gaussian1DFromMeanVariance(μ2, σ2_2)

            # Test mean and variance
            @test Gaussian.mean(g1) ≈ μ1
            @test Gaussian.variance(g1) ≈ σ1_2
            @test Gaussian.mean(g2) ≈ μ2
            @test Gaussian.variance(g2) ≈ σ2_2

            # Test multiplication
            prod_g = g1 * g2
            ρ1 = 1.0 / σ1_2
            ρ2 = 1.0 / σ2_2
            ρ_p = ρ1 + ρ2
            τ1 = μ1 * ρ1
            τ2 = μ2 * ρ2
            τ_p = τ1 + τ2
            
            @test prod_g.τ ≈ τ_p
            @test prod_g.ρ ≈ ρ_p
            @test Gaussian.mean(prod_g) ≈ τ_p / ρ_p
            @test Gaussian.variance(prod_g) ≈ 1.0 / ρ_p

            # Test division (g2 / g1)
            div_g = g2 / g1
            ρ_d = ρ2 - ρ1
            τ_d = τ2 - τ1

            @test div_g.τ ≈ τ_d
            @test div_g.ρ ≈ ρ_d
            @test ρ_d >= 0.0 # Explicitly test our assumption
            @test Gaussian.mean(div_g) ≈ τ_d / ρ_d
            @test Gaussian.variance(div_g) ≈ 1.0 / ρ_d

            # Test KL Divergence
            kl_val = 0.5 * (log(σ2_2 / σ1_2) + (σ1_2 + (μ1 - μ2)^2) / σ2_2 - 1)
            @test Gaussian.KL_divergence(g1, g2) ≈ kl_val

            # Test NonNormalized Round-trip
            log_norm = 5 * (rand() - 0.5)
            nn1 = Gaussian.NonNormalizedGaussian1D(g1.τ, g1.ρ, log_norm)
            
            # (nn1 * g2) / g2 should be nn1
            nn_round_trip = (nn1 * g2) / g2
            @test nn_round_trip.τ ≈ nn1.τ
            @test nn_round_trip.ρ ≈ nn1.ρ
            @test nn_round_trip.log_norm ≈ nn1.log_norm
        end
    end

end
