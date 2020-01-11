using Test
using AdvancedMH
using Random
using Distributions
using Random

@testset "AdvancedMH" begin
    # Set a seed
    Random.seed!(1234)

    # Generate a set of data from the posterior we want to estimate.
    data = rand(Normal(0, 1), 300)

    # Define the components of a basic model.
    insupport(θ) = θ[2] >= 0
    dist(θ) = Normal(θ[1], θ[2])
    density(θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf

    # Construct a DensityModel.
    model = DensityModel(density)

    # Set up our sampler with initial parameters.
    spl1 = RWMH([0.0, 0.0])
    spl2 = StaticMH([0.0, 0.0], MvNormal([0.0, 0.0], 1))

    @testset "Inference" begin

        # Sample from the posterior.
        chain1 = sample(model, spl1, 100000; param_names=["μ", "σ"])
        chain2 = sample(model, spl2, 100000; param_names=["μ", "σ"])

        # chn_mean ≈ dist_mean atol=atol_v
        @test mean(chain1["μ"].value) ≈ 0.0 atol=0.1
        @test mean(chain1["σ"].value) ≈ 1.0 atol=0.1
        @test mean(chain2["μ"].value) ≈ 0.0 atol=0.1
        @test mean(chain2["σ"].value) ≈ 1.0 atol=0.1
    end

    @testset "psample" begin
        chain1 = psample(model, spl1, 10000, 4)
        @test mean(chain1["μ"].value) ≈ 0.0 atol=0.1
        @test mean(chain1["σ"].value) ≈ 1.0 atol=0.1
    end
end

