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
    spl = MetropolisHastings([0.0, 0.0])

    # Sample from the posterior.
    chain = sample(model, spl, 100000; param_names=["μ", "σ"])

    # chn_mean ≈ dist_mean atol=atol_v
    @test mean(chain["μ"].value) ≈ 0.0 atol=0.1
    @test mean(chain["σ"].value) ≈ 1.0 atol=0.1
end
