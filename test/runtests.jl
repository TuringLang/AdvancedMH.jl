using Test
using AdvancedMH
using Random

@testset "AdvancedMH" begin
    # Define the components of a basic model.
    insupport(θ) = θ[2] >= 0
    dist(θ) = Normal(θ[1], θ[2])
    density(data, θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf

    # Generate a set of data from the posterior we want to estimate.
    Random.seed!(1)
    data = rand(Normal(0, 1), 30)

    # Construct a DensityModel.
    model = DensityModel(density, data)

    # Set up our sampler.
    spl = MetropolisHastings([0.0, 0.0], x -> MvNormal(x, 1.0))

    # Sample from the posterior.
    chain = sample(model, spl, 100000; param_names=["μ", "σ"])

    # chn_mean ≈ dist_mean atol=atol_v
    @test mean(chain["μ"].value) ≈ 0.0 atol=0.1
    @test mean(chain["σ"].value) ≈ 1.0 atol=0.1
end