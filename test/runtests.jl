using AdvancedMH
using Distributions
using StructArrays
using MCMCChains

using Random
using Test
using DiffResults
using ForwardDiff

include("util.jl")

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

    @testset "StaticMH" begin
        # Set up our sampler with initial parameters.
        spl1 = StaticMH([Normal(0,1), Normal(0, 1)])
        spl2 = StaticMH(MvNormal([0.0, 0.0], 1))

        # Sample from the posterior.
        chain1 = sample(model, spl1, 100000; chain_type=StructArray, param_names=["μ", "σ"])
        chain2 = sample(model, spl2, 100000; chain_type=StructArray, param_names=["μ", "σ"])

        # chn_mean ≈ dist_mean atol=atol_v
        @test mean(chain1.μ) ≈ 0.0 atol=0.1
        @test mean(chain1.σ) ≈ 1.0 atol=0.1
        @test mean(chain2.μ) ≈ 0.0 atol=0.1
        @test mean(chain2.σ) ≈ 1.0 atol=0.1
    end

    @testset "RandomWalk" begin
        # Set up our sampler with initial parameters.
        spl1 = RWMH([Normal(0,1), Normal(0, 1)])
        spl2 = RWMH(MvNormal([0.0, 0.0], 1))

        # Sample from the posterior.
        chain1 = sample(model, spl1, 100000; chain_type=StructArray, param_names=["μ", "σ"])
        chain2 = sample(model, spl2, 100000; chain_type=StructArray, param_names=["μ", "σ"])

        # chn_mean ≈ dist_mean atol=atol_v
        @test mean(chain1.μ) ≈ 0.0 atol=0.1
        @test mean(chain1.σ) ≈ 1.0 atol=0.1
        @test mean(chain2.μ) ≈ 0.0 atol=0.1
        @test mean(chain2.σ) ≈ 1.0 atol=0.1
    end

    @testset "parallel sampling" begin
        spl1 = StaticMH([Normal(0,1), Normal(0, 1)])

        chain1 = sample(model, spl1, MCMCDistributed(), 10000, 4;
                        param_names=["μ", "σ"], chain_type=Chains)
        @test mean(chain1["μ"]) ≈ 0.0 atol=0.1
        @test mean(chain1["σ"]) ≈ 1.0 atol=0.1

        if VERSION >= v"1.3"
            chain2 = sample(model, spl1, MCMCThreads(), 10000, 4;
                            param_names=["μ", "σ"], chain_type=Chains)
            @test mean(chain2["μ"]) ≈ 0.0 atol=0.1
            @test mean(chain2["σ"]) ≈ 1.0 atol=0.1
        end
    end

    @testset "Proposal styles" begin
        m1 = DensityModel(x -> logpdf(Normal(x,1), 1.0))
        m2 = DensityModel(x -> logpdf(Normal(x[1], x[2]), 1.0))
        m3 = DensityModel(x -> logpdf(Normal(x.a, x.b), 1.0))
        m4 = DensityModel(x -> logpdf(Normal(x,1), 1.0))

        p1 = StaticProposal(Normal(0,1))
        p2 = StaticProposal([Normal(0,1), InverseGamma(2,3)])
        p3 = (a=StaticProposal(Normal(0,1)), b=StaticProposal(InverseGamma(2,3)))
        p4 = StaticProposal((x=1.0) -> Normal(x, 1))

        c1 = sample(m1, MetropolisHastings(p1), 100; chain_type=Vector{NamedTuple})
        c2 = sample(m2, MetropolisHastings(p2), 100; chain_type=Vector{NamedTuple})
        c3 = sample(m3, MetropolisHastings(p3), 100; chain_type=Vector{NamedTuple})
        c4 = sample(m4, MetropolisHastings(p4), 100; chain_type=Vector{NamedTuple})

        @test keys(c1[1]) == (:param_1, :lp)
        @test keys(c2[1]) == (:param_1, :param_2, :lp)
        @test keys(c3[1]) == (:a, :b, :lp)
        @test keys(c4[1]) == (:param_1, :lp)
    end

    @testset "Initial parameters" begin
        # Set up our sampler with initial parameters.
        spl1 = StaticMH([Normal(0,1), Normal(0, 1)])

        val = [0.4, 1.2]

        # Sample from the posterior.
        chain1 = sample(model, spl1, 10, init_params = val)

        @test chain1[1].params == val
    end

    @testset "is_symmetric_proposal" begin
        # True distributions
        d1 = Normal(5, .7)

        # Model definition.
        m1 = DensityModel(x -> logpdf(d1, x))

        # Set up the proposal (StandardNormal is a custom distribution in "util.jl").
        p1 = RandomWalkProposal(StandardNormal())

        # Implement `is_symmetric_proposal` for StandardNormal random walk proposal.
        AdvancedMH.is_symmetric_proposal(::RandomWalkProposal{<:StandardNormal}) = true

        # Make sure `is_symmetric_proposal` behaves correctly.
        @test AdvancedMH.is_symmetric_proposal(p1)

        # Sample from the posterior with initial parameters.
        chain1 = sample(m1, MetropolisHastings(p1), 100000;
                        chain_type=StructArray, param_names=["x"])

        @test mean(chain1.x) ≈ mean(d1) atol=0.05
        @test std(chain1.x) ≈ std(d1) atol=0.05
    end

    @testset "MALA" begin
        
        # Set up the sampler.
        sigma = 1e-1
        spl1 = MALA(x -> MvNormal((sigma^2 / 2) .* x, sigma))

        # Sample from the posterior with initial parameters.
        chain1 = sample(model, spl1, 100000; init_params=ones(2), chain_type=StructArray, param_names=["μ", "σ"])

        @test mean(chain1.μ) ≈ 0.0 atol=0.1
        @test mean(chain1.σ) ≈ 1.0 atol=0.1 
    end

    @testset "EMCEE" begin include("emcee.jl") end
  
end
