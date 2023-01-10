using AdvancedMH
using DiffResults
using Distributions
using ForwardDiff
using MCMCChains
using StructArrays

using LogDensityProblems: LogDensityProblems
using LogDensityProblemsAD: LogDensityProblemsAD

using LinearAlgebra
using Random
using Test

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

    # `LogDensityModel`
    LogDensityProblems.logdensity(::typeof(density), θ) = density(θ)
    LogDensityProblems.dimension(::typeof(density)) = 2

    @testset "StaticMH" begin
        # Set up our sampler with initial parameters.
        spl1 = StaticMH([Normal(0,1), Normal(0, 1)])
        spl2 = StaticMH(MvNormal(zeros(2), I))

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
        spl2 = RWMH(MvNormal(zeros(2), I))

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

    @testset "MCMCChains" begin
        # Array of parameters
        chain1 = sample(
            model, StaticMH([Normal(0,1), Normal(0, 1)]), 10_000;
            param_names=["μ", "σ"], chain_type=Chains
        )
        @test chain1 isa Chains
        @test range(chain1) == 1:10_000
        @test mean(chain1["μ"]) ≈ 0.0 atol=0.1
        @test mean(chain1["σ"]) ≈ 1.0 atol=0.1

        chain1b = sample(
            model, StaticMH([Normal(0,1), Normal(0, 1)]), 10_000;
            param_names=["μ", "σ"], chain_type=Chains, discard_initial=25, thinning=4,
        )
        @test chain1b isa Chains
        @test range(chain1b) == range(26; step=4, length=10_000)
        @test mean(chain1b["μ"]) ≈ 0.0 atol=0.1
        @test mean(chain1b["σ"]) ≈ 1.0 atol=0.1

        # NamedTuple of parameters
        chain2 = sample(
            model,
            MetropolisHastings(
                (μ = StaticProposal(Normal(0,1)), σ = StaticProposal(Normal(0, 1)))
            ), 10_000;
            chain_type=Chains
        )
        @test chain2 isa Chains
        @test range(chain2) == 1:10_000
        @test mean(chain2["μ"]) ≈ 0.0 atol=0.1
        @test mean(chain2["σ"]) ≈ 1.0 atol=0.1

        chain2b = sample(
            model,
            MetropolisHastings(
                (μ = StaticProposal(Normal(0,1)), σ = StaticProposal(Normal(0, 1)))
            ), 10_000;
            chain_type=Chains, discard_initial=25, thinning=4,
        )
        @test chain2b isa Chains
        @test range(chain2b) == range(26; step=4, length=10_000)
        @test mean(chain2b["μ"]) ≈ 0.0 atol=0.1
        @test mean(chain2b["σ"]) ≈ 1.0 atol=0.1

        # Scalar parameter
        chain3 = sample(
            DensityModel(x -> loglikelihood(Normal(x, 1), data)),
            StaticMH(Normal(0, 1)), 10_000; param_names=["μ"], chain_type=Chains
        )
        @test chain3 isa Chains
        @test range(chain3) == 1:10_000
        @test mean(chain3["μ"]) ≈ 0.0 atol=0.1

        chain3b = sample(
            DensityModel(x -> loglikelihood(Normal(x, 1), data)),
            StaticMH(Normal(0, 1)), 10_000;
            param_names=["μ"], chain_type=Chains, discard_initial=25, thinning=4,
        )
        @test chain3b isa Chains
        @test range(chain3b) == range(26; step=4, length=10_000)
        @test mean(chain3b["μ"]) ≈ 0.0 atol=0.1
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

    @testset "symmetric proposals" begin
        # True distributions
        d1 = Normal(5, .7)

        # Model definition.
        m1 = DensityModel(x -> logpdf(d1, x))

        # Custom normal distribution without `logpdf` defined errors since the
        # acceptance probability cannot be computed
        p1 = RandomWalkProposal(CustomNormal())
        @test p1 isa RandomWalkProposal{false}
        @test_throws MethodError AdvancedMH.logratio_proposal_density(p1, randn(), randn())
        @test_throws MethodError sample(m1, MetropolisHastings(p1), 10)

        p1 = StaticProposal(x -> CustomNormal(x))
        @test p1 isa StaticProposal{false}
        @test_throws MethodError AdvancedMH.logratio_proposal_density(p1, randn(), randn())
        @test_throws MethodError sample(m1, MetropolisHastings(p1), 10)

        # If the proposal is declared to be symmetric, the log ratio of the proposal
        # density is not evaluated.
        p2 = SymmetricRandomWalkProposal(CustomNormal())
        @test p2 isa RandomWalkProposal{true}
        p2 = SymmetricStaticProposal(x -> CustomNormal(x))
        @test p2 isa StaticProposal{true}

        for p2 in (
            SymmetricRandomWalkProposal(CustomNormal()),
            SymmetricStaticProposal((x=0) -> CustomNormal(x)),
        )
            @test iszero(AdvancedMH.logratio_proposal_density(p2, randn(), randn()))
            @test iszero(AdvancedMH.logratio_proposal_density([p2], randn(1), randn(1)))
            @test iszero(AdvancedMH.logratio_proposal_density(
                (p2,), (randn(),), (randn(),)
            ))
            @test iszero(AdvancedMH.logratio_proposal_density(
                (; x=p2), (; x=randn()), (; x=randn())
            ))
            chain1 = sample(
                m1, MetropolisHastings(p2), 100000;
                chain_type=StructArray, param_names=["x"]
            )
            @test mean(chain1.x) ≈ mean(d1) atol=0.05
            @test std(chain1.x) ≈ std(d1) atol=0.05
        end

        # type inference checks (arrays of proposals are not guaranteed to be type stable)
        proposals = (
            StaticProposal(Normal()),
            StaticProposal(x -> Normal(x, 1)),
            StaticProposal{true}(Cauchy()),
            StaticProposal{true}(x -> Cauchy(x, 2)),
            RandomWalkProposal(Laplace()),
            RandomWalkProposal(x -> Laplace(x, 1)),
            RandomWalkProposal{true}(TDist(1)),
            RandomWalkProposal{true}(x -> TDist(1)),
        )
        states = randn(2)
        candidates = randn(2)
        for (p1, p2) in Iterators.product(proposals, proposals)
            val = AdvancedMH.logratio_proposal_density(p1, states[1], candidates[1]) +
                AdvancedMH.logratio_proposal_density(p2, states[2], candidates[2])
            @test AdvancedMH.logratio_proposal_density([p1, p2], states, candidates) ≈ val
            @test @inferred AdvancedMH.logratio_proposal_density(
                (p1, p2), (states[1], states[2]), (candidates[1], candidates[2])
            ) ≈ val
            @test @inferred AdvancedMH.logratio_proposal_density(
                (x=p1, y=p2), (y=states[2], x=states[1]), (x=candidates[1], y=candidates[2])
            ) ≈ val
        end
    end

    @testset "MALA" begin
        # Set up the sampler.
        σ² = 0.01
        spl1 = MALA(x -> MvNormal((σ² / 2) .* x, σ² * I))

        # Sample from the posterior with initial parameters.
        chain1 = sample(model, spl1, 100000; init_params=ones(2), chain_type=StructArray, param_names=["μ", "σ"])

        @test mean(chain1.μ) ≈ 0.0 atol=0.1
        @test mean(chain1.σ) ≈ 1.0 atol=0.1

        @testset "LogDensityProblems interface" begin
            admodel = LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), density)
            chain2 = sample(
                admodel,
                spl1,
                100000;
                init_params=ones(2),
                chain_type=StructArray,
                param_names=["μ", "σ"]
            )

            @test mean(chain2.μ) ≈ 0.0 atol=0.1
            @test mean(chain2.σ) ≈ 1.0 atol=0.1
        end
    end

    @testset "EMCEE" begin include("emcee.jl") end
end
