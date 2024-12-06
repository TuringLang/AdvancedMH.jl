struct Gaussian{A}
    Σ::A
end
LogDensityProblems.dimension(model::Gaussian) = size(model.Σ, 1)
LogDensityProblems.capabilities(::Gaussian) = LogDensityProblems.LogDensityOrder{0}()
function LogDensityProblems.logdensity(model::Gaussian, x)
    d = LogDensityProblems.dimension(model)
    return logpdf(MvNormal(zeros(d), model.Σ), x)
end

Base.@kwdef struct StatesExtractor{A}
    states::A = Vector{Any}()
end
function (callback::StatesExtractor)(
    rng,
    model,
    sampler,
    sample,
    state,
    iteration;
    kwargs...,
)
    if iteration == 1
        empty!(callback.states)
    end

    push!(callback.states, state)
end

@testset "RobustAdaptiveMetropolis" begin
    # Testing of sampling is done in the docstring. Here we test explicit properties of the sampler.
    @testset "eigenvalue bounds" begin
        for (σ², lower_bound, upper_bound) in [
            (10.0, 0.5, 1.1),  # should hit upper bound easily
            (0.01, 0.5, 1.1), # should hit lower bound easily
        ]
            ρ = σ² / 2
            model = Gaussian([σ² ρ; ρ σ²])
            callback = StatesExtractor()

            # Use aggressive adaptation.
            sampler =
                RAM(γ = 0.51, eigenvalue_lower_bound = 0.9, eigenvalue_upper_bound = 1.1)
            num_warmup = 1000
            discard_initial = 0  # we're only keeping the warmup samples
            chain = sample(
                model,
                sampler,
                num_warmup;
                chain_type = Chains,
                num_warmup,
                discard_initial,
                progress = false,
                initial_params = zeros(2),
                callback = callback,
            )
            S_samples = getproperty.(callback.states, :S)
            eigval_min = map(minimum, eachrow(mapreduce(eigvals, hcat, S_samples)))
            eigval_max = map(maximum, eachrow(mapreduce(eigvals, hcat, S_samples)))
            @test all(>=(sampler.eigenvalue_lower_bound), eigval_min)
            @test all(<=(sampler.eigenvalue_upper_bound), eigval_max)

            if σ² < lower_bound
                # We should hit the lower bound.
                @test all(≈(sampler.eigenvalue_lower_bound, atol = 0.05), eigval_min)
            else
                # We should hit the upper bound.
                @test all(≈(sampler.eigenvalue_upper_bound, atol = 0.05), eigval_max)
            end
        end
    end
end
