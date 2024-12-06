module RobustAdaptiveMetropolis

using Random, LogDensityProblems, LinearAlgebra, AbstractMCMC
using DocStringExtensions: FIELDS

using AdvancedMH: AdvancedMH

export RAM

# TODO: Should we generalise this arbitrary symmetric proposals?
"""
    RAM

Robust Adaptive Metropolis-Hastings (RAM).

This is a simple implementation of the RAM algorithm described in [^VIH12].

# Fields

$(FIELDS)

# Examples

The following demonstrates how to implement a simple Gaussian model and sample from it using the RAM algorithm.

```jldoctest ram-gaussian
julia> using AdvancedMH, Random, Distributions, MCMCChains, LogDensityProblems, LinearAlgebra

julia> # Define a Gaussian with zero mean and some covariance.
       struct Gaussian{A}
           Σ::A
       end

julia> # Implement the LogDensityProblems interface.
       LogDensityProblems.dimension(model::Gaussian) = size(model.Σ, 1)

julia> function LogDensityProblems.logdensity(model::Gaussian, x)
           d = LogDensityProblems.dimension(model)
           return logpdf(MvNormal(zeros(d),model.Σ), x)
       end

julia> LogDensityProblems.capabilities(::Gaussian) = LogDensityProblems.LogDensityOrder{0}()

julia> # Construct the model. We'll use a correlation of 0.5.
       model = Gaussian([1.0 0.5; 0.5 1.0]);

julia> # Number of samples we want in the resulting chain.
       num_samples = 10_000;

julia> # Number of warmup steps, i.e. the number of steps to adapt the covariance of the proposal.
       # Note that these are not included in the resulting chain, as `discard_initial=num_warmup`
       # by default in the `sample` call. To include them, pass `discard_initial=0` to `sample`.
       num_warmup = 10_000;

julia> # Set the seed so get some consistency.
       Random.seed!(1234);

julia> # Sample!
       chain = sample(model, RAM(), 10_000; chain_type=Chains, num_warmup, progress=false, initial_params=zeros(2));

julia> norm(cov(Array(chain)) - [1.0 0.5; 0.5 1.0]) < 0.2
true
```

It's also possible to restrict the eigenvalues to avoid either too small or too large values. See p. 13 in [^VIH12].

```jldoctest ram-gaussian`
julia> chain = sample(
           model,
           RAM(eigenvalue_lower_bound=0.1, eigenvalue_upper_bound=2.0),
           10_000;
           chain_type=Chains, num_warmup, progress=false, initial_params=zeros(2)
       );

julia> norm(cov(Array(chain)) - [1.0 0.5; 0.5 1.0]) < 0.2
true
````

# References
[^VIH12]: Vihola (2012) Robust adaptive Metropolis algorithm with coerced acceptance rate, Statistics and computing.
"""
Base.@kwdef struct RAM{T,A<:Union{Nothing,AbstractMatrix{T}}} <: AdvancedMH.MHSampler
    "target acceptance rate. Default: 0.234."
    α::T=0.234
    "negative exponent of the adaptation decay rate. Default: `0.6`."
    γ::T=0.6
    "initial lower-triangular Cholesky factor. Default: `nothing`."
    S::A=nothing
    "lower bound on eigenvalues of the adapted Cholesky factor. Default: `0.0`."
    eigenvalue_lower_bound::T=0.0
    "upper bound on eigenvalues of the adapted Cholesky factor. Default: `Inf`."
    eigenvalue_upper_bound::T=Inf
end

"""
    RAMState

State of the Robust Adaptive Metropolis-Hastings (RAM) algorithm.

See also: [`RAM`](@ref).

# Fields
$(FIELDS)
"""
struct RAMState{T1,L,A,T2,T3}
    "current realization of the chain."
    x::T1
    "log density of `x` under the target model."
    logprob::L
    "current lower-triangular Cholesky factor."
    S::A
    "log acceptance ratio of the previous iteration (not necessarily of `x`)."
    logα::T2
    "current step size for adaptation of `S`."
    η::T3
    "current iteration."
    iteration::Int
    "whether the previous iteration was accepted."
    isaccept::Bool
end

AbstractMCMC.getparams(state::RAMState) = state.x
AbstractMCMC.setparams!!(state::RAMState, x) = RAMState(x, state.logprob, state.S, state.logα, state.η, state.iteration, state.isaccept)

function step_inner(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::RAM,
    state::RAMState
)
    # This is the initial state.
    f = model.logdensity
    d = LogDensityProblems.dimension(f)

    # Sample the proposal.
    x = state.x
    U = randn(rng, d)
    x_new = x + state.S * U

    # Compute the acceptance probability.
    lp = state.logprob
    lp_new = LogDensityProblems.logdensity(f, x_new)
    logα = min(lp_new - lp, zero(lp))  # `min` because we'll use this for updating
    # TODO: use `randexp` instead.
    isaccept = log(rand(rng)) < logα

    return x_new, lp_new, U, logα, isaccept
end

function adapt(sampler::RAM, state::RAMState, logα::Real, U::AbstractVector)
    # Update `
    Δα = exp(logα) - sampler.α
    S = state.S
    # TODO: Make this configurable by defining a more general path.
    η = state.iteration^(-sampler.γ)
    ΔS = η * abs(Δα) * S * U / norm(U)
    # TODO: Maybe do in-place and then have the user extract it with a callback if they really want it.
    S_new = if sign(Δα) == 1
        # One rank update.
        LinearAlgebra.lowrankupdate(Cholesky(S), ΔS).L
    else
        # One rank downdate.
        LinearAlgebra.lowrankdowndate(Cholesky(S), ΔS).L
    end
    return S_new, η
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::RAM;
    initial_params=nothing,
    kwargs...
)
    # This is the initial state.
    f = model.logdensity
    d = LogDensityProblems.dimension(f)

    # Initial parameter state.
    x = initial_params === nothing ? rand(rng, d) : initial_params
    # Initialize the Cholesky factor of the covariance matrix.
    S = LowerTriangular(sampler.S === nothing ? diagm(0 => ones(eltype(sampler.γ), d)) : sampler.S)

    # Constuct the initial state.
    lp = LogDensityProblems.logdensity(f, x)
    state = RAMState(x, lp, S, 0.0, 0, 1, true)

    return AdvancedMH.Transition(x, lp, true), state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::RAM,
    state::RAMState;
    kwargs...
)
    # Take the inner step.
    x_new, lp_new, U, logα, isaccept = step_inner(rng, model, sampler, state)
    # Accept / reject the proposal.
    state_new = RAMState(isaccept ? x_new : state.x, isaccept ? lp_new : state.logprob, state.S, logα, state.η, state.iteration + 1, isaccept)
    return AdvancedMH.Transition(state_new.x, state_new.logprob, state_new.isaccept), state_new
end

function valid_eigenvalues(S, lower_bound, upper_bound)
    # Short-circuit if the bounds are the default.
    (lower_bound == 0 && upper_bound == Inf) && return true
    # Note that this is just the diagonal when `S` is triangular.
    eigenvals = LinearAlgebra.eigvals(S)
    return all(lower_bound .<= eigenvals .<= upper_bound)
end

function AbstractMCMC.step_warmup(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::RAM,
    state::RAMState;
    kwargs...
)
    # Take the inner step.
    x_new, lp_new, U, logα, isaccept = step_inner(rng, model, sampler, state)
    # Adapt the proposal.
    S_new, η = adapt(sampler, state, logα, U)
    # Check that `S_new` has eigenvalues in the desired range.
    if !valid_eigenvalues(S_new, sampler.eigenvalue_lower_bound, sampler.eigenvalue_upper_bound)
        # In this case, we just keep the old `S` (p. 13 in Vihola, 2012).
        S_new = state.S
    end

    # Update state.
    state_new = RAMState(isaccept ? x_new : state.x, isaccept ? lp_new : state.logprob, S_new, logα, η, state.iteration + 1, isaccept)
    return AdvancedMH.Transition(state_new.x, state_new.logprob, state_new.isaccept), state_new
end

end
