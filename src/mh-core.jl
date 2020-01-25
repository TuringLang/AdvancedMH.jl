"""
    MetropolisHastings{P<:ProposalStyle, T, D}

Fields:

- `init_params` is the vector form of the parameters needed for the likelihood function.
- `proposal` is a function that dynamically constructs a conditional distribution.

Example:

```julia
MetropolisHastings([0.0, 0.0], x -> MvNormal(x, 1.0))
````
"""
mutable struct MetropolisHastings{P<:ProposalStyle, D, T} <: Metropolis
    proposal_type :: P
    proposal :: D
    init_params :: T
end

# Define the first step! function, which is called at the 
# beginning of sampling. Return the initial parameter used
# to define the sampler.
function step!(
    rng::AbstractRNG,
    model::DensityModel,
    spl::MetropolisHastings,
    N::Integer;
    kwargs...
)
    return Transition(model, spl.init_params)
end

# Define the other step functions. Returns a Transition containing
# either a new proposal (if accepted) or the previous proposal 
# (if not accepted).
function step!(
    rng::AbstractRNG,
    model::DensityModel,
    spl::MetropolisHastings,
    ::Integer,
    params_prev::Transition;
    kwargs...
)
    # Generate a new proposal.
    params = propose(spl, model, params_prev)

    # Calculate the log acceptance probability.
    α = logdensity(model, params) - logdensity(model, params_prev) + 
        q(spl, params_prev, params) - q(spl, params, params_prev)

    # Decide whether to return the previous params or the new one.
    if log(rand(rng)) < min(α, 0.0)
        return params
    else
        return params_prev
    end
end