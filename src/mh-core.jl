"""
    MetropolisHastings{P<:ProposalStyle, T, D}

Fields:

- `init_θ` is the vector form of the parameters needed for the likelihood function.
- `proposal` is a function that dynamically constructs a conditional distribution.

Example:

```julia
MetropolisHastings([0.0, 0.0], x -> MvNormal(x, 1.0))
````
"""
struct MetropolisHastings{P<:ProposalStyle, T, D} <: Metropolis
    proposal_type :: P
    init_θ :: T
    proposal :: D
end

# Default constructors.
MetropolisHastings(init_θ::Real) = MetropolisHastings(init_θ, Normal(0,1))
MetropolisHastings(init_θ::Vector{<:Real}) = MetropolisHastings(init_θ, MvNormal(length(init_θ),1))

"""
    propose(spl::MetropolisHastings, model::DensityModel, t::Transition)

Generates a new parameter proposal conditional on the model, the sampler, and the previous
sample.
"""
@inline propose(spl::MetropolisHastings, model::DensityModel, t::Transition) = propose(spl, model, t.θ)

"""
    q(spl::MetropolisHastings, t1::Transition, t2::Transition)

Calculates the probability `q(θ | θcond)`, using the proposal distribution `spl.proposal`.
"""
@inline q(spl::MetropolisHastings, t1::Transition, t2::Transition) = q(spl, t1.θ, t2.θ)

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
    return Transition(model, spl.init_θ)
end

# Define the other step functions. Returns a Transition containing
# either a new proposal (if accepted) or the previous proposal 
# (if not accepted).
function step!(
    rng::AbstractRNG,
    model::DensityModel,
    spl::MetropolisHastings,
    ::Integer,
    θ_prev::Transition;
    kwargs...
)
    # Generate a new proposal.
    θ = propose(spl, model, θ_prev)

    # Calculate the log acceptance probability.
    α = ℓπ(model, θ) - ℓπ(model, θ_prev) + q(spl, θ_prev, θ) - q(spl, θ, θ_prev)

    # Decide whether to return the previous θ or the new one.
    if log(rand(rng)) < min(α, 0.0)
        return θ
    else
        return θ_prev
    end
end