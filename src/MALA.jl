using ForwardDiff: gradient!
using DiffResults: GradientResult, value, gradient

struct MALA{D} <: MHSampler
    proposal::D
end


# Create a RandomWalkProposal if we weren't given one already.
MALA(d) = MALA(RandomWalkProposal(d))

# If we were given a RandomWalkProposal, just use that instead.
MALA(d::RandomWalkProposal) = MALA{typeof(d)}(d)


struct GradientTransition{T<:Union{Vector, Real, NamedTuple}, L<:Real, G<:Union{Vector, Real, NamedTuple}}
    params :: T
    lp :: L
    gradient :: G
end


# Store the new draw, its log density and its gradient
Transition_w∇(model::DensityModel, params) = Transition_w∇(params, ∂ℓπ∂θ(model, params)...)


function propose(
    rng::Random.AbstractRNG,
    spl::MALA{<:Proposal},
    model::DensityModel,
    params_prev::Transition_w∇
)
    proposal = propose(rng, spl.proposal(params_prev.∇), model, params_prev.params)
    return Transition_w∇(model, proposal)
end


function q(
    spl::MALA{<:Proposal},
    t::Transition_w∇,
    t_cond::Transition_w∇
)
    return q(spl.proposal(-t_cond.∇), t.params, t_cond.params)
end


"""
    ∂ℓπ∂θ(model::DensityModel, θ::T)

Efficiently returns the value and gradient of the model
"""
function ∂ℓπ∂θ(model::DensityModel, params)
    res = GradientResult(params)
    gradient!(res, model.logdensity, params)
    return (value(res), gradient(res))
end


logdensity(model::DensityModel, t::Transition_w∇) = t.lp


# Define the first step! function, which is called at the 
# beginning of sampling. Return the initial parameter used
# to define the sampler.
function AbstractMCMC.step!(
    rng::Random.AbstractRNG,
    model::DensityModel,
    spl::MALA{<:Proposal},
    N::Integer,
    ::Nothing;
    init_params=nothing,
    kwargs...
)
    if init_params === nothing
        @warn "need init_params"
    else
        return Transition_w∇(model, init_params)
    end
end



# Define the other step functions. Returns a Transition containing
# either a new proposal (if accepted) or the previous proposal 
# (if not accepted).
function AbstractMCMC.step!(
    rng::Random.AbstractRNG,
    model::DensityModel,
    spl::MALA{<:Proposal},
    ::Integer,
    params_prev;
    kwargs...
)
    # Generate a new proposal.
    params = propose(rng, spl, model, params_prev)

    # Calculate the log acceptance probability.
    logα = logdensity(model, params) - logdensity(model, params_prev) +
        q(spl, params_prev, params) - q(spl, params, params_prev)

    # Decide whether to return the previous params or the new one.
    if -Random.randexp(rng) < logα
        return params
    else
        return params_prev
    end
end
