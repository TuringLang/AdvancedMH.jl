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
GradientTransition(model::DensityModel, params) = GradientTransition(params, logdensity_and_gradient(model, params)...)


function propose(
    rng::Random.AbstractRNG,
    spl::MALA{<:Proposal},
    model::DensityModel,
    params_prev::GradientTransition
)
    proposal = propose(rng, spl.proposal(params_prev.gradient), model, params_prev.params)
    return GradientTransition(model, proposal)
end


function q(
    spl::MALA{<:Proposal},
    t::GradientTransition,
    t_cond::GradientTransition
)
    return q(spl.proposal(-t_cond.gradient), t.params, t_cond.params)
end


"""
    logdensity_and_gradient(model::DensityModel, params)

Efficiently returns the value and gradient of the model
"""
function logdensity_and_gradient(model::DensityModel, params)
    res = GradientResult(params)
    gradient!(res, model.logdensity, params)
    return (value(res), gradient(res))
end


logdensity(model::DensityModel, t::GradientTransition) = t.lp


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
        return GradientTransition(model, init_params)
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
