using .ForwardDiff: gradient!
using .DiffResults: GradientResult, value, gradient

struct MALA{D} <: MHSampler
    proposal::D
end


# Create a RandomWalkProposal if we weren't given one already.
MALA(d) = MALA(RandomWalkProposal(d))

# If we were given a RandomWalkProposal, just use that instead.
MALA(d::RandomWalkProposal) = MALA{typeof(d)}(d)


struct GradientTransition{T<:Union{Vector, Real, NamedTuple}, L<:Real, G<:Union{Vector, Real, NamedTuple}} <: AbstractTransition
    params::T
    lp::L
    gradient::G
end

logdensity(model::DensityModel, t::GradientTransition) = t.lp

propose(rng::Random.AbstractRNG, ::MALA, model) = error("please specify initial parameters")
function propose(
    rng::Random.AbstractRNG,
    spl::MALA{<:Proposal},
    model::DensityModel,
    params_prev::GradientTransition
)
    return propose(rng, spl.proposal(params_prev.gradient), model, params_prev.params)
end

function transition(sampler::MALA, model::DensityModel, params)
    logdensity_gradient = logdensity_and_gradient(model, params)
    return transition(sampler, model, params, logdensity_gradient)
end
function transition(::MALA, model::DensityModel, params, (logdensity, gradient))
    return GradientTransition(params, logdensity, gradient)
end

function logacceptance_logdensity(
    sampler::MALA,
    model::DensityModel,
    transition_prev::GradientTransition,
    candidate,
)
    # compute both the value of the log density and its gradient
    logdensity_candidate, gradient_logdensity_candidate = logdensity_and_gradient(
        model, candidate
    )

    # compute log ratio of proposal densities
    proposal = sampler.proposal
    state = transition_prev.params
    gradient_logdensity_state = transition_prev.gradient
    logratio_proposal_density = q(
        proposal(-gradient_logdensity_candidate), state, candidate
    ) - q(proposal(-gradient_logdensity_state), candidate, state)

    # compute log acceptance probability
    logα = logdensity_candidate - logdensity(model, transition_prev) +
        logratio_proposal_density

    return logα, (logdensity_candidate, gradient_logdensity_candidate)
end

"""
    logdensity_and_gradient(model::DensityModel, params)

Return the value and gradient of the log density of the parameters `params` for the `model`.
"""
function logdensity_and_gradient(model::DensityModel, params)
    res = GradientResult(params)
    gradient!(res, model.logdensity, params)
    return value(res), gradient(res)
end

