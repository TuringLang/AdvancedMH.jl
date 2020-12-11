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

transition(::MALA, model, params) = GradientTransition(model, params)

# Store the new draw, its log density and its gradient
GradientTransition(model::DensityModel, params) = GradientTransition(params, logdensity_and_gradient(model, params)...)

propose(rng::Random.AbstractRNG, ::MALA, model) = error("please specify initial parameters")

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
