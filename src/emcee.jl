

struct EmceeTransition{T<:Transition}
    walkers :: Vector{T}
end

mutable struct Emcee{F<:Real} <: ProposalStyle
    stretch_size :: F
    num_walkers :: Int
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::DensityModel,
    spl::MetropolisHastings,
    ::Integer,
    params_prev::EmceeTransition;
    kwargs...
)
    # Generate a new proposal.
    params = propose(spl, model, params_prev)
    return params
end

# Make a proposal from a distribution.
function propose(
    spl::MetropolisHastings{<:Proposal{<:Emcee}},
    model::DensityModel
)
    proposal = propose(spl.proposal, model)
    return EmceeTransition(map(p -> Transition(model, p), proposal))
end

function propose(
    spl::MetropolisHastings{<:Proposal{<:Emcee}},
    model::DensityModel,
    t
)
    proposal = propose(spl.proposal, model, t)
    return EmceeTransition(proposal)
end

function propose(
    proposal::Proposal{<:Emcee, <:Distribution}, 
    model::DensityModel
)
    g = map(i -> rand(proposal), 1:proposal.type.num_walkers)
    return g
end

function propose(
    proposal::Proposal{<:Emcee, <:AbstractArray{T, 2}}, 
    model::DensityModel
) where T
    size(proposal.proposal, 2) == proposal.type.num_walkers || 
        error("""Initial matrix must have n_walkers ($(proposal.type.num_walkers)) columns.
        Provided matrix has $(size(proposal.proposal, 2)) columns.""")
    return map(i -> proposal.proposal[:, i], 1:size(proposal.proposal, 2))
end

function q(
    proposal::Proposal{<:Emcee, <:Distribution}, 
    t,
    t_cond
)
    return sum(logpdf(proposal, t.params - t_cond.params))
end

function propose(
    proposal::Proposal{<:Emcee}, 
    model::DensityModel,
    t
)
    walkers = t.walkers
    ns = length(walkers[1].params)
    zs = ((proposal.type.stretch_size - 1.0) .* rand(proposal.type.num_walkers) .+ 1) .^ 2.0 ./ proposal.type.stretch_size
    interval = collect(1:proposal.type.num_walkers)
    partition = interval[1:div(proposal.type.num_walkers, 2)]
    alphamult = (ns - 1) .* log.(zs)

    vals = map(interval) do k
        xk = walkers[k]
        xj = walkers[rand(filter(z -> z != k, interval))]
        z = zs[k]
        y = xk.params + z .* (xj.params - xk.params)
        new_params = Transition(model, y)
        alpha = alphamult[k] + new_params.lp - xk.lp + q(proposal, xk, new_params) - q(proposal, new_params, xk)

        if alpha >= log(rand())
            new_params
        else
            xk
        end
    end

    return vals
end