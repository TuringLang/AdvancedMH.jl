struct AMSampler{D} <: MHSampler
    proposal::D
end

propose(spl::AMSampler, args...) = propose(Random.GLOBAL_RNG, spl, args...)

function propose(rng::Random.AbstractRNG, spl::AMSampler{<:Proposal}, model::DensityModel)
    proposal = propose(rng, spl.proposal, model)
    return Transition(model, proposal)
end

function propose(rng::Random.AbstractRNG, spl::AMSampler{<:Proposal}, model::DensityModel,
                 params_prev::Transition)
    proposal = propose(rng, spl.proposal, model, params_prev.params)
    return Transition(model, proposal)
end

function AbstractMCMC.step(rng::Random.AbstractRNG, model::DensityModel, spl::AMSampler;
                           init_params=nothing, kwargs...)
    if init_params === nothing
        transition = propose(rng, spl, model)
    else
        transition = transition(spl, model, init_params)
    end

    trackstep(spl.proposal, transition)
    return transition, transition
end

function AbstractMCMC.step(rng::Random.AbstractRNG, model::DensityModel, spl::AMSampler,
                           params_prev::AbstractTransition; kwargs...)
    # Generate a new proposal.
    params = propose(rng, spl, model, params_prev)
    
    # Calculate the log acceptance probability.
    logα = logdensity(model, params) - logdensity(model, params_prev) +
        logratio_proposal_density(spl, params_prev, params)

    # Decide whether to return the previous params or the new one.
    if -Random.randexp(rng) < logα
        trackstep(spl.proposal, params, Val(true))
        return params, params
   else
        trackstep(spl.proposal, params_prev, Val(false))
        return params_prev, params_prev
    end
end

# Called when the first sample is drawn
trackstep(proposal::Proposal, params::Transition) = nothing

# Called when a new step is performed, the last argument determines if the step is an acceptance step
trackstep(proposal::Proposal, params::Transition, 
          ::Union{Val{true}, Val{false}}) = nothing

# Simple Adaptive Metropolis Proposal
# The proposal at each step is equal to a scalar multiple
# of the posterior covariance plus a fixed, small covariance
# matrix epsilon which is also used for initial exploration.
mutable struct AMProposal <: Proposal{MvNormal}
    epsilon::Symmetric
    scalefactor::Float64
    proposal::MvNormal
    samplemean::AbstractArray
    samplesqmean::AbstractMatrix
    N::Int
end

function AMProposal(epsilon::AbstractMatrix, scalefactor=2.38^2 / size(epsilon, 1))
    sym = Symmetric(epsilon)
    proposal = MvNormal(zeros(size(sym, 1)), sym)
    AMProposal(sym, scalefactor, proposal, mean(proposal), zeros(size(sym)...), 0)
end

function logratio_proposal_density(sampler::AMSampler, params_prev::Transition, params::Transition)
    return logratio_proposal_density(sampler.proposal, params_prev.params, params.params)
end

logratio_proposal_density(::AMProposal, params_prev, params) = 0

function trackstep(proposal::AMProposal, trans::Transition)
    proposal.samplemean = copy(trans.params)
    proposal.samplesqmean = trans.params * trans.params'
    proposal.proposal = MvNormal(zeros(size(proposal.samplemean)), proposal.epsilon)
    proposal.N = 1
end

function trackstep(proposal::AMProposal, trans::Transition, ::Union{Val{true}, Val{false}})
    proposal.samplemean = (proposal.samplemean .* proposal.N .+ trans.params) ./ (proposal.N + 1)

    proposal.samplesqmean = (proposal.samplesqmean .* proposal.N + trans.params * trans.params') ./ 
                             (proposal.N + 1)

    samplecov = proposal.samplesqmean .- proposal.samplemean * proposal.samplemean'
    prop_cov = proposal.scalefactor .* samplecov
    proposal.proposal = MvNormal(mean(proposal.proposal), prop_cov .+ proposal.epsilon)
    proposal.N += 1
end

function propose(rng::Random.AbstractRNG, p::AMProposal, ::DensityModel)
    return rand(rng, p.proposal)
end

function propose(rng::Random.AbstractRNG, p::AMProposal, ::DensityModel, t)
    return t + rand(rng, p.proposal)
end

