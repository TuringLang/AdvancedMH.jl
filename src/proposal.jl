abstract type Proposal{P} end

struct StaticProposal{P} <: Proposal{P}
    proposal::P
end

struct RandomWalkProposal{issymmetric,P} <: Proposal{P}
    proposal::P
end

RandomWalkProposal(proposal) = RandomWalkProposal{false}(proposal)
function RandomWalkProposal{issymmetric}(proposal) where {issymmetric}
    return RandomWalkProposal{issymmetric,typeof(proposal)}(proposal)
end

# Random draws
Base.rand(p::Proposal, args...) = rand(Random.GLOBAL_RNG, p, args...)
Base.rand(rng::Random.AbstractRNG, p::Proposal{<:Distribution}) = rand(rng, p.proposal)
function Base.rand(rng::Random.AbstractRNG, p::Proposal{<:AbstractArray})
    return map(x -> rand(rng, x), p.proposal)
end

# Densities
Distributions.logpdf(p::Proposal{<:Distribution}, v) = logpdf(p.proposal, v)
function Distributions.logpdf(p::Proposal{<:AbstractArray}, v)
    # `mapreduce` with multiple iterators requires Julia 1.2 or later
    return mapreduce(((pi, vi),) -> logpdf(pi, vi), +, zip(p.proposal, v))
end

###############
# Random Walk #
###############

function propose(rng::Random.AbstractRNG, p::RandomWalkProposal, m::DensityModel)
    return propose(rng, StaticProposal(p.proposal), m)
end

function propose(
    rng::Random.AbstractRNG,
    proposal::RandomWalkProposal,
    model::DensityModel, 
    t
)
    return t + rand(rng, proposal)
end

function q(
    proposal::RandomWalkProposal,
    t,
    t_cond
)
    return logpdf(proposal, t - t_cond)
end

##########
# Static #
##########

function propose(
    rng::Random.AbstractRNG,
    proposal::StaticProposal{<:Union{Distribution,AbstractArray}},
    model::DensityModel,
    t=nothing
)
    return rand(rng, proposal)
end

function q(
    proposal::StaticProposal{<:Union{Distribution,AbstractArray}},
    t,
    t_cond
)
    return logpdf(proposal, t)
end

############
# Function #
############

# function definition with abstract types requires Julia 1.3 or later
(p::StaticProposal{<:Function})() = StaticProposal(p.proposal())
(p::StaticProposal{<:Function})(t) = StaticProposal(p.proposal(t))
function (p::RandomWalkProposal{issymmetric,<:Function})() where {issymmetric}
    return RandomWalkProposal{issymmetric}(p.proposal())
end
function (p::RandomWalkProposal{issymmetric,<:Function})(t) where {issymmetric}
    return RandomWalkProposal{issymmetric}(p.proposal(t))
end

function propose(
    rng::Random.AbstractRNG,
    proposal::Proposal{<:Function},
    model::DensityModel
)
    return propose(rng, proposal(), model)
end

function propose(
    rng::Random.AbstractRNG,
    proposal::Proposal{<:Function}, 
    model::DensityModel,
    t
)
    return propose(rng, proposal(t), model)
end

function q(
    proposal::Proposal{<:Function},
    t,
    t_cond
)
    return q(proposal(t_cond), t, t_cond)
end

"""
    logratio_proposal_density(proposal, state, candidate)

Compute the log-ratio of the proposal densities in the Metropolis-Hastings algorithm.

The log-ratio of the proposal densities is defined as
```math
\\log \\frac{g(x | x')}{g(x' | x)},
```
where ``x`` is the current state, ``x'`` is the proposed candidate for the next state,
and ``g(y' | y)`` is the conditional probability of proposing state ``y'`` given state
``y`` (proposal density).
"""
function logratio_proposal_density(proposal::Proposal, state, candidate)
    return q(proposal, state, candidate) - q(proposal, candidate, state)
end

# ratio is always 0 for symmetric random walk proposals
logratio_proposal_density(::RandomWalkProposal{true}, state, candidate) = 0

function logratio_proposal_density(proposals::NamedTuple, states, candidates)
    return sum(keys(proposals)) do key
        return logratio_proposal_density(proposals[key], states[key], candidates[key])
    end
end

# fallback for iterators (arrays, tuples, etc.)
function logratio_proposal_density(proposals, states, candidates)
    return sum(zip(proposals, states, candidates)) do (proposal, state, candidate)
        return logratio_proposal_density(proposal, state, candidate)
    end
end
        
