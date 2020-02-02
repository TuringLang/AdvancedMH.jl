struct Proposal{T<:ProposalStyle, P}
    type :: T
    proposal :: P
end

# Random draws
Base.rand(p::Proposal{<:ProposalStyle, <:Distribution}) = rand(p.proposal)
function Base.rand(p::Proposal{<:ProposalStyle, <:AbstractArray})
    return map(rand, p.proposal)
end

# Densities
function Distributions.logpdf(p::Proposal{<:ProposalStyle, <:UnivariateDistribution}, v) 
    return sum(logpdf(p.proposal, v))
end
function Distributions.logpdf(p::Proposal{<:ProposalStyle, <:MultivariateDistribution}, v) 
    return sum(logpdf(p.proposal, v))
end
function Distributions.logpdf(p::Proposal{<:ProposalStyle, <:MatrixDistribution}, v) 
    return sum(logpdf(p.proposal, v))
end
function Distributions.logpdf(p::Proposal{<:ProposalStyle, <:AbstractArray}, v)
    return sum(map(x -> logpdf(x[1], x[2]), zip(p.proposal, v)))
end
function Distributions.logpdf(p::Proposal{<:ProposalStyle, <:Function}, v)
    return logpdf(p.proposal(v), v)
end

###############
# Random Walk #
###############

function propose(
    proposal::Proposal{RandomWalk, <:Distribution}, 
    model::DensityModel, 
    t
)
    return t + rand(proposal)
end

function propose(
    proposal::Proposal{RandomWalk, <:AbstractArray}, 
    model::DensityModel, 
    t
)
    return t + rand(proposal)
end

function q(
    proposal::Proposal{RandomWalk, <:Distribution}, 
    t,
    t_cond
)
    return sum(logpdf(proposal, t - t_cond))
end

function q(
    proposal::Proposal{RandomWalk, <:AbstractArray}, 
    t,
    t_cond
)
    return sum(logpdf(proposal, t - t_cond))
end

##########
# Static #
##########

propose(p::Proposal{RandomWalk}, m::DensityModel) = propose(Proposal(Static(), p.proposal), m)
function propose(
    proposal::Proposal{Static, <:Distribution}, 
    model::DensityModel, 
    t=nothing
)
    return rand(proposal)
end

function propose(
    p::Proposal{Static, <:AbstractArray}, 
    model::DensityModel, 
    t=nothing
)
    props = map(x -> rand(x), p.proposal)
    return props
end

function q(
    proposal::Proposal{Static, <:Distribution}, 
    t,
    t_cond
)
    return sum(logpdf(proposal, t))
end

function q(
    proposal::Proposal{Static, <:AbstractArray}, 
    t,
    t_cond
)
    return sum(logpdf(proposal, t))
end

############
# Function #
############

function propose(
    proposal::Proposal{<:ProposalStyle, <:Function}, 
    model::DensityModel
)
    p = proposal.proposal 
    return rand(proposal.proposal())
end

function propose(
    proposal::Proposal{<:ProposalStyle, <:Function}, 
    model::DensityModel,
    t
)
    p = proposal.proposal 
    return rand(proposal.proposal(t))
end

function q(
    proposal::Proposal{<:ProposalStyle, <:Function}, 
    t,
    t_cond
)
    p = proposal.proposal
    return sum(logpdf.(p(t_cond), t))
end