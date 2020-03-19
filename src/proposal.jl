abstract type Proposal{P} end

struct StaticProposal{P} <: Proposal{P}
    proposal::P
end

struct RandomWalkProposal{P} <: Proposal{P}
    proposal::P
end

# Random draws
Base.rand(p::Proposal{<:Distribution}) = rand(p.proposal)
Base.rand(p::Proposal{<:AbstractArray}) = map(rand, p.proposal)

# Densities
Distributions.logpdf(p::Proposal{<:Distribution}, v) = logpdf(p.proposal, v)
function Distributions.logpdf(p::Proposal{<:AbstractArray}, v)
    # `mapreduce` with multiple iterators requires Julia 1.2 or later
    return mapreduce(((pi, vi),) -> logpdf(pi, vi), +, zip(p.proposal, v))
end

###############
# Random Walk #
###############

function propose(p::RandomWalkProposal, m::DensityModel)
    return propose(StaticProposal(p.proposal), m)
end

function propose(
    proposal::RandomWalkProposal{<:Union{Distribution,AbstractArray}}, 
    model::DensityModel, 
    t
)
    return t + rand(proposal)
end

function q(
    proposal::RandomWalkProposal{<:Union{Distribution,AbstractArray}}, 
    t,
    t_cond
)
    return logpdf(proposal, t - t_cond)
end

##########
# Static #
##########

function propose(
    proposal::StaticProposal{<:Union{Distribution,AbstractArray}},
    model::DensityModel,
    t=nothing
)
    return rand(proposal)
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
for T in (StaticProposal, RandomWalkProposal)
    @eval begin
        (p::$T{<:Function})() = $T(p.proposal())
        (p::$T{<:Function})(t) = $T(p.proposal(t))
    end
end

function propose(
    proposal::Proposal{<:Function},
    model::DensityModel
)
    return propose(proposal(), model)
end

function propose(
    proposal::Proposal{<:Function}, 
    model::DensityModel,
    t
)
    return propose(proposal(t), model)
end

function q(
    proposal::Proposal{<:Function},
    t,
    t_cond
)
    return q(proposal(t_cond), t, t_cond)
end