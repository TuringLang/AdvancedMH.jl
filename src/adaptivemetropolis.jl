# Simple Adaptive Metropolis Proposal
# The proposal at each step is equal to a scalar multiple
# of the empirical posterior covariance plus a fixed, small covariance
# matrix epsilon which is also used for initial exploration.
# 
# Reference:
#   H. Haario, E. Saksman, and J. Tamminen, "An adaptive Metropolis algorithm",
#   Bernoulli 7(2): 223-242 (2001)
mutable struct AMProposal{FT <: Real, CT <: AbstractMatrix{FT}, 
                          MNT <: AbstractMvNormal} <: Proposal{MNT}
    epsilon::CT
    scalefactor::FT
    proposal::MNT
    μ::Vector{FT}
    M::Matrix{FT}
    δ::Vector{FT}
    N::Int
end

function AMProposal(epsilon::AbstractMatrix{FT}, 
                    scalefactor=FT(2.38^2 / size(epsilon, 1))) where {FT <: Real}
    sym = PDMat(Symmetric(epsilon))
    proposal = MvNormal(zeros(size(sym, 1)), sym)
    AMProposal(sym, scalefactor, proposal, zeros(size(sym,1)), zeros(size(sym)...), 
               zeros(size(sym,1)), 0)
end

logratio_proposal_density(::AMProposal, params_prev, params) = 0

# When the proposal is initialised the empirical posterior covariance is zero
function trackstep!(proposal::AMProposal, params)
    proposal.μ .= params
    proposal.M .= 0
    proposal.proposal = MvNormal(zeros(size(proposal.μ)), proposal.epsilon)
    proposal.N = 1
end

# Recompute the empirical posterior covariance matrix
function trackstep!(proposal::AMProposal, params, ::Union{Val{true}, Val{false}})
    proposal.N += 1
    proposal.δ .= params .- proposal.μ
    proposal.μ .+= proposal.δ ./ proposal.N

    mul!(proposal.M, params .- proposal.μ, proposal.δ', 1.0, 1.0)

    prop_cov = proposal.M .* proposal.scalefactor ./ proposal.N .+ proposal.epsilon
    proposal.proposal = MvNormal(mean(proposal.proposal), Symmetric(prop_cov))
end

function propose(rng::Random.AbstractRNG, p::AMProposal, ::DensityModel)
    return rand(rng, p.proposal)
end

function propose(rng::Random.AbstractRNG, p::AMProposal, ::DensityModel, t)
    return t + rand(rng, p.proposal)
end
