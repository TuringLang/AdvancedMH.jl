mutable struct Adaptor
    accepted::Integer
    total::Integer
    tune::Integer     # tuning interval
    target::Float64   # target acceptance rate
    bound::Float64    # bound on logσ of Gaussian kernel
    δmax::Float64     # maximum adaptation step
end

function Adaptor(; tune=25, target=0.44, bound=10., δmax=0.2)
    return Adaptor(0, 0, tune, target, bound, δmax)
end

"""
    AdaptiveProposal{P} 

An adaptive Metropolis-Hastings proposal. In order for this to work, the
proposal kernel should implement the `adapted(proposal, δ)` method, where `δ`
is the increment/decrement applied to the scale of the proposal distribution
during adaptation (e.g. for a Normal distribution the scale is `log(σ)`, so
that after adaptation the proposal is `Normal(0, exp(log(σ) + δ))`).

# Example
```julia
julia>  p = AdaptiveProposal(Uniform(-0.2, 0.2));

julia> rand(p)
0.07975590594518434
```

# References 

Roberts, Gareth O., and Jeffrey S. Rosenthal. "Examples of adaptive MCMC."
Journal of Computational and Graphical Statistics 18.2 (2009): 349-367.
"""
mutable struct AdaptiveProposal{P} <: Proposal{P}
    proposal::P
    adaptor::Adaptor
end

function AdaptiveProposal(p; kwargs...) 
    return AdaptiveProposal(p, Adaptor(; kwargs...))
end

accepted!(p::AdaptiveProposal) = p.adaptor.accepted += 1
accepted!(p::Vector{<:AdaptiveProposal}) = map(accepted!, p)
accepted!(p::NamedTuple{names}) where names = map(x->accepted!(getfield(p, x)), names)

# this is defined because the first draw has no transition yet (I think)
function propose(rng::Random.AbstractRNG, p::AdaptiveProposal, m::DensityModel)
    return rand(rng, p.proposal)
end

# the actual proposal happens here
function propose(
    rng::Random.AbstractRNG,
    proposal::AdaptiveProposal{<:Union{Distribution,Proposal}},
    model::DensityModel,
    t
)
    consider_adaptation!(proposal) 
    return t + rand(rng, proposal.proposal)
end

function q(proposal::AdaptiveProposal, t, t_cond) 
    logpdf(proposal, t - t_cond)
end

function consider_adaptation!(p)
    (p.adaptor.total % p.adaptor.tune == 0) && adapt!(p)
    p.adaptor.total += 1
end

function adapt!(p::AdaptiveProposal)
    a = p.adaptor
    a.total == 0 && return 
    δ  = min(a.δmax, 1. /√(a.total/a.tune))  # diminishing adaptation
    α  = a.accepted / a.tune  # acceptance ratio
    p_ = adapted(p.proposal, α > a.target ? δ : -δ, a.bound) 
    a.accepted = 0 
    p.proposal = p_
end

function adapted(d::Normal, δ, bound=Inf)
    lσ = log(d.σ) + δ
    lσ = abs(lσ) > bound ? sign(lσ) * bound : lσ
    Normal(d.μ, exp(lσ))
end

function adapted(d::Uniform, δ, bound=Inf)
    lσ = log(d.b) + δ
    σ  = abs(lσ) > bound ? exp(sign(lσ) * bound) : exp(lσ)
    Uniform(-σ, σ)
end
