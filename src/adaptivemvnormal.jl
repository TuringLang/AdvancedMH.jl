"""
    AdaptiveMvNormal(constant_component; σ=2.38, β=0.05)

An adaptive multivariate normal proposal. More precisely this uses a
two-component mixture (with weights `β` and `1 - β`) of multivariate normal
proposal distributions, where the low-weight component is constant (the
`constant` field and the high-weight component has its covariance matrix
adapted to the covariance structure of the target (the `adaptive` field).
"""
mutable struct AdaptiveMvNormal{T1,T2,V} <: Proposal{T1}
    d::Int  # dimensionality
    n::Int  # iteration
    β::Float64  # constant component mixture weight
    σ::Float64  # scale factor for adapted covariance matrix
    constant::T1  
    adaptive::T2
    Ex::Vector{V}  # rolling mean vector
    EX::Matrix{V}  # scatter matrix of previous draws
end

function AdaptiveMvNormal(dist; σ=2.38, β=0.05)
    n = length(dist)
    # the `adaptive` MvNormal part must have PDMat covariance matrix, not
    # ScalMat or PDDiagMat (as the `constant` part will have)
    adaptive = MvNormal(PDMat(Symmetric(dist.Σ)))
    AdaptiveMvNormal(n, -1, β, σ, dist, adaptive, zeros(n), zeros(n,n))
end

"""
    adapt!(p::AdaptiveMvNormal, x::AbstractVector)

Adaptation for the adaptive multivariate normal mixture proposal as described
in Haario et al. and Roberts & Rosenthal (2009). The code for this routine is
largely taken from `Mamba.jl`.

# References 

Roberts, Gareth O., and Jeffrey S. Rosenthal. "Examples of adaptive MCMC."
Journal of Computational and Graphical Statistics 18.2 (2009): 349-367.
"""
function adapt!(p::AdaptiveMvNormal, x::AbstractVector)
    p.n += 1
    # adapt mean vector and scatter matrix
    f = p.n / (p.n + 1.0)
    p.Ex = f * p.Ex + (1.0 - f) * x
    p.EX = f * p.EX + (1.0 - f) * x * x'
    # compute adapted covariance matrix
    Σ = (p.σ^2 / (p.d * f)) * (p.EX - p.Ex * p.Ex') 
    #F = cholesky(Hermitian(Σ), Val{true}(), check=false)
    F = cholesky(Hermitian(Σ), check=false) 
    if rank(F.L) == p.d
        #p.adaptive = MvNormal(PDMat(Symmetric(F.P * F.L)))
        p.adaptive = MvNormal(PDMat(Σ, F))
    end
end

function propose(rng::Random.AbstractRNG, p::AdaptiveMvNormal)
    return if p.n > 2p.d 
        p.β * rand(rng, p.constant) + (1.0 - p.β) * rand(rng, p.adaptive)
    else
        rand(rng, p.constant)
    end
end

function propose(rng::Random.AbstractRNG, p::AdaptiveMvNormal, m::DensityModel)
    return propose(rng, p)
end

function propose(
    rng::Random.AbstractRNG,
    proposal::AdaptiveMvNormal,
    model::DensityModel,
    t
)
    adapt!(proposal, t)
    return t + propose(rng, proposal)
end

# this is always symmetric, so ignore
function q(proposal::AdaptiveMvNormal, t, t_cond) 
    return 0.
end
