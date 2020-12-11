"""
    AdaptiveMvNormal(constant_component::MvNormal; σ=2.38, β=0.05)

Adaptive multivariate normal mixture proposal as described in Haario et al. and
Roberts & Rosenthal (2009). Uses a two-component mixture of MvNormal
distributions. One of the components (with mixture weight `β`) remains
constant, while the other component is adapted to the target covariance
structure. The proposal is initialized by providing the constant component to
the constructor. 

`σ` is the scale factor for the covariance matrix, where 2.38 is supposedly
optimal in a high-dimensional context according to Roberts & Rosenthal.

# References

- Haario, Heikki, Eero Saksman, and Johanna Tamminen. 
  "An adaptive Metropolis algorithm." Bernoulli 7.2 (2001): 223-242.
- Roberts, Gareth O., and Jeffrey S. Rosenthal. "Examples of adaptive MCMC."
  Journal of Computational and Graphical Statistics 18.2 (2009): 349-367.
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

function AdaptiveMvNormal(dist::MvNormal; σ=2.38, β=0.05)
    n = length(dist)
    adaptive = MvNormal(PDMat(Symmetric(dist.Σ)))
    AdaptiveMvNormal(n, -1, β, σ, dist, adaptive, zeros(n), zeros(n,n))
end

is_symmetric_proposal(::AdaptiveMvNormal) = true

"""
    adapt!(p::AdaptiveMvNormal, x::AbstractVector)

Adaptation for the adaptive multivariate normal mixture proposal as described
in Haario et al. (2001) and Roberts & Rosenthal (2009). Will perform an online
estimation of the target covariance matrix and mean. The code for this routine
is largely based on `Mamba.jl`.
"""
function adapt!(p::AdaptiveMvNormal, x::AbstractVector)
    p.n += 1
    # adapt mean vector and scatter matrix
    f = p.n / (p.n + 1.0)
    p.Ex = f * p.Ex + (1.0 - f) * x
    p.EX = f * p.EX + (1.0 - f) * x * x'
    # compute adapted covariance matrix
    Σ = (p.σ^2 / (p.d * f)) * (p.EX - p.Ex * p.Ex') 
    F = cholesky(Hermitian(Σ), check=false) 
    if rank(F.L) == p.d
        p.adaptive = MvNormal(PDMat(Σ, F))
    end
end

function Base.rand(rng::Random.AbstractRNG, p::AdaptiveMvNormal)
    return if p.n > 2p.d 
        p.β * rand(rng, p.constant) + (1.0 - p.β) * rand(rng, p.adaptive)
    else
        rand(rng, p.constant)
    end
end

function propose(rng::Random.AbstractRNG, proposal::AdaptiveMvNormal, m::DensityModel)
    return rand(rng, proposal)
end

function propose(
    rng::Random.AbstractRNG,
    proposal::AdaptiveMvNormal,
    model::DensityModel,
    t
)
    adapt!(proposal, t)
    return t + rand(rng, proposal)
end


