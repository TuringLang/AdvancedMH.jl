# Define a custom Normal distribution, for illustrative puspose.
struct CustomNormal{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    m::T
end
CustomNormal() = CustomNormal(0)

Distributions.rand(rng::AbstractRNG, d::CustomNormal) = d.m + randn(rng)
