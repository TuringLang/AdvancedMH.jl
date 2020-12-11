# Define a (custom) Standard Normal distribution, for illustrative puspose.
struct StandardNormal <: Distributions.ContinuousUnivariateDistribution end
Distributions.logpdf(::StandardNormal, x::Real) = -(x ^ 2 + log(2 * pi)) / 2
Distributions.rand(rng::AbstractRNG, ::StandardNormal) = randn(Random.GLOBAL_RNG)
