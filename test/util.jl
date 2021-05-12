# Define a (custom) Standard Normal distribution, for illustrative puspose.
struct StandardNormal <: Distributions.ContinuousUnivariateDistribution end
Distributions.rand(rng::AbstractRNG, ::StandardNormal) = randn(Random.GLOBAL_RNG)
