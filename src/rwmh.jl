struct RandomWalk <: ProposalStyle end

"""
    RWMH(init_theta::Real, proposal = Normal(init_theta, 1))
    RWMH(init_theta::Vector{Real}, proposal = MvNormal(init_theta, 1))

Random walk Metropolis-Hastings.

Fields:

- `init_θ` is the vector form of the parameters needed for the likelihood function.
- `proposal` is a function that dynamically constructs a conditional distribution.

Example:

```julia
RWMH([0.0, 0.0], x -> MvNormal(x, 1.0))
````
"""
RWMH(init_theta::Real, proposal = Normal(init_theta, 1)) = MetropolisHastings(RandomWalk(), init_theta, proposal)
RWMH(init_theta::Vector{<:Real}, proposal = MvNormal(init_theta, 1)) = MetropolisHastings(RandomWalk(), init_theta, proposal)

# Define a function that makes a basic proposal depending on a univariate
# parameterization or a multivariate parameterization.
propose(spl::MetropolisHastings{RandomWalk}, model::DensityModel, θ::Real) = Transition(model, θ + rand(spl.proposal))
propose(spl::MetropolisHastings{RandomWalk}, model::DensityModel, θ::Vector{<:Real}) = Transition(model, θ + rand(spl.proposal))

"""
    q(θ::Real, dist::Sampleable)
    q(θ::Vector{<:Real}, dist::Sampleable)
    q(t1::Transition, dist::Sampleable)

Calculates the probability `q(θ | θcond)`, using the proposal distribution `spl.proposal`.
"""
@inline q(spl::MetropolisHastings{RandomWalk}, θ::Real, θcond::Real) = logpdf(spl.proposal, θ - θcond)
@inline q(spl::MetropolisHastings{RandomWalk}, θ::Vector{<:Real}, θcond::Vector{<:Real}) = logpdf(spl.proposal, θ - θcond)