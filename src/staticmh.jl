struct Static <: ProposalStyle end

"""
    StaticMH(init_theta::Real, proposal = Normal(init_theta, 1))
    StaticMH(init_theta::Vector{Real}, proposal = MvNormal(init_theta, 1))

Static Metropolis-Hastings. Proposes only from the prior distribution.

Fields:

- `init_θ` is the vector form of the parameters needed for the likelihood function.
- `proposal` is a distribution.

Example:

```julia
RWMH([0.0, 0.0], MvNormal(x, 1.0))
````
"""
StaticMH(init_theta::Real, proposal = Normal(init_theta, 1)) = MetropolisHastings(Static(), init_theta, proposal)
StaticMH(init_theta::Vector{<:Real}, proposal = MvNormal(init_theta, 1)) = MetropolisHastings(Static(), init_theta, proposal)

# Define a function that makes a basic proposal depending on a univariate
# parameterization or a multivariate parameterization.
propose(spl::MetropolisHastings{Static}, model::DensityModel, θ::Real) = Transition(model, rand(spl.proposal))
propose(spl::MetropolisHastings{Static}, model::DensityModel, θ::Vector{<:Real}) = Transition(model, rand(spl.proposal))

"""
    q(θ::Real, dist::Sampleable)
    q(θ::Vector{<:Real}, dist::Sampleable)
    q(t1::Transition, dist::Sampleable)

Calculates the probability `q(θ | θcond)`, using the proposal distribution `spl.proposal`.
"""
q(spl::MetropolisHastings{Static}, θ::Real, θcond::Real) = logpdf(spl.proposal, θ)
q(spl::MetropolisHastings{Static}, θ::Vector{<:Real}, θcond::Vector{<:Real}) = logpdf(spl.proposal, θ)

