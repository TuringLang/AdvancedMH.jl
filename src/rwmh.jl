struct RandomWalk <: ProposalStyle end

"""
    RWMH(init_theta::Real, proposal = Normal(init_theta, 1))
    RWMH(init_theta::Vector{Real}, proposal = MvNormal(init_theta, 1))

Random walk Metropolis-Hastings.

Fields:

- `init_params` is the vector form of the parameters needed for the likelihood function.
- `proposal` is a function that dynamically constructs a conditional distribution.

Example:

```julia
RWMH([0.0, 0.0], x -> MvNormal(x, 1.0))
````
"""
RWMH(init_theta, proposal) = MetropolisHastings(RandomWalk(), proposal, init_theta)
function RWMH(init_theta::Vector, proposal)
    if proposal isa Vector
        # Verify that there are proposal distributions for each parameter.
        length(proposal) == length(init_theta) || 
            throw("The number of proposal distributions must match the number of parameters.")
    end

    return MetropolisHastings(RandomWalk(), proposal, init_theta)
end

# Define a function that makes a basic proposal depending on a univariate
# parameterization or a multivariate parameterization.
propose(spl::MetropolisHastings{RandomWalk, <:Distribution}, model::DensityModel, t::Transition) = Transition(model, t.params + rand(spl.proposal))
function propose(spl::MetropolisHastings{RandomWalk, <:AbstractArray}, model::DensityModel, t::Transition)
    props = map(x -> x[2] + rand(x[1]), zip(spl.proposal, t.params))
    return Transition(model, props)
end

"""
    q(params::Real, dist::Sampleable)
    q(params::Vector{<:Real}, dist::Sampleable)
    q(t1::Transition, dist::Sampleable)

Calculates the probability `q(params | paramscond)`, using the proposal distribution `spl.proposal`.
"""
q(spl::MetropolisHastings{RandomWalk, <:Distribution}, t::Transition, t_cond::Transition) = logpdf(spl.proposal, t.params - t_cond.params)
function q(spl::MetropolisHastings{RandomWalk, <:AbstractArray}, t::Transition, t_cond::Transition)
    return sum(map(x -> logpdf(x[1], x[2] - x[3]), zip(spl.proposal, t.params, t_cond.params)))
end