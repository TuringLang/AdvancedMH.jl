struct Static <: ProposalStyle end

"""
    StaticMH(init_theta::Real, proposal = Normal(init_theta, 1))
    StaticMH(init_theta::Vector{Real}, proposal = MvNormal(init_theta, 1))

Static Metropolis-Hastings. Proposes only from the prior distribution.

Fields:

- `init_params` is the vector form of the parameters needed for the likelihood function.
- `proposal` is a distribution.

Example:

```julia
RWMH([0.0, 0.0], MvNormal(x, 1.0))
````
"""
StaticMH(init_theta, proposal) = MetropolisHastings(Static(), proposal, init_theta)
function StaticMH(init_theta::Vector, proposal)
    if proposal isa Vector
        # Verify that there are proposal distributions for each parameter.
        length(proposal) == length(init_theta) || 
            throw("The number of proposal distributions must match the number of parameters.")
    end

    return MetropolisHastings(Static(), proposal, init_theta)
end

# Define a function that makes a basic proposal depending on a univariate
# parameterization or a multivariate parameterization.
propose(spl::MetropolisHastings{Static, <:Distribution}, model::DensityModel, params::Transition) = Transition(model, rand(spl.proposal))
function propose(spl::MetropolisHastings{Static, <:AbstractArray}, model::DensityModel, params::Transition)
    props = map(rand, spl.proposal)
    return Transition(model, props)
end
function propose(spl::MetropolisHastings{Static, <:NamedTuple}, model::DensityModel, params::Transition)
    return Transition(model, _propose(spl.proposal))
end
@generated function _propose(proposals::NamedTuple{names}) where {names}
    expr = Expr(:tuple)
    map(names) do f
        push!(expr.args, Expr(:(=), f, :(rand(proposals.$f)) ))
    end
    return expr
end

"""
    q(params::Real, dist::Sampleable)
    q(params::Vector{<:Real}, dist::Sampleable)
    q(t1::Transition, dist::Sampleable)

Calculates the probability `q(params | paramscond)`, using the proposal distribution `spl.proposal`.
"""
function q(spl::MetropolisHastings{Static, <:Distribution}, t::Transition, t_cond::Transition)
    return logpdf(spl.proposal, t.params)
end

function q(spl::MetropolisHastings{Static, <:AbstractArray}, t::Transition, t_cond::Transition)
    return sum(map(x -> logpdf(x[1], x[2]), zip(spl.proposal, t.params)))
end

function q(spl::MetropolisHastings{Static, <:NamedTuple}, t::Transition, t_cond::Transition)
    total = 0.0
    for p in keys(t.params)
        if length(t.params[p]) == 1
            total += logpdf(spl.proposal[p], t.params[p][1])
        else
            total += logpdf(spl.proposal[p], t.params[p])
        end
    end
    return total
end

