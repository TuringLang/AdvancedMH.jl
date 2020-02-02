"""
    MetropolisHastings{D}

`MetropolisHastings` has one field, `proposal`. 
`proposal` is a `Proposal`, `NamedTuple` of `Proposal`, or `Array{Proposal}` in the shape of your data.
For example, if you wanted the sampler to return a `NamedTuple` with shape

```julia
x = (a = 1.0, b=3.8)
```

The proposal would be

```julia
proposal = (a=Proposal(Static(), Normal(0,1)), b=Proposal(Static(), Normal(0,1)))
````

Other allowed proposal styles are

```
p1 = Proposal(Static(), Normal(0,1))
p2 = Proposal(Static(), [Normal(0,1), InverseGamma(2,3)])
p3 = Proposal(Static(), (a=Normal(0,1), b=InverseGamma(2,3)))
p4 = Proposal(Static(), (x=1.0) -> Normal(x, 1))
```

The sampler is constructed using

```julia
spl = MetropolisHastings(proposal)
```
"""
mutable struct MetropolisHastings{D} <: Metropolis
    proposal :: D
end

StaticMH(d) = MetropolisHastings(Proposal(Static(), d))
RWMH(d) = MetropolisHastings(Proposal(RandomWalk(), d))

# Propose from a vector of proposals
function propose(
    spl::MetropolisHastings{<:AbstractArray},
    model::DensityModel
)
    proposal = map(i -> propose(spl.proposal[i], model), 1:length(spl.proposal))
    return Transition(model, proposal)
end

function propose(
    spl::MetropolisHastings{<:AbstractArray},
    model::DensityModel,
    params_prev::Transition
)
    proposal = map(i -> propose(spl.proposal[i], model, params_prev.params[i]), 1:length(spl.proposal))
    return Transition(model, proposal)
end

# Make a proposal from one Proposal struct.
function propose(
    spl::MetropolisHastings{<:Proposal},
    model::DensityModel
)
    proposal = propose(spl.proposal, model)
    return Transition(model, proposal)
end

function propose(
    spl::MetropolisHastings{<:Proposal},
    model::DensityModel,
    params_prev::Transition
)
    proposal = propose(spl.proposal, model, params_prev.params)
    return Transition(model, proposal)
end

# Make a proposal from a NamedTuple of Proposal.
function propose(
    spl::MetropolisHastings{<:NamedTuple},
    model::DensityModel
)
    proposal = _propose(spl.proposal, model)
    return Transition(model, proposal)
end

@generated function _propose(
    proposals::NamedTuple{names},
    model::DensityModel
) where {names}
    expr = Expr(:tuple)
    map(names) do f
        push!(expr.args, Expr(:(=), f, :(propose(proposals.$f, model)) ))
    end
    return expr
end

function propose(
    spl::MetropolisHastings{<:NamedTuple},
    model::DensityModel,
    params_prev::Transition
)
    proposal = _propose(spl.proposal, model, params_prev.params)
    return Transition(model, proposal)
end

@generated function _propose(
    proposals::NamedTuple{names},
    model::DensityModel, 
    params_prev::NamedTuple
) where {names}
    expr = Expr(:tuple)
    map(names) do f
        push!(expr.args, Expr(:(=), f, :(propose(proposals.$f, model, params_prev.$f)) ))
    end
    return expr
end


# Evaluate the likelihood of t conditional on t_cond.
function q(
    spl::MetropolisHastings{<:AbstractArray},
    t::Transition,
    t_cond::Transition
)
    return sum(map(i -> q(spl.proposal[i], t.params[i], t_cond.params[i]), 1:length(spl.proposal)))
end

function q(
    spl::MetropolisHastings{<:Proposal},
    t::Transition,
    t_cond::Transition
)
    return q(spl.proposal, t.params, t_cond.params)
end

function q(
    spl::MetropolisHastings{<:NamedTuple},
    t::Transition,
    t_cond::Transition
)
    ks = keys(t.params)
    total = 0.0

    for k in ks
        total += q(spl.proposal[k], t.params[k], t_cond.params[k])
    end

    return total
end

# Define the first step! function, which is called at the 
# beginning of sampling. Return the initial parameter used
# to define the sampler.
function step!(
    rng::AbstractRNG,
    model::DensityModel,
    spl::MetropolisHastings,
    N::Integer;
    kwargs...
)
    return propose(spl, model)
end

# Define the other step functions. Returns a Transition containing
# either a new proposal (if accepted) or the previous proposal 
# (if not accepted).
function step!(
    rng::AbstractRNG,
    model::DensityModel,
    spl::MetropolisHastings,
    ::Integer,
    params_prev::Transition;
    kwargs...
)
    # Generate a new proposal.
    params = propose(spl, model, params_prev)

    # Calculate the log acceptance probability.
    α = logdensity(model, params) - logdensity(model, params_prev) + 
        q(spl, params_prev, params) - q(spl, params, params_prev)

    # Decide whether to return the previous params or the new one.
    if log(rand(rng)) < min(0.0, α)
        return params
    else
        return params_prev
    end
end