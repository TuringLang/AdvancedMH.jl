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
proposal = (a=StaticProposal(Normal(0,1)), b=StaticProposal(Normal(0,1)))
````

Other allowed proposals are

```
p1 = StaticProposal(Normal(0,1))
p2 = StaticProposal([Normal(0,1), InverseGamma(2,3)])
p3 = StaticProposal(a=Normal(0,1), b=InverseGamma(2,3))
p4 = StaticProposal((x=1.0) -> Normal(x, 1))
```

The sampler is constructed using

```julia
spl = MetropolisHastings(proposal)
```

When using `MetropolisHastings` with the function `sample`, the following keyword
arguments are allowed:

- `init_params` defines the initial parameterization for your model. If
none is given, the initial parameters will be drawn from the sampler's proposals.
- `param_names` is a vector of strings to be assigned to parameters. This is only
used if `chain_type=Chains`.
- `chain_type` is the type of chain you would like returned to you. Supported
types are `chain_type=Chains` if `MCMCChains` is imported, or 
`chain_type=StructArray` if `StructArrays` is imported.
"""
struct MetropolisHastings{D} <: Metropolis
    proposal::D
end

StaticMH(d) = MetropolisHastings(StaticProposal(d))
RWMH(d) = MetropolisHastings(RandomWalkProposal(d))

# Propose from a vector of proposals
function propose(
    spl::MetropolisHastings{<:AbstractArray},
    model::DensityModel
)
    proposal = map(p -> propose(p, model), spl.proposal)
    return Transition(model, proposal)
end

function propose(
    spl::MetropolisHastings{<:AbstractArray},
    model::DensityModel,
    params_prev::Transition
)
    proposal = map(spl.proposal, params_prev.params) do p, params
        propose(p, model, params)
    end
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
    proposal = map(spl.proposal) do p
        propose(p, model)
    end
    return Transition(model, proposal)
end

function propose(
    spl::MetropolisHastings{<:NamedTuple},
    model::DensityModel,
    params_prev::Transition
)
    proposal = map(spl.proposal, params_prev.params) do p, params
        propose(p, model, params)
    end
    return Transition(model, proposal)
end

# Evaluate the likelihood of t conditional on t_cond.
function q(
    spl::MetropolisHastings{<:AbstractArray},
    t::Transition,
    t_cond::Transition
)
    # mapreduce with multiple iterators requires Julia 1.2 or later
    return mapreduce(+, 1:length(spl.proposal)) do i
        q(spl.proposal[i], t.params[i], t_cond.params[i])
    end
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
    # mapreduce with multiple iterators requires Julia 1.2 or later
    return mapreduce(+, keys(t.params)) do k
        q(spl.proposal[k], t.params[k], t_cond.params[k])
    end
end

# Define the first step! function, which is called at the 
# beginning of sampling. Return the initial parameter used
# to define the sampler.
function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::DensityModel,
    spl::MetropolisHastings,
    N::Integer,
    ::Nothing;
    init_params=nothing,
    kwargs...
)
    if init_params === nothing
        return propose(spl, model)
    else
        return Transition(model, init_params)
    end
end

# Define the other step functions. Returns a Transition containing
# either a new proposal (if accepted) or the previous proposal 
# (if not accepted).
function AbstractMCMC.step!(
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
    logα = logdensity(model, params) - logdensity(model, params_prev) + 
        q(spl, params_prev, params) - q(spl, params, params_prev)

    # Decide whether to return the previous params or the new one.
    if -Random.randexp(rng) < logα
        return params
    else
        return params_prev
    end
end