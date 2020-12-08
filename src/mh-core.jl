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
struct MetropolisHastings{D} <: MHSampler
    proposal::D
end

StaticMH(d) = MetropolisHastings(StaticProposal(d))
RWMH(d) = MetropolisHastings(RandomWalkProposal(d))

# default function without RNG
propose(spl::MetropolisHastings, args...) = propose(Random.GLOBAL_RNG, spl, args...)

# Propose from a vector of proposals
function propose(
    rng::Random.AbstractRNG,
    spl::MetropolisHastings{<:AbstractArray},
    model::DensityModel
)
    proposal = map(p -> propose(rng, p, model), spl.proposal)
    return Transition(model, proposal)
end

function propose(
    rng::Random.AbstractRNG,
    spl::MetropolisHastings{<:AbstractArray},
    model::DensityModel,
    params_prev::Transition
)
    proposal = map(spl.proposal, params_prev.params) do p, params
        propose(rng, p, model, params)
    end
    return Transition(model, proposal)
end

# Make a proposal from one Proposal struct.
function propose(
    rng::Random.AbstractRNG,
    spl::MetropolisHastings{<:Proposal},
    model::DensityModel
)
    proposal = propose(rng, spl.proposal, model)
    return Transition(model, proposal)
end

function propose(
    rng::Random.AbstractRNG,
    spl::MetropolisHastings{<:Proposal},
    model::DensityModel,
    params_prev::Transition
)
    proposal = propose(rng, spl.proposal, model, params_prev.params)
    return Transition(model, proposal)
end

# Make a proposal from a NamedTuple of Proposal.
function propose(
    rng::Random.AbstractRNG,
    spl::MetropolisHastings{<:NamedTuple},
    model::DensityModel
)
    proposal = _propose(rng, spl.proposal, model)
    return Transition(model, proposal)
end

function propose(
    rng::Random.AbstractRNG,
    spl::MetropolisHastings{<:NamedTuple},
    model::DensityModel,
    params_prev::Transition
)
    proposal = _propose(rng, spl.proposal, model, params_prev.params)
    return Transition(model, proposal)
end

@generated function _propose(
    rng::Random.AbstractRNG,
    proposal::NamedTuple{names},
    model::DensityModel
) where {names}
    isempty(names) && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[:($name = propose(rng, proposal.$name, model)) for name in names]
    return expr
end

@generated function _propose(
    rng::Random.AbstractRNG,
    proposal::NamedTuple{names},
    model::DensityModel,
    params_prev::NamedTuple
) where {names}
    isempty(names) && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[
        :($name = propose(rng, proposal.$name, model, params_prev.$name)) for name in names
    ]
    return expr
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

transition(sampler, model, params) = transition(model, params)
transition(model, params) = Transition(model, params)

# Define the first sampling step.
# Return a 2-tuple consisting of the initial sample and the initial state.
# In this case they are identical.
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    spl::MHSampler;
    init_params=nothing,
    kwargs...
)
    if init_params === nothing
        transition = propose(rng, spl, model)
    else
        transition = AdvancedMH.transition(spl, model, init_params)
    end

    return transition, transition
end

# Define the other sampling steps.
# Return a 2-tuple consisting of the next sample and the the next state.
# In this case they are identical, and either a new proposal (if accepted)
# or the previous proposal (if not accepted).
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    spl::MHSampler,
    params_prev::AbstractTransition;
    kwargs...
)
    # Generate a new proposal.
    params = propose(rng, spl, model, params_prev)

    # Calculate the log acceptance probability.
    logα = compute_log_acceptance_prob(model, params, spl, params_prev)

    # Decide whether to return the previous params or the new one.
    if -Random.randexp(rng) < logα
        return params, params
    else
        return params_prev, params_prev
    end
end

"""
Type alias for symmetric distributions. This is not an exhaustive list. (e.g.
`Uniform` can be symmetric.)
"""
const SymmetricDistribution = Union{Normal,MvNormal,MvTDist,Cauchy,TDist}

"""
Type alias for symmetric random walk proposals 
"""
const SymmetricRandomWalkProposal = let
    Union{RandomWalkProposal{<:SymmetricDistribution},
          RandomWalkProposal{<:AbstractVector{<:SymmetricDistribution}}}
end

"""
Type alias for symmetric proposals.

NOTE to developers: Currently, this is `SymmetricRandomWalkProposal`, but new
proposals (e.g. an adaptive random walk proposal) can be added to the `Union`.
"""
const SymmetricProposal = Union{SymmetricRandomWalkProposal}

"""
Computes metropolis acceptance ratio (without Hastings).
"""
function compute_log_metropolis_ratio(
    model::DensityModel,
    params::T,
    spl::MetropolisHastings,
    params_prev::Transition
) where T
  return logdensity(model, params) - logdensity(model, params_prev)
end

"""
Computes log acceptance ratio for symmetric proposals. This is simply
the log Metropolis ratio.
"""
function compute_log_acceptance_prob(
    model::DensityModel,
    params::T,
    spl::MetropolisHastings{<:SymmetricProposal},
    params_prev::Transition) where T

    # Don't compute Hastings ratio for symmetric proposal distributions.
    return compute_log_metropolis_ratio(model, params, spl, params_prev)
end

"""
Computes log acceptance ratio for symmetric proposals. This is the 
full log-Metropolis-Hastings acceptance probability.
"""
function compute_log_acceptance_prob(
    model::DensityModel,
    params::T,
    spl::MetropolisHastings,
    params_prev::Transition) where T

    metropolis_ratio = compute_log_metropolis_ratio(model, params, spl, params_prev)
    hastings_ratio = q(spl, params_prev, params) - q(spl, params, params_prev)
    return metropolis_ratio + hastings_ratio
end
