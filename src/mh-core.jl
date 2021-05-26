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
p3 = StaticProposal((a=Normal(0,1), b=InverseGamma(2,3)))
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

function propose(rng::Random.AbstractRNG, sampler::MHSampler, model::DensityModel)
    return propose(rng, sampler.proposal, model)
end
function propose(
    rng::Random.AbstractRNG,
    sampler::MHSampler,
    model::DensityModel,
    params_prev::Transition,
)
    return propose(rng, sampler.proposal, model, params_prev.params)
end

transition(sampler, model, params) = transition(model, params)
transition(model, params) = Transition(model, params)

# Define the first sampling step.
# Return a 2-tuple consisting of the initial sample and the initial state.
# In this case they are identical.
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::MHSampler;
    init_params=nothing,
    kwargs...
)
    params = init_params === nothing ? propose(rng, sampler, model) : init_params
    transition = AdvancedMH.transition(sampler, model, params)
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
    params_prev::Transition;
    kwargs...
)
    # Generate a new proposal.
    candidate = propose(rng, spl, model, params_prev)

    # Calculate the log acceptance probability.
    logdensity_candidate = logdensity(model, candidate)
    logα = logdensity_candidate - logdensity(model, params_prev) +
        logratio_proposal_density(spl, params_prev, candidate)

    # Decide whether to return the previous params or the new one.
    params = if -Random.randexp(rng) < logα
        Transition(candidate, logdensity_candidate)
    else
        params_prev
    end

    return params, params
end

function logratio_proposal_density(
    sampler::MetropolisHastings, params_prev::Transition, candidate
)
    return logratio_proposal_density(sampler.proposal, params_prev.params, candidate)
end
