module AdvancedMH

# Import the relevant libraries.
using AbstractMCMC
using Distributions
using Requires

import Random

# Exports
export MetropolisHastings, DensityModel, RWMH, StaticMH, StaticProposal, 
    RandomWalkProposal, Ensemble, StretchProposal, MALA

# Reexports
export sample, MCMCThreads, MCMCDistributed

# Abstract type for MH-style samplers. Needs better name? 
abstract type MHSampler <: AbstractMCMC.AbstractSampler end

# Abstract type for MH-style transitions.
abstract type AbstractTransition end

# Define a model type. Stores the log density function and the data to 
# evaluate the log density on.
"""
    DensityModel{F<:Function} <: AbstractModel

`DensityModel` wraps around a self-contained log-liklihood function `logdensity`.

Example:

```julia
l
DensityModel
```
"""
struct DensityModel{F<:Function} <: AbstractMCMC.AbstractModel
    logdensity :: F
end

# Create a very basic Transition type, only stores the
# parameter draws and the log probability of the draw.
struct Transition{T<:Union{Vector, Real, NamedTuple}, L<:Real} <: AbstractTransition
    params :: T
    lp :: L
end

# Store the new draw and its log density.
Transition(model::DensityModel, params) = Transition(params, logdensity(model, params))

# Calculate the density of the model given some parameterization.
logdensity(model::DensityModel, params) = model.logdensity(params)
logdensity(model::DensityModel, t::Transition) = t.lp

# A basic chains constructor that works with the Transition struct we defined.
function AbstractMCMC.bundle_samples(
    ts::Vector{<:AbstractTransition},
    model::DensityModel,
    sampler::MHSampler,
    state,
    chain_type::Type{Vector{NamedTuple}};
    param_names=missing,
    kwargs...
)
    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = ["param_$i" for i in 1:length(keys(ts[1].params))]
    else
        # Deepcopy to be thread safe.
        param_names = deepcopy(param_names)
    end

    push!(param_names, "lp")

    # Turn all the transitions into a vector-of-NamedTuple.
    ks = tuple(Symbol.(param_names)...)
    nts = [NamedTuple{ks}(tuple(t.params..., t.lp)) for t in ts]

    return nts
end

function AbstractMCMC.bundle_samples(
    ts::Vector{<:Transition{<:NamedTuple}},
    model::DensityModel,
    sampler::MHSampler,
    state,
    chain_type::Type{Vector{NamedTuple}};
    param_names=missing,
    kwargs...
)
    # If the element type of ts is NamedTuples, just use the names in the
    # struct.

    # Extract NamedTuples
    nts = map(x -> merge(x.params, (lp=x.lp,)), ts)

    # Return em'
    return nts
end

function __init__()
    @require MCMCChains="c7f686f2-ff18-58e9-bc7b-31028e88f75d" include("mcmcchains-connect.jl")
    @require StructArrays="09ab397b-f2b6-538f-b94a-2f83cf4a842a" include("structarray-connect.jl")
    @require DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5" begin
        @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" include("MALA.jl")
    end
end

# Include inference methods.
include("proposal.jl")
include("mh-core.jl")
include("emcee.jl")

end # module AdvancedMH
