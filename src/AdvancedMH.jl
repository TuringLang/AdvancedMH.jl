module AdvancedMH

# Import the relevant libraries.
using AbstractMCMC
using Random
using Requires
using Distributions

# Import specific functions and types to use or overload.
import AbstractMCMC: step!, AbstractSampler, AbstractTransition, 
    transition_type, bundle_samples

# Exports
export MetropolisHastings, DensityModel, sample, psample, RWMH, StaticMH, Proposal, Static, RandomWalk

# Abstract type for MH-style samplers.
abstract type Metropolis <: AbstractSampler end
abstract type ProposalStyle end

struct RandomWalk <: ProposalStyle end
struct Static <: ProposalStyle end

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
struct DensityModel{F<:Function} <: AbstractModel
    logdensity :: F
end

# Create a very basic Transition type, only stores the 
# parameter draws and the log probability of the draw.
struct Transition{T<:Union{Vector, Real, NamedTuple}, L<:Real} <: AbstractTransition
    params :: T
    lp :: L
end

# Store the new draw and its log density.
Transition(model::M, params::T) where {M<:DensityModel, T} = Transition(params, logdensity(model, params))

# Tell the interface what transition type we would like to use.
transition_type(model::DensityModel, spl::Metropolis) = typeof(Transition(spl.init_params, logdensity(model, spl.init_params)))

# Calculate the density of the model given some parameterization.
logdensity(model::DensityModel, params) = model.logdensity(params)
logdensity(model::DensityModel, t::Transition) = t.lp

# A basic chains constructor that works with the Transition struct we defined.
function bundle_samples(
    rng::AbstractRNG, 
    model::DensityModel, 
    s::Metropolis, 
    N::Integer, 
    ts::Type{Any}; 
    param_names=missing,
    kwargs...
) where {ModelType<:AbstractModel, T<:AbstractTransition}
    return ts
end

function bundle_samples(
    rng::AbstractRNG, 
    model::DensityModel, 
    s::Metropolis, 
    N::Integer, 
    ts::Vector{T},
    chain_type::Type{Vector{NamedTuple}}; 
    param_names=missing,
    kwargs...
) where {ModelType<:AbstractModel, T<:AbstractTransition}
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

function __init__()
    @require MCMCChains="c7f686f2-ff18-58e9-bc7b-31028e88f75d" include("mcmcchains-connect.jl")
    @require StructArrays="09ab397b-f2b6-538f-b94a-2f83cf4a842a" include("structarray-connect.jl")
end

# Include inference methods.
include("proposal.jl")
include("mh-core.jl")

end # module AdvancedMH
