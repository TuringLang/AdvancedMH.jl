module AdvancedMH

# Import the relevant libraries.
using AbstractMCMC
using Distributions
using LinearAlgebra: I
using FillArrays: Zeros

using LogDensityProblems: LogDensityProblems

import Random

# Exports
export
    MetropolisHastings,
    DensityModel,
    RWMH,
    StaticMH,
    StaticProposal,
    SymmetricStaticProposal,
    RandomWalkProposal,
    SymmetricRandomWalkProposal,
    Ensemble,
    StretchProposal,
    MALA

# Reexports
export sample, MCMCThreads, MCMCDistributed, MCMCSerial

# Abstract type for MH-style samplers. Needs better name?
abstract type MHSampler <: AbstractMCMC.AbstractSampler end

# Abstract type for MH-style transitions.
abstract type AbstractTransition end

# Define a model type. Stores the log density function and the data to
# evaluate the log density on.
"""
    DensityModel{F} <: AbstractModel

`DensityModel` wraps around a self-contained log-liklihood function `logdensity`.

Example:

```julia
l(x) = logpdf(Normal(), x)
DensityModel(l)
```
"""
struct DensityModel{F} <: AbstractMCMC.AbstractModel
    logdensity :: F
end

const DensityModelOrLogDensityModel = Union{<:DensityModel,<:AbstractMCMC.LogDensityModel}

# Create a very basic Transition type, stores the
# parameter draws, the log probability of the draw,
# and the draw information until this point
struct Transition{T,L<:Real} <: AbstractTransition
    params :: T
    lp :: L
    accepted :: Bool
end

# Store the new draw, its log density, and draw information
Transition(model::DensityModelOrLogDensityModel, params, accepted) = Transition(params, logdensity(model, params), accepted)
function Transition(model::AbstractMCMC.LogDensityModel, params, accepted)
    return Transition(params, LogDensityProblems.logdensity(model.logdensity, params), accepted)
end

# Calculate the density of the model given some parameterization.
logdensity(model::DensityModel, params) = model.logdensity(params)
logdensity(::DensityModel, t::Transition) = t.lp
logdensity(model::AbstractMCMC.LogDensityModel, params) = LogDensityProblems.logdensity(model.logdensity, params)
logdensity(::AbstractMCMC.LogDensityModel, t::Transition) = t.lp

# A basic chains constructor that works with the Transition struct we defined.
function AbstractMCMC.bundle_samples(
    ts::Vector{<:AbstractTransition},
    model::DensityModelOrLogDensityModel,
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
    model::DensityModelOrLogDensityModel,
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

if !isdefined(Base, :get_extension)
    using Requires
end

function __init__()
    # Better error message if users forget to load ForwardDiff
    Base.Experimental.register_error_hint(MethodError) do io, exc, arg_types, kwargs
        if exc.f === logdensity_and_gradient && length(arg_types) == 2 && first(arg_types) <: DensityModel && isempty(kwargs)
            print(io, "\\nDid you forget to load ForwardDiff?")
        end
    end
    @static if !isdefined(Base, :get_extension)
        @require MCMCChains="c7f686f2-ff18-58e9-bc7b-31028e88f75d" include("../ext/AdvancedMHMCMCChainsExt.jl")
        @require StructArrays="09ab397b-f2b6-538f-b94a-2f83cf4a842a" include("../ext/AdvancedMHStructArraysExt.jl")
        @require DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5" begin
            @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" include("../ext/AdvancedMHForwardDiffExt.jl")
        end
    end
end

# AbstractMCMC.jl interface
function AbstractMCMC.getparams(t::Transition)
    return t.params
end

# TODO (sunxd): remove `DensityModel` in favor of `AbstractMCMC.LogDensityModel`
function AbstractMCMC.setparams!!(model::DensityModelOrLogDensityModel, t::Transition, params)
    return Transition(
        params,
        logdensity(model, params),
        t.accepted
    )
end

# Include inference methods.
include("proposal.jl")
include("mh-core.jl")
include("emcee.jl")
include("MALA.jl")

include("RobustAdaptiveMetropolis.jl")
using .RobustAdaptiveMetropolis
export RAM

end # module AdvancedMH
