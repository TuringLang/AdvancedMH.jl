module AdvancedMH

# Import the relevant libraries.
using Reexport
using AbstractMCMC
using Distributions
using Random

# Import specific functions and types to use or overload.
import MCMCChains: Chains
import AbstractMCMC: step!, AbstractSampler, AbstractTransition, transition_type, bundle_samples

# Exports
export MetropolisHastings, DensityModel, sample, psample, RWMH, StaticMH

# Abstract type for MH-style samplers.
abstract type Metropolis <: AbstractSampler end
abstract type ProposalStyle end

# Define a model type. Stores the log density function and the data to 
# evaluate the log density on.
"""
    DensityModel{F<:Function} <: AbstractModel

`DensityModel` wraps around a self-contained log-liklihood function `ℓπ`.

Example:

```julia
l
DensityModel
```
"""
struct DensityModel{F<:Function} <: AbstractModel
    ℓπ :: F
end

# Create a very basic Transition type, only stores the 
# parameter draws and the log probability of the draw.
struct Transition{T<:Union{Vector{<:Real}, <:Real}, L<:Real} <: AbstractTransition
    θ :: T
    lp :: L
end

# Store the new draw and its log density.
Transition(model::M, θ::T) where {M<:DensityModel, T} = Transition(θ, ℓπ(model, θ))

# Tell the interface what transition type we would like to use.
transition_type(model::DensityModel, spl::Metropolis) = typeof(Transition(spl.init_θ, ℓπ(model, spl.init_θ)))

# Calculate the density of the model given some parameterization.
ℓπ(model::DensityModel, θ::T) where T = model.ℓπ(θ)
ℓπ(model::DensityModel, t::Transition) = t.lp

# A basic chains constructor that works with the Transition struct we defined.
function bundle_samples(
    rng::AbstractRNG, 
    ℓ::DensityModel, 
    s::Metropolis, 
    N::Integer, 
    ts::Vector{T}; 
    param_names=missing,
    kwargs...
) where {ModelType<:AbstractModel, T<:AbstractTransition}
    # Turn all the transitions into a vector-of-vectors.
    vals = copy(reduce(hcat,[vcat(t.θ, t.lp) for t in ts])')

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = ["Parameter $i" for i in 1:length(s.init_θ)]
    else
        # Deepcopy to be thread safe.
        param_names = deepcopy(param_names)
    end

    # Add the log density field to the parameter names.
    push!(param_names, "lp")

    # Bundle everything up and return a Chains struct.
    return Chains(vals, param_names, (internals=["lp"],))
end

# Include inference methods.
include("mh-core.jl")
include("rwmh.jl")
include("staticmh.jl")

end # module AdvancedMH
