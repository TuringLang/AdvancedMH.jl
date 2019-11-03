module AdvancedMH

# Import the relevant libraries.
using Reexport
using Turing.Interface
@reexport using Distributions
using Random

# Import specific functions and types to use or overload.
import Distributions: VariateForm, ValueSupport, variate_form, value_support, Sampleable
import MCMCChains: Chains
import Turing.Interface: step!, AbstractSampler, AbstractTransition, transition_type

# Exports
export MetropolisHastings, DensityModel, sample

"""
    MetropolisHastings{T, F<:Function}

Fields:

- `init_θ` is the vector form of the parameters needed for the likelihood function.
- `proposal` is a function that dynamically constructs a conditional distribution.

Example:

```julia
MetropolisHastings([0.0, 0.0], x -> MvNormal(x, 1.0))
````
"""
struct MetropolisHastings{T, F<:Function} <: AbstractSampler 
    init_θ :: T
    proposal :: F
end

# Default constructors.
MetropolisHastings(init_θ::Real) = MetropolisHastings(init_θ, x -> Normal(x, 1.0))
MetropolisHastings(init_θ::Vector{<:Real}) = MetropolisHastings(init_θ, x -> MvNormal(x, 1.0))

# Define a model type. Stores the log density function and the data to 
# evaluate the log density on.
"""
    DensityModel{V<:VariateForm, S<:ValueSupport, F<:Function} <: Sampleable{V, S}

`DensityModel` wraps around a self-contained log-liklihood function `ℓπ`.

Example:

```julia
l
DensityModel
```
"""
struct DensityModel{V<:VariateForm, S<:ValueSupport, F<:Function} <: Sampleable{V, S}
    ℓπ :: F
end

# Default density constructor.
DensityModel(ℓπ::F) where F = DensityModel{VariateForm, ValueSupport, F}(ℓπ)

# Create a very basic Transition type, only stores the 
# parameter draws and the log probability of the draw.
struct Transition{T<:Union{Vector{<:Real}, <:Real}, L<:Real} <: AbstractTransition
    θ :: T
    lp :: L
end

# Store the new draw and its log density.
Transition(model::M, θ::T) where {M<:DensityModel, T} = Transition(θ, ℓπ(model, θ))

# Tell the interface what transition type we would like to use.
transition_type(model::DensityModel, spl::MetropolisHastings) = typeof(Transition(spl.init_θ, ℓπ(model, spl.init_θ)))

# Define a function that makes a basic proposal depending on a univariate
# parameterization or a multivariate parameterization.
propose(spl::MetropolisHastings, model::DensityModel, θ::Real) = Transition(model, rand(spl.proposal(θ)))
propose(spl::MetropolisHastings, model::DensityModel, θ::Vector{<:Real}) = Transition(model, rand(spl.proposal(θ)))
propose(spl::MetropolisHastings, model::DensityModel, t::Transition) = propose(spl, model, t.θ)

"""
    q(spl::MetropolisHastings, θ1::Real, θ2::Real)
    q(spl::MetropolisHastings, θ1::Vector{<:Real}, θ2::Vector{<:Real})
    q(spl::MetropolisHastings, t1::Transition, t2::Transition)

Calculates the probability `q(θ1 | θ2)`, using the proposal distribution `spl.proposal`.
"""
q(spl::MetropolisHastings, θ1::Real, θ2::Real) = logpdf(spl.proposal(θ2), θ1)
q(spl::MetropolisHastings, θ1::Vector{<:Real}, θ2::Vector{<:Real}) = logpdf(spl.proposal(θ2), θ1)
q(spl::MetropolisHastings, t1::Transition, t2::Transition) = q(spl, t1.θ, t2.θ)

# Calculate the density of the model given some parameterization.
ℓπ(model::DensityModel, θ::T) where T = model.ℓπ(θ)
ℓπ(model::DensityModel, t::Transition) = t.lp

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
    return Transition(model, spl.init_θ)
end

# Define the other step functions. Returns a Transition containing
# either a new proposal (if accepted) or the previous proposal 
# (if not accepted).
function step!(
    rng::AbstractRNG,
    model::DensityModel,
    spl::MetropolisHastings,
    ::Integer,
    θ_prev::Transition;
    kwargs...
)
    # Generate a new proposal.
    θ = propose(spl, model, θ_prev)
    
    # Calculate the log acceptance probability.
    α = ℓπ(model, θ) - ℓπ(model, θ_prev) + q(spl, θ, θ_prev) - q(spl, θ_prev, θ)

    # Decide whether to return the previous θ or the new one.
    if log(rand(rng)) < min(α, 0.0)
        return θ
    else
        return θ_prev
    end
end

# A basic chains constructor that works with the Transition struct we defined.
function Chains(
    rng::AbstractRNG, 
    ℓ::DensityModel, 
    s::MetropolisHastings, 
    N::Integer, 
    ts::Vector{T}; 
    param_names=missing,
    kwargs...
) where {T <: Transition}
    # Turn all the transitions into a vector-of-vectors.
    vals = [vcat(t.θ, t.lp) for t in ts]

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = ["Parameter $i" for i in 1:(length(first(vals))-1)]
    end

    # Add the log density field to the parameter names.
    push!(param_names, "lp")

    # Bundle everything up and return a Chains struct.
    return Chains(vals, param_names, (internals=["lp"],))
end

end