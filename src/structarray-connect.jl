using StructArrays

# A basic chains constructor that works with the Transition struct we defined.
function bundle_samples(
    rng::AbstractRNG, 
    model::DensityModel, 
    s::Metropolis, 
    N::Integer, 
    ts::Vector{T},
    chain_type::Type{StructArray}; 
    param_names=missing,
    kwargs...
) where {ModelType<:AbstractModel, T<:AbstractTransition}
    return StructArray(bundle_samples(rng, model, s, N, ts, NamedTuple; param_names=param_names, kwargs...))
end