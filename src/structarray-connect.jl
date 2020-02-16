import .StructArrays: StructArray

# A basic chains constructor that works with the Transition struct we defined.
function bundle_samples(
    rng::AbstractRNG, 
    model::DensityModel, 
    s::Metropolis, 
    N::Integer, 
    ts::Vector{<:AbstractTransition},
    chain_type::Type{StructArray}; 
    param_names=missing,
    kwargs...
)
    return StructArray(bundle_samples(rng, model, s, N, ts, Vector{NamedTuple}; param_names=param_names, kwargs...))
end

AbstractMCMC.chainscat(cs::StructArray...) = [cs[i] for i in 1:length(cs)]