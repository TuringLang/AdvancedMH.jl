import .StructArrays: StructArray

# A basic chains constructor that works with the Transition struct we defined.
function AbstractMCMC.bundle_samples(
    rng::AbstractRNG, 
    model::DensityModel, 
    s::Metropolis, 
    N::Integer, 
    ts::Vector,
    chain_type::Type{StructArray}; 
    param_names=missing,
    kwargs...
)
    samples = AbstractMCMC.bundle_samples(rng, model, s, N, ts, Vector{NamedTuple};
                                          param_names=param_names, kwargs...)
    return StructArray(samples)
end

AbstractMCMC.chainscat(c::StructArray, cs::StructArray...) = vcat(c, cs...)