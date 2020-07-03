import .StructArrays: StructArray

# A basic chains constructor that works with the Transition struct we defined.
function AbstractMCMC.bundle_samples(
    ts,
    model::DensityModel,
    sampler::MHSampler,
    state,
    chain_type::Type{StructArray};
    kwargs...
)
    samples = AbstractMCMC.bundle_samples(
        ts, model, sampler, state, Vector{NamedTuple};
        kwargs...
    )
    return StructArray(samples)
end

AbstractMCMC.chainscat(c::StructArray, cs::StructArray...) = vcat(c, cs...)
