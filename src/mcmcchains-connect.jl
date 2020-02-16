import .MCMCChains: Chains

# A basic chains constructor that works with the Transition struct we defined.
function AbstractMCMC.bundle_samples(
    rng::AbstractRNG, 
    model::DensityModel, 
    s::Metropolis, 
    N::Integer, 
    ts::Vector,
    chain_type::Type{Chains}; 
    param_names=missing,
    kwargs...
)
    # Turn all the transitions into a vector-of-vectors.
    vals = copy(reduce(hcat,[vcat(t.params, t.lp) for t in ts])')

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = ["Parameter $i" for i in 1:length(s.init_params)]
    else
        # Deepcopy to be thread safe.
        param_names = deepcopy(param_names)
    end

    # Add the log density field to the parameter names.
    push!(param_names, "lp")

    # Bundle everything up and return a Chains struct.
    return Chains(vals, param_names, (internals=["lp"],))
end