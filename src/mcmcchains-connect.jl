import .MCMCChains: Chains

# A basic chains constructor that works with the Transition struct we defined.
function AbstractMCMC.bundle_samples(
    rng::Random.AbstractRNG, 
    model::DensityModel, 
    s::MHSampler, 
    N::Integer, 
    ts,
    chain_type::Type{Chains}; 
    param_names=missing,
    kwargs...
)
    # Turn all the transitions into a vector-of-vectors.
    vals = [vcat(t.params, t.lp) for t in ts]

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = [Symbol(:param_, i) for i in 1:length(s.init_params)]
    else
        # Generate new array to be thread safe.
        param_names = Symbol.(param_names)
    end

    # Add the log density field to the parameter names.
    push!(param_names, :lp)

    # Bundle everything up and return a Chains struct.
    return Chains(vals, param_names, (internals = [:lp],))
end

function AbstractMCMC.bundle_samples(
    rng::Random.AbstractRNG, 
    model::DensityModel, 
    s::Ensemble, 
    N::Integer, 
    ts::Vector,
    chain_type::Type{Chains}; 
    param_names=missing,
    kwargs...
)
    # Preallocate return array
    # NOTE: requires constant dimensionality.
    n_params = length(ts[1][1].params)
    vals = Array{Float64, 3}(undef, N, n_params + 1, s.n_walkers)  # add 1 parameter for lp

    for n in 1:N
        for i in 1:s.n_walkers
            walker = ts[n][i]
            for j in 1:n_params
                vals[n, j, i] = walker.params[j]
            end
            vals[n, n_params + 1, i] = walker.lp
        end
    end

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = [Symbol(:param_, i) for i in 1:length(ts[1][1].params)]
    else
        # Generate new array to be thread safe.
        param_names = Symbol.(param_names)
    end

    # Add the log density field to the parameter names.
    push!(param_names, :lp)

    # Bundle everything up and return a Chains struct.
    return Chains(vals, param_names, (internals=[:lp],))
end
