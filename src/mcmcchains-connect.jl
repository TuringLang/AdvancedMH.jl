import .MCMCChains: Chains

# A basic chains constructor that works with the Transition struct we defined.
function AbstractMCMC.bundle_samples(
    ts::Vector{<:AbstractTransition},
    model::DensityModel,
    sampler::MHSampler,
    state,
    chain_type::Type{Chains};
    param_names=missing,
    kwargs...
)
    # Turn all the transitions into a vector-of-vectors.
    vals = [vcat(t.params, t.lp) for t in ts]

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = [Symbol(:param_, i) for i in 1:length(keys(ts[1].params))]
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
    ts::Vector{<:Transition{<:NamedTuple}},
    model::DensityModel,
    sampler::MHSampler,
    state,
    chain_type::Type{Chains};
    param_names=missing,
    kwargs...
)
    # Convert to a Vector{NamedTuple} first
    nts = AbstractMCMC.bundle_samples(ts, model, sampler, state, Vector{NamedTuple}; param_names=param_names, kwargs...)

    # Get all the keys
    all_keys = unique(mapreduce(collect∘keys, vcat, nts))

    # Preallocate array
    # vals = []

    # Push linearized draws onto array
    trygetproperty(thing, key) = key in keys(thing) ? getproperty(thing, key) : missing
    vals = map(nt -> [trygetproperty(nt, k) for k in all_keys], nts)

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = all_keys
    else
        # Generate new array to be thread safe.
        param_names = Symbol.(param_names)
    end

    # Bundle everything up and return a Chains struct.
    return Chains(vals, param_names, (internals = [:lp],))
end

function AbstractMCMC.bundle_samples(
    ts::Vector{<:Vector{<:AbstractTransition}},
    model::DensityModel,
    sampler::Ensemble,
    state,
    chain_type::Type{Chains};
    param_names=missing,
    kwargs...
)
    # Preallocate return array
    # NOTE: requires constant dimensionality.
    n_params = length(ts[1][1].params)
    nsamples = length(ts)
    # add 1 parameter for lp
    vals = Array{Float64, 3}(undef, nsamples, n_params + 1, sampler.n_walkers)

    for n in 1:nsamples
        for i in 1:sampler.n_walkers
            walker = ts[n][i]
            for j in 1:n_params
                vals[n, j, i] = walker.params[j]
            end
            vals[n, n_params + 1, i] = walker.lp
        end
    end

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = [Symbol(:param_, i) for i in 1:length(keys(ts[1][1].params))]
    else
        # Generate new array to be thread safe.
        param_names = Symbol.(param_names)
    end

    # Add the log density field to the parameter names.
    push!(param_names, :lp)

    # Bundle everything up and return a Chains struct.
    return Chains(vals, param_names, (internals=[:lp],))
end
