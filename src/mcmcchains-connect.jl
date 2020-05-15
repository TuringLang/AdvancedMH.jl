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
    vals = copy(reduce(hcat,[vcat(t.params, t.lp) for t in ts])')

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = ["param_$i" for i in 1:length(s.init_params)]
    else
        # Deepcopy to be thread safe.
        param_names = deepcopy(param_names)
    end

    # Add the log density field to the parameter names.
    push!(param_names, "lp")

    # Bundle everything up and return a Chains struct.
    return Chains(vals, param_names, (internals=["lp"],))
end

# function AbstractMCMC.bundle_samples(
#     rng::Random.AbstractRNG, 
#     model::DensityModel, 
#     s::Ensemble, 
#     N::Integer, 
#     ts::Vector{<:Walker},
#     chain_type::Type{Chains}; 
#     param_names=missing,
#     kwargs...
# )
#     # return ts
#     vals = mapreduce(
#         t -> map(i -> vcat(ts[t].walkers[i].params, 
#                  ts[t].walkers[i].lp, t, i),
#                  1:length(ts[t].walkers)), 
#         vcat, 
#         1:length(ts))
    
#     vals = Array(reduce(hcat, vals)')

#     # return vals

#     # Check if we received any parameter names.
#     if ismissing(param_names)
#         param_names = ["param_$i" for i in 1:length(ts[1].walkers[1].params)]
#     else
#         # Deepcopy to be thread safe.
#         param_names = deepcopy(param_names)
#     end

#     # Add the log density field to the parameter names.
#     push!(param_names, "lp", "iteration", "walker")

#     # Bundle everything up and return a Chains struct.
#     return Chains(vals, param_names, (internals=["lp", "iteration", "walker"],))
# end

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
    # return ts
    vals = mapreduce(
        t -> map(i -> vcat(ts[t][i].params, 
                 ts[t][i].lp, t, i),
                 1:length(ts[t])), 
        vcat, 
        1:length(ts))
    
    vals = Array(reduce(hcat, vals)')

    # return vals

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = ["param_$i" for i in 1:length(ts[1][1].params)]
    else
        # Deepcopy to be thread safe.
        param_names = deepcopy(param_names)
    end

    # Add the log density field to the parameter names.
    push!(param_names, "lp", "iteration", "walker")

    # Bundle everything up and return a Chains struct.
    return Chains(vals, param_names, (internals=["lp", "iteration", "walker"],), sorted=true)
end