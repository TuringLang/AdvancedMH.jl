struct Ensemble{D} <: AMH
    n_walkers::Int
    proposal::D
end

struct Walker{T<:Union{Vector,Real,NamedTuple},L<:Real}
    params::T
    lp::L
end

Walker(t::Transition) = Walker(t.params, t.lp)
Walker(model::DensityModel, params) = Walker(params, logdensity(model, params))

# Define the first step! function, which is called at the 
# beginning of sampling. Return the initial parameter used
# to define the sampler.
function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::DensityModel,
    spl::Ensemble,
    N::Integer,
    ::Nothing;
    init_params = nothing,
    kwargs...,
)
    if init_params === nothing
        return propose(spl, model)
    else
        return Transition(model, init_params)
    end
end

# Define the other step functions. Returns a Transition containing
# either a new proposal (if accepted) or the previous proposal 
# (if not accepted).
function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::DensityModel,
    spl::Ensemble,
    ::Integer,
    params_prev;
    kwargs...,
)
    # Generate a new proposal. Accept/reject happens at proposal level.
    return propose(spl, model, params_prev)
end

#
# Initial proposal
# 
function propose(spl::Ensemble, model::DensityModel)
    # Make the first proposal with a static draw from the prior.
    static_prop = Proposal(Static(), spl.proposal.proposal)
    mh_spl = MetropolisHastings(static_prop)
    return map(x -> Walker(propose(mh_spl, model)), 1:spl.n_walkers)
end



#
# Every other proposal
# 
function propose(spl::Ensemble, model::DensityModel, walkers::Vector{W}) where {W<:Walker}
    new_walkers = Vector{W}(undef, spl.n_walkers)
    interval = 1:spl.n_walkers

    for i in interval
        walker = walkers[i]
        other_walker = rand(walkers[interval.!=i])
        new_walkers[i] = move(spl, model, walker, other_walker)
    end

    return new_walkers
end


#####################################
# Basic stretch move implementation #
#####################################
struct Stretch{F<:AbstractFloat} <: ProposalStyle
    stretch_length::F
end

Stretch() = Stretch(2.0)

function move(
    # spl::Ensemble,
    spl::Ensemble{Proposal{T,P}},
    model::DensityModel,
    walker::Walker,
    other_walker::Walker,
) where {T<:Stretch,P}
    # Calculate intermediate values
    proposal = spl.proposal
    n = length(walker.params)
    a = proposal.type.stretch_length
    z = ((a - 1.0) * rand() + 1.0)^2.0 / a
    alphamult = (n - 1) * log(z)

    # Make new parameters
    y = walker.params + z .* (other_walker.params - walker.params)

    # Construct a new walker
    new_walker = Walker(model, y)

    # Calculate accept/reject value.
    alpha = alphamult + new_walker.lp - walker.lp

    if alpha >= log(rand())
        return new_walker
    else
        return walker
    end
end

#########################
# Elliptical slice step #
#########################

struct EllipticalSlice{E} <: ProposalStyle
    ellipse::E
end

function move(
    # spl::Ensemble,
    spl::Ensemble{Proposal{T,P}},
    model::DensityModel,
    walker::Walker,
    other_walker::Walker,
) where {T<:EllipticalSlice,P}
    # Calculate intermediate values
    proposal = spl.proposal
    n = length(walker.params)
    nu = rand(proposal.type.ellipse)

    u = rand()
    y = walker.lp + log(u)

    theta_min = 0.0
    theta_max = 2.0*π
    theta = rand(Uniform(theta_min, theta_max))

    theta_min = theta - 2.0*π
    theta_max = theta
    
    f = walker.params
    while true
        ctheta = cos(theta)
        stheta = sin(theta)

        f_prime = f .* ctheta + nu .* stheta

        new_walker = Walker(model, f_prime)

        if new_walker.lp > y
            return new_walker
        else
            if theta < 0 
                theta_min = theta
            else
                theta_max = theta
            end

            theta = rand(Uniform(theta_min, theta_max))
        end
    end 
end

#####################
# Slice and stretch #
#####################
struct EllipticalSliceStretch{E, S<:Stretch} <: ProposalStyle
    ellipse::E
    stretch::S
end

EllipticalSliceStretch(e) = EllipticalSliceStretch(e, Stretch(2.0))

function move(
    # spl::Ensemble,
    spl::Ensemble{Proposal{T,P}},
    model::DensityModel,
    walker::Walker,
    other_walker::Walker,
) where {T<:EllipticalSliceStretch,P}
    # Calculate intermediate values
    proposal = spl.proposal
    n = length(walker.params)
    nu = rand(proposal.type.ellipse)

    # Calculate stretch step first
    subspl = Ensemble(spl.n_walkers, Proposal(proposal.type.stretch, proposal.proposal))
    walker = move(subspl, model, walker, other_walker)

    u = rand()
    y = walker.lp + log(u)

    theta_min = 0.0
    theta_max = 2.0*π
    theta = rand(Uniform(theta_min, theta_max))

    theta_min = theta - 2.0*π
    theta_max = theta
    
    f = walker.params

    i = 0
    while true
        i += 1
        
        ctheta = cos(theta)
        stheta = sin(theta)
        
        f_prime = f .* ctheta + nu .* stheta

        if i >100
            @warn "Rejecting slice sample"
            return walker
        end

        new_walker = Walker(model, f_prime)

        # @info "Slice step" i f f_prime y new_walker.lp theta theta_max theta_min

        if new_walker.lp > y
            return new_walker
        else
            if theta < 0 
                theta_min = theta
            else
                theta_max = theta
            end

            theta = rand(Uniform(theta_min, theta_max))
        end
    end 
end