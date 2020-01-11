# AdvancedMH.jl

AdvancedMH.jl currently provides a robust implementation of random walk Metropolis-Hastings samplers.

Further development aims to provide a suite of adaptive Metropolis-Hastings implementations.

Currently there are two sampler types. The first is `RWMH`, which represents random-walk MH sampling, and the second is `StaticMH`, which draws proposals
from only a prior distribution without incrementing the previous sample.

## Usage

AdvancedMH works by accepting some log density function which is used to construct a `DensityModel`. The `DensityModel` is then used in a `sample` call.

```julia
# Import the package.
using AdvancedMH
using Distributions

# Generate a set of data from the posterior we want to estimate.
data = rand(Normal(0, 1), 30)

# Define the components of a basic model.
insupport(θ) = θ[2] >= 0
dist(θ) = Normal(θ[1], θ[2])
density(θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf

# Construct a DensityModel.
model = DensityModel(density)

# Set up our sampler with initial parameters.
spl = RWMH([0.0, 0.0])

# Sample from the posterior.
chain = sample(model, spl, 100000; param_names=["μ", "σ"])
```

Output:

```julia
Object of type Chains, with data of type 100000×3×1 Array{Float64,3}

Iterations        = 1:100000
Thinning interval = 1
Chains            = 1
Samples per chain = 100000
internals         = lp
parameters        = μ, σ

2-element Array{MCMCChains.ChainDataFrame,1}

Summary Statistics

│ Row │ parameters │ mean      │ std      │ naive_se   │ mcse       │ ess     │ r_hat   │
│     │ Symbol     │ Float64   │ Float64  │ Float64    │ Float64    │ Any     │ Any     │
├─────┼────────────┼───────────┼──────────┼────────────┼────────────┼─────────┼─────────┤
│ 1   │ μ          │ 0.0834188 │ 0.241418 │ 0.00076343 │ 0.00341067 │ 4693.04 │ 1.00008 │
│ 2   │ σ          │ 1.33116   │ 0.184111 │ 0.00058221 │ 0.00258778 │ 4965.83 │ 1.00001 │

Quantiles

│ Row │ parameters │ 2.5%      │ 25.0%      │ 50.0%     │ 75.0%    │ 97.5%    │
│     │ Symbol     │ Float64   │ Float64    │ Float64   │ Float64  │ Float64  │
├─────┼────────────┼───────────┼────────────┼───────────┼──────────┼──────────┤
│ 1   │ μ          │ -0.393769 │ -0.0771134 │ 0.0801688 │ 0.241162 │ 0.564331 │
│ 2   │ σ          │ 1.03685   │ 1.2044     │ 1.30992   │ 1.43609  │ 1.75745  │
```

## Custom proposals

Custom proposal distributions can be specified by passing a distribution to `MetropolisHastings`:

```julia
# Set up our sampler with initial parameters.
spl1 = RWMH([0.0, 0.0], MvNormal(2, 0.5)) 
spl2 = StaticMH([0.0, 0.0], MvNormal(2, 0.5)) 
```

## Multithreaded sampling

AdvancedMH.jl implements the interface of [AbstractMCMC](https://github.com/TuringLang/AbstractMCMC.jl/), which means you get multiple chain sampling
in parallel for free:

```julia
# Sample 4 chains from the posterior.
chain = psample(model, RWMH(init_params), 100000, 4; param_names=["μ","σ"])
```
