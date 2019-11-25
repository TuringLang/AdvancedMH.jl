# AdvancedMH.jl

AdvancedMH.jl currently provides a robust implementation of random walk Metropolis-Hastings samplers.

Further development aims to provide a suite of adaptive Metropolis-Hastings implementations.

## Usage

AdvancedMH works by accepting some log density function which is used to construct a `DensityModel`. The `DensityModel` is then used in a `sample` call.

```julia
# Import the package.
using AdvancedMH

# Define the components of a basic model.
insupport(θ) = θ[2] >= 0
dist(θ) = Normal(θ[1], θ[2])
density(data, θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf

# Generate a set of data from the posterior we want to estimate.
data = rand(Normal(0, 1), 30)

# Construct a DensityModel.
model = DensityModel(density, data)

# Set up our sampler with initial parameters.
spl = MetropolisHastings([0.0, 0.0])

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

│ Row │ parameters │ mean     │ std      │ naive_se    │ mcse       │ ess     │ r_hat   │
│     │ Symbol     │ Float64  │ Float64  │ Float64     │ Float64    │ Any     │ Any     │
├─────┼────────────┼──────────┼──────────┼─────────────┼────────────┼─────────┼─────────┤
│ 1   │ μ          │ -0.23908 │ 0.187608 │ 0.00059327  │ 0.00310081 │ 3225.02 │ 1.00003 │
│ 2   │ σ          │ 0.98903  │ 0.138437 │ 0.000437777 │ 0.00242748 │ 2830.8  │ 1.0003  │

Quantiles

│ Row │ parameters │ 2.5%     │ 25.0%     │ 50.0%     │ 75.0%     │ 97.5%    │
│     │ Symbol     │ Float64  │ Float64   │ Float64   │ Float64   │ Float64  │
├─────┼────────────┼──────────┼───────────┼───────────┼───────────┼──────────┤
│ 1   │ μ          │ -0.61185 │ -0.360519 │ -0.236016 │ -0.119666 │ 0.134902 │
│ 2   │ σ          │ 0.758477 │ 0.887894  │ 0.973777  │ 1.07983   │ 1.29455  │
```

## Custom proposals

Custom proposal distributions can be specified by passing a function to `MetropolisHastings`:

```julia
# Set up our sampler with initial parameters.
spl = MetropolisHastings([0.0, 0.0], x -> MvNormal(x, 0.5))
```