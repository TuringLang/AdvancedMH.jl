using .ForwardDiff: gradient!
using .DiffResults: GradientResult, value, gradient
using .AdvancedMH: AdvancedMH

function AdvancedMH.logdensity_and_gradient(model::AdvancedMH.DensityModel, params)
    res = GradientResult(params)
    gradient!(res, model.logdensity, params)
    return value(res), gradient(res)
end
