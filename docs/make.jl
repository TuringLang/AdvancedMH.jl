using Pkg

using Documenter
using AdvancedMH

# cp(joinpath(@__DIR__, "../README.md"), joinpath(@__DIR__, "src/index.md"))

makedocs(sitename = "AdvancedMH", format = Documenter.HTML(), modules = [AdvancedMH], checkdocs = :exports)
