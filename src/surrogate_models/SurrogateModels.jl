module SurrogateModels

using Reexport

include("surrogate_hits.jl")
include("fisher_surrogate.jl")

@reexport using .FisherSurrogate
@reexport using .SurrogateModelHits
end
