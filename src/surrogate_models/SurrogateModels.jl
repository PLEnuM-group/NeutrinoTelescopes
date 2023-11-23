module SurrogateModels

using Reexport

include("rq_spline_flow.jl")
include("neural_flow_surrogate.jl")
include("surrogate_hits.jl")
include("fisher_surrogate.jl")

@reexport using .RQSplineFlow
@reexport using .NeuralFlowSurrogate
@reexport using .SurrogateModelHits
@reexport using .FisherSurrogate
end
