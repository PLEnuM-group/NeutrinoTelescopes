module SurrogateModels

using Reexport

include("rq_spline_flow.jl")
include("photon_surrogate/PhotonSurrogates.jl")
include("surrogate_hits.jl")
include("fisher_surrogate.jl")

@reexport using .RQSplineFlow
@reexport using .PhotonSurrogates
@reexport using .FisherSurrogate
@reexport using .SurrogateModelHits
end
