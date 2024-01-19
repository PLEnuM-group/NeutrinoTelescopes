module SurrogateModels

using Reexport

include("rq_spline_flow.jl")
include("surrogate_hits.jl")
include("fisher_surrogate.jl")
include("photon_surrogate/PhotonSurrogates.jl")

@reexport using .RQSplineFlow
@reexport using .SurrogateModelHits
@reexport using .FisherSurrogate
@reexport using .PhotonSurrogates
end
