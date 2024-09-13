module NeutrinoTelescopes

using Reexport

include("processing/Processing.jl")
include("event_generation/EventGeneration.jl")
include("surrogate_models/SurrogateModels.jl")
include("event_reconstruction/EventReconstruction.jl")
include("fisher_information/FisherInformation.jl")
include("weighting/Weighting.jl")

@reexport using .Processing
@reexport using .EventGeneration
@reexport using .SurrogateModels
@reexport using .EventReconstruction
@reexport using .FisherInformation
@reexport using .Weighting
end
