module NeutrinoTelescopes

using Reexport

include("processing/Processing.jl")
include("pmt_frontend/PMTFrontEnd.jl")
include("event_generation/EventGeneration.jl")
include("surrogate_models/SurrogateModels.jl")
include("plotting/Plotting.jl")
include("event_reconstruction/EventReconstruction.jl")


@reexport using .Processing
@reexport using .PMTFrontEnd
@reexport using .SurrogateModels
@reexport using .EventGeneration
@reexport using .Plotting
@reexport using .EventReconstruction
end
