module NeutrinoTelescopes

using Reexport

include("processing/Processing.jl")
include("pmt_frontend/PMTFrontEnd.jl")
include("event_generation/EventGeneration.jl")
include("surrogate_models/SurrogateModels.jl")


@reexport using .Processing
@reexport using .PMTFrontEnd
@reexport using .SurrogateModels
@reexport using .EventGeneration
end
