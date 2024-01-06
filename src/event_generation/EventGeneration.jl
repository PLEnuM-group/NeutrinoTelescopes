module EventGeneration
using Reexport

include("event.jl")
include("injectors.jl")
include("event_collection.jl")
include("detectors.jl")
include("pmt_hits.jl")

export Event, get_lightemitting_particles
export EventCollection
@reexport using .Injectors
@reexport using .Detectors
@reexport using .PMTHits


end
