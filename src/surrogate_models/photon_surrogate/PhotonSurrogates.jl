module PhotonSurrogates

using Flux
using BSON: @save, load
using StructTypes

include("utils.jl")
include("dataio.jl")
include("time_models.jl")
include("amplitude_models.jl")

end