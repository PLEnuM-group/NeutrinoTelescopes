module EventGeneration
include("injectors.jl")
include("detectors.jl")
include("proposal_interface.jl")
using Reexport
@reexport using .Injectors
@reexport using .Detectors
@reexport using .ProposalInterface
end