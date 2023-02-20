module EventGeneration
using Reexport

include("injectors.jl")
include("detectors.jl")

@reexport using .Injectors
@reexport using .Detectors

try
    include("proposal_interface.jl")
    @reexport using .ProposalInterface
catch y
    @warn "Could not load proposal interface."
end
end
