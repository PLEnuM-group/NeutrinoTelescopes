using StructTypes
using .Injectors

export EventCollection, GenerationInfo

mutable struct GenerationInfo{I <: Union{Nothing, <:Injector}}
    injector::I
    n_events::Int64
end

function Base.:+(g::GenerationInfo, g2::GenerationInfo)
    if g.injector != g2.injector
        error("Injectors do not match")
    end

    return GenerationInfo(g.injector, g.n_events + g2.n_events)
end


mutable struct EventCollection{E <: Event, G <: Union{Nothing, <:GenerationInfo}}
    events::Vector{E}
    gen_info::G
end

EventCollection() = EventCollection(Vector{Event}(), nothing)

Base.getindex(e::EventCollection, i) = e.events[i]
Base.setindex!(e::EventCollection, v, i) = e.events[i] = v
Base.firstindex(e::EventCollection) = firstindex(e.events)
Base.lastindex(e::EventCollection) = lastindex(e.events)
Base.push!(e::EventCollection, v) = push!(e.events, v)
Base.size(e::EventCollection) = size(e.events)
Base.iterate(e::EventCollection) = iterate(e.events)
Base.iterate(e::EventCollection, state) = iterate(e.events, state)
Base.length(e::EventCollection) = length(e.events)


StructTypes.StructType(::Type{<:EventCollection}) = StructTypes.Struct()


function write_plain_hdf!(ec::EventCollection)
end

