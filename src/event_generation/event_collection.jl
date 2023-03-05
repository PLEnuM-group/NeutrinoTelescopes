using .Injectors
using StructTypes

struct EventCollection{E <: Event, I<:Injector}
    events::Vector{E}
    injector::I
end

EventCollection(i::Injector) = EventCollection(Vector{Event}(), i)

Base.getindex(e::EventCollection, i) = e.events[i]
Base.setindex!(e::EventCollection, v, i) = e.events[i] = v
Base.firstindex(e::EventCollection) = firstindex(e.events)
Base.lastindex(e::EventCollection) = lastindex(e.events)
Base.push!(e::EventCollection, v) = push!(e.events, v)
Base.size(e::EventCollection) = size(e.events)
Base.iterate(e::EventCollection) = iterate(e.events)
Base.iterate(e::EventCollection, state) = iterate(e.events, state)

StructTypes.StructType(::Type{<:EventCollection}) = StructTypes.Struct()
