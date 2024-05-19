using StructTypes

struct EventCollection{E <: Event}
    events::Vector{E}
end

EventCollection() = EventCollection(Vector{Event}())

Base.getindex(e::EventCollection, i) = e.events[i]
Base.setindex!(e::EventCollection, v, i) = e.events[i] = v
Base.firstindex(e::EventCollection) = firstindex(e.events)
Base.lastindex(e::EventCollection) = lastindex(e.events)
Base.push!(e::EventCollection, v) = push!(e.events, v)
Base.size(e::EventCollection) = size(e.events)
Base.iterate(e::EventCollection) = iterate(e.events)
Base.iterate(e::EventCollection, state) = iterate(e.events, state)
Base.length(e::EventCollection) = length(e.events)


function Base.vcat(ecs::Vararg{<:EventCollection})

    combined_events = mapreduce(ec -> getproperty(ec, :events), vcat, ecs)
    return EventCollection(combined_events)
end

StructTypes.StructType(::Type{<:EventCollection}) = StructTypes.Struct()


function write_plain_hdf!(ec::EventCollection)
end

