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
Base.length(e::EventCollection) = length(e.events)


function Base.vcat(ecs::Vararg{<:EventCollection})

    ij0 = first(ecs).injector
    for ec in ecs
        if ec.injector != ij0
            error("All injectors have to be identical for combination")
        end
    end
    combined_events = mapreduce(ec -> getproperty(ec, :events), vcat, ecs)

    return EventCollection(combined_events, ij0)
end


StructTypes.StructType(::Type{<:EventCollection}) = StructTypes.Struct()
