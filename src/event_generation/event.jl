using UUIDs
using PhysicsTools
using JSON
using StructTypes


struct Event
    data::Dict{Symbol, Any}
    id::UUID
end
Event() = Event(Dict{Symbol, Any}(), uuid4())

Base.getindex(e::Event, key::Symbol) = e.data[key]
Base.setindex!(e::Event, v, key::Symbol) = e.data[key] = v

StructTypes.StructType(::Type{<:Event}) = StructTypes.Struct()

# TODO: Add logic for dark / parent particles
function get_lightemitting_particles(e::Event)
    return e[:particles]
end
