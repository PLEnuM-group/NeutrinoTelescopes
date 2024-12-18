using UUIDs
using PhysicsTools
using StructTypes


struct Event
    data::Dict{Symbol, Any}
    id::UUID
end
Event() = Event(Dict{Symbol, Any}(), uuid4())

Base.getindex(e::Event, key::Symbol) = e.data[key]
Base.setindex!(e::Event, v, key::Symbol) = e.data[key] = v
Base.haskey(e::Event, key) = haskey(e.data, key)

StructTypes.StructType(::Type{<:Event}) = StructTypes.Struct()

# TODO: Add logic for dark / parent particles
function get_lightemitting_particles(e::Event)
    return e[:particles]
end
