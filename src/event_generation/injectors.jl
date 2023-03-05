module Injectors

using Random
using StaticArrays
using Distributions
using PhysicsTools
using UUIDs
using StructTypes
import Base.rand
import ..Event

export sample_volume, inject
export Cylinder, Cuboid, VolumeType
export VolumeInjector, Injector
export ParticleTypeDistribution
export AngularDistribution, UniformAngularDistribution

"""
    VolumeType

Abstract type for volumes
"""
abstract type VolumeType end

"""
    Cylinder{T} <: VolumeType

Type for cylindrical volumes.
"""
struct Cylinder{T} <: VolumeType
    center::SVector{3,T}
    height::T
    radius::T
end

StructTypes.StructType(::Type{<:Cylinder}) = StructTypes.Struct()

"""
    Cylinder{T} <: VolumeType

Type for cuboid volumes.
"""
struct Cuboid{T} <: VolumeType
    center::SVector{3,T}
    l_x::T
    l_y::T
    l_z::T
end

StructTypes.StructType(::Type{<:Cuboid}) = StructTypes.Struct()

struct FixedPosition{T} <: VolumeType
    position::SVector{3,T}
end

StructTypes.StructType(::Type{<:FixedPosition}) = StructTypes.Struct()
"""
    rand(::VolumeType)

Sample a random point in volume
"""
rand(::VolumeType) = error("Not implemented")
rand(vol::FixedPosition) = vol.position

function rand(vol::Cylinder{T}) where {T}
    uni = Uniform(-vol.height / 2, vol.height / 2)
    rng_z = rand(uni)

    rng_r = sqrt(rand(T) * vol.radius)
    rng_phi = rand(T) * 2 * π
    rng_x = rng_r * cos(rng_phi)
    rng_y = rng_r * sin(rng_phi)

    return SA{T}[rng_x, rng_y, rng_z] + vol.center

end

function rand(vol::Cuboid{T}) where {T}
    uni_x = Uniform(-vol.l_x / 2, vol.l_x / 2)
    uni_y = Uniform(-vol.l_y / 2, vol.l_y / 2)
    uni_z = Uniform(-vol.l_z / 2, vol.l_z / 2)
    return SA{T}[rand(uni_x), rand(uni_y), rand(uni_z)] + vol.center

end

abstract type AngularDistribution end
struct UniformAngularDistribution <: AngularDistribution end

StructTypes.StructType(::Type{UniformAngularDistribution}) = StructTypes.Struct()

function Base.rand(::UniformAngularDistribution)
    phi = rand() * 2 * π
    theta = acos(2 * rand() - 1)
    return sph_to_cart(theta, phi)
end


abstract type Injector end
struct VolumeInjector{
    V<:VolumeType,
    E<:UnivariateDistribution,
    A<:AngularDistribution,
    L<:UnivariateDistribution,
    T<:UnivariateDistribution} <: Injector
    volume::V
    e_dist::E
    type_dist::CategoricalSetDistribution
    angular_dist::A
    length_dist::L
    time_dist::T
end

StructTypes.StructType(::Type{VolumeInjector}) = StructTypes.Struct()


function Base.rand(inj::VolumeInjector)
    pos = rand(inj.volume)
    energy = rand(inj.e_dist)
    ptype = rand(inj.type_dist)
    dir = rand(inj.angular_dist)
    length = rand(inj.length_dist)
    time = rand(inj.time_dist)

    event = Event()
    event[:particles] = [Particle(pos, dir, time, energy, length, ptype)]

    return event

end

end
