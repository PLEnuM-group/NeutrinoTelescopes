module Utils
using StaticArrays
using FastGaussQuadrature
using LinearAlgebra
using DataStructures
using Distributions

export fast_linear_interp, transform_integral_range
export integrate_gauss_quad
export sph_to_cart
export CategoricalSetDistribution

const GL10 = gausslegendre(10)

function fast_linear_interp(x_eval::T, xs::AbstractVector{T}, ys::AbstractVector{T}) where {T}

    lower = first(xs)
    upper = last(xs)
    x_eval = clamp(x_eval, lower, upper)


    ix_upper = searchsortedfirst(xs, x_eval)
    ix_lower = ix_upper - 1

    @inbounds edge_l = xs[ix_lower]
    @inbounds edge_h = xs[ix_upper]

    step = edge_h - edge_l

    along_step = (x_eval - edge_l) / step

    @inbounds y_low = ys[ix_lower]
    @inbounds slope = (ys[ix_upper] - y_low)

    interpolated = y_low + slope * along_step

    return interpolated

end


function fast_linear_interp(x::T, knots::AbstractVector{T}, lower::T, upper::T) where {T}
    # assume equidistant

    x = clamp(x, lower, upper)
    range = upper - lower
    n_knots = size(knots, 1)
    step_size = range / (n_knots - 1)

    along_range = (x - lower) / step_size
    along_range_floor = floor(along_range)
    lower_knot = Int64(along_range_floor) + 1

    if lower_knot == n_knots
        return @inbounds knots[end]
    end

    along_step = along_range - along_range_floor
    @inbounds y_low = knots[lower_knot]
    @inbounds slope = (knots[lower_knot+1] - y_low)

    interpolated = y_low + slope * along_step

    return interpolated
end


function transform_integral_range(x::Real, f::T, xrange::Tuple{<:Real,<:Real}) where {T<:Function}
    ba_half = (xrange[2] - xrange[1]) / 2

    u_traf = ba_half * x + (xrange[1] + xrange[2]) / 2
    oftype(x, f(u_traf) * ba_half)

end

function integrate_gauss_quad(f::T, a::Real, b::Real) where {T<:Function}
    integrate_gauss_quad(f, a, b, GL10[1], GL10[2])
end

function integrate_gauss_quad(f::T, a::Real, b::Real, order::Integer) where {T<:Function}
    nodes, weights = gausslegendre(order)
    integrate_gauss_quad(f, a, b, nodes, weights)
end

function integrate_gauss_quad(f::T, a::Real, b::Real, nodes::AbstractVector{U}, weights::AbstractVector{U}) where {T<:Function,U<:Real}
    dot(weights, map(x -> transform_integral_range(x, f, (a, b)), nodes))
end

function sph_to_cart(theta::Real, phi::Real)
    sin_theta, cos_theta = sincos(theta)
    sin_phi, cos_phi = sincos(phi)

    T = promote_type(typeof(theta), typeof(phi))
    x::T = cos_phi * sin_theta
    y::T = sin_phi * sin_theta
    z::T = cos_theta

    return SA[x, y, z]
end

"""
CategoricalSetDistribution{T, U<:Real}

Represents a Categorical distribution on a set

### Examples

- `p = CategoricalSetDistribution(Set([:EMinus, :EPlus]), Categorical([0.1, 0.9]))
   rand(p)` -- returns `:EMinus` with 10% probability and `:Eplus` with 90% probability

- `p = CategoricalSetDistribution(Set([:EMinus, :EPlus]), [0.1, 0.9])` -- convenience constructor 
"""
struct CategoricalSetDistribution{T}
    set::OrderedSet{T}
    cat::Categorical

    function CategoricalSetDistribution(set::OrderedSet{T}, cat::Categorical) where {T}
        if length(set) != ncategories(cat)
            error("Set and categorical have to be of same length")
        end
        new{T}(set, cat)
    end

    function CategoricalSetDistribution(set::OrderedSet{T}, probs::Vector{<:Real}) where {T}
        new{T}(set, Categorical(probs))
    end
end

Base.rand(pdist::CategoricalSetDistribution) = pdist.set[rand(pdist.cat)]



end