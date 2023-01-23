module PulseTemplates

using Plots
using Polynomials
using Distributions
using DSP
using Interpolations
using Base.Iterators
using DataFrames
using PhysicsTools
import Base:@kwdef

using ..SPETemplates


export PulseTemplate, PDFPulseTemplate, GumbelPulse, InterpolatedPulse
export make_pulse_dist, evaluate_pulse_template, make_filtered_pulse
export PulseSeries, evaluate_pulse_series
export gumbel_width_from_fwhm
export get_template_mode


"""
    fit_gumbel_fwhm_width()

Fit a polynomial to the relationship between Gumbel width and FWHM
"""
function fit_gumbel_fwhm_width()
    # find relationship between Gumbel width and FWHM


    widths = 0.5:0.01:5

    # Fit the function width = a * fwhm + b
    poly = Polynomials.fit(map(w -> fwhm(Gumbel(0, w), w), widths), widths, 1)
    poly
end

gumbel_width_from_fwhm = fit_gumbel_fwhm_width()


"""
Abstract type for pulse templates
"""
abstract type PulseTemplate end



"""
Abstract type for pulse templates that use a PDF to define the pulse shape
"""
@kwdef struct PDFPulseTemplate{U<:UnivariateDistribution} <: PulseTemplate
    dist::U
    amplitude::Float64
end

get_template_mode(::PulseTemplate) = error("Not implemented")
get_template_mode(p::PDFPulseTemplate) = mode(p.dist)
get_template_mode(p::PDFPulseTemplate{<: Distributions.Truncated}) = mode(p.dist.untruncated)

"""
Pulse template using an interpolation to define its shape
"""
@kwdef struct InterpolatedPulse <: PulseTemplate
    interp
    amplitude::Float64
end


"""
    evaluate_pulse_template(pulse_shape::PulseTemplate, pulse_time::T, timestamp::T)

Evaluate a pulse template `pulse_shape` placed at time `pulse_time` at time `timestamp`
"""
evaluate_pulse_template(
    ::PulseTemplate,
    ::Real,
    ::Real) = error("not implemented")

function evaluate_pulse_template(
    pulse_shape::PDFPulseTemplate,
    pulse_time::Real,
    timestamp::Real)

    shifted_time = timestamp - pulse_time

    return pdf(pulse_shape.dist, shifted_time) * pulse_shape.amplitude
end


function evaluate_pulse_template(
    pulse_shape::InterpolatedPulse,
    pulse_time::Real,
    timestamp::Real)

    shifted_time = timestamp - pulse_time
    return pulse_shape.interp(shifted_time) * pulse_shape.amplitude
end


function evaluate_pulse_template(
    pulse_shape::PulseTemplate,
    pulse_time::Real,
    timestamps::AbstractVector{<:Real})
    return evaluate_pulse_template.(Ref(pulse_shape), Ref(pulse_time), timestamps)
end



"""
    make_filteres_pulse(orig_pulse, sampling_frequency, eval_range, filter)

    Create filtered response of `orig_pulse` using `filter` and return
    `InterpolatedPulse`.
"""
function make_filtered_pulse(
    orig_pulse::PulseTemplate,
    sampling_freq::Real,
    eval_range::Tuple{<:Real,<:Real},
    filter)

    timesteps = range(eval_range[1], eval_range[2], step=1 / sampling_freq)
    orig_eval = evaluate_pulse_template(orig_pulse, 0.0, timesteps)
    filtered = filt(filter, orig_eval)
    interp_linear = linear_interpolation(timesteps, filtered,extrapolation_bc=0.)

    return InterpolatedPulse(interp_linear, 1.0)
end

struct PulseSeries{T<:AbstractVector{<:Real}, U<:PulseTemplate}
    times::T
    charges::T
    pulse_shape::U

    function PulseSeries(
        times::AbstractVector,
        charges::AbstractVector,
        shape::PulseTemplate)

        ptype = promote_type(eltype(times), eltype(charges))
        ix = sortperm(times)
        t = convert.(ptype, times[ix])
        c = convert.(ptype, charges[ix])
        return new{typeof(t), typeof(shape)}(t, c, shape)
    end
end

function PulseSeries(times::AbstractVector{<:Real}, spe_template::SPEDistribution, pulse_shape::PulseTemplate)
    spe_d = make_spe_dist(spe_template)
    charges = rand(spe_d, length(times))
    PulseSeries(times, charges, pulse_shape)
end

function PulseSeries(df::AbstractDataFrame, spe_template::SPEDistribution, pulse_shape::PulseTemplate)
    PulseSeries(df[:, :time], spe_template, pulse_shape)
end




Base.length(ps::PulseSeries) = length(ps.times)

function Base.:+(a::PulseSeries, b::PulseSeries)
    # Could instead parametrize PulseSeries by PulseShape
    if a.pulse_shape != b.pulse_shape
        throw(ArgumentError("Pulse shapes are not compatible"))
    end
    PulseSeries([a.times; b.times], [a.charges; b.charges], a.pulse_shape)
end


function evaluate_pulse_series(time::Real, ps::PulseSeries)
    sum(evaluate_pulse_template.(Ref(ps.pulse_shape), ps.times, Ref(time)) .* ps.charges)
end

function evaluate_pulse_series(times::AbstractVector{<:Real}, ps::PulseSeries)

    u = length(times)
    v = length(ps)

    output = Matrix{eltype(times)}(undef, u, v)

    @inbounds for (i, j) in product(eachindex(times), eachindex(ps.times))
        output[i, j] = evaluate_pulse_template(ps.pulse_shape, ps.times[j], times[i]) * ps.charges[j]
    end

    return vec(sum(output, dims=2))

end

(p::PulseSeries)(times) = evaluate_pulse_series(times, p)

@recipe function f(::Type{T}, ps::T) where {T<:PulseSeries}
    xlim := (ps.times[1]-10, ps.times[end]+10)
    xi -> evaluate_pulse_series(xi, ps)
end

end
