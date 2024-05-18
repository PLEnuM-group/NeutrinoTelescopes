
export FisherCalculator
export mask_events_in_calc
export calc_fisher
export evaluate_optimization_metric

using ...SurrogateModels

struct FisherCalculator{FM <: FisherSurrogateModel}
    events::Vector{Event}
    fisher_model::FM
end

mask_events_in_calc(fc::FisherCalculator, mask) = FisherCalculator(fc.events[mask], fc.fisher_model)

function calc_fisher(
    calc::FisherCalculator,    
    detector::Detector;
    abs_scale,
    sca_scale)

    events = calc.events
    fisher_model = calc.fisher_model

    targets = get_detector_modules(detector)
    event_mask = get_events_in_range(events, targets, fisher_model)
    valid_events = events[event_mask]
    lines = get_detector_lines(detector)

    fishers = predict_fisher(valid_events, lines, fisher_model, abs_scale=abs_scale, sca_scale=sca_scale)
    return fishers, event_mask
end

function calc_fisher(
    calc::FisherCalculator,    
    xy::NTuple{2, <:Real};
    abs_scale,
    sca_scale)

    events = calc.events
    fisher_model = calc.fisher_model

    x, y = xy
    targets = new_line(x, y)
    event_mask = get_events_in_range(events, targets, fisher_model)
    valid_events = events[event_mask]
    fishers = predict_fisher(valid_events, [targets], fisher_model, abs_scale=abs_scale, sca_scale=sca_scale)
    return fishers, event_mask
end




abstract type OptimizationMetric end

abstract type FisherOptimizationMetric <: OptimizationMetric end


struct SingleEventTypeResolution <: FisherOptimizationMetric end
struct SingleEventTypeTotalResolution <: FisherOptimizationMetric end
struct SingleEventTypeTotalRelativeResolution <: FisherOptimizationMetric end
struct SingleEventTypePerEventRelativeResolution <: FisherOptimizationMetric end
struct SingleEventTypeMeanDeviation <: FisherOptimizationMetric end
struct SingleEventTypePerEventMeanDeviation <: FisherOptimizationMetric end
struct SingleEventTypePerEventMeanAngularError <: FisherOptimizationMetric end
struct SingleEventTypeTotalResolutionNoVertex <:  FisherOptimizationMetric end
struct SingleEventTypeDOptimal <: FisherOptimizationMetric end
struct SingleEventTypeDNoVertex <: FisherOptimizationMetric end

struct MultiEventTypeTotalResolution{OM <: OptimizationMetric} <: OptimizationMetric
    metrics::Vector{OM}
end

function evaluate_optimization_metric(::SingleEventTypeTotalResolution, fishers, events)
    covs = invert_fishers(fishers)
    valid = isposdef.(covs)
    mean_cov = @views mean(covs[valid])
    sigmasq = diag(mean_cov)
    return sqrt(sum(sigmasq))
end

function get_event_props(event::Event)
    p = first(event[:particles])
    dir_sph = cart_to_sph(p.direction)
    shift_to_closest_approach(p, [0., 0., 0.])
    return [log10(p.energy), dir_sph..., p.position...]
end


function get_positional_uncertainty(pos, cov)
    d = MvNormal(pos, cov)
    smpl = rand(d, 100)
    dpos = norm.(Ref(pos) .- eachcol(smpl))
    return mean(dpos)
end

function get_directional_uncertainty(dir_sph, cov)
    dir_cart = sph_to_cart(dir_sph)

    dist = MvNormal(dir_sph, cov)
    rdirs = rand(dist, 100)

    dangles = rad2deg.(acos.(dot.(sph_to_cart.(rdirs[1, :], rdirs[2, :]), Ref(dir_cart))))
    return mean(dangles)
end

function evaluate_optimization_metric(::SingleEventTypePerEventMeanDeviation, fishers, events)
    covs = invert_fishers(fishers)
    valid = isposdef.(covs)
    results = zeros(length(events))

    for (i, (c, e)) in enumerate(zip(covs, events))
        if !valid[i]
            results[i] = NaN
            continue
        end
        event_props = get_event_props(e)
        mean_pos_dev = get_positional_uncertainty(event_props[4:6], c[4:6, 4:6])
        mean_ang_dev = get_directional_uncertainty(event_props[2:3], c[2:3, 2:3])
        mean_energy_dev = sqrt(c[1, 1])
        results[i] = mean_pos_dev + mean_ang_dev + mean_energy_dev
    end

    return results
end

function evaluate_optimization_metric(::SingleEventTypePerEventMeanAngularError, fishers, events)
    covs = invert_fishers(fishers)
    valid = isposdef.(covs)
    results = zeros(length(events))

    for (i, (c, e)) in enumerate(zip(covs, events))
        if !valid[i]
            results[i] = NaN
            continue
        end
        event_props = get_event_props(e)
        mean_ang_dev = get_directional_uncertainty(event_props[2:3], c[2:3, 2:3])
        results[i] = mean_ang_dev
    end

    return results
end


function evaluate_optimization_metric(::SingleEventTypeMeanDeviation, fishers, events)
    m = SingleEventTypePerEventMeanDeviation()

    results = evaluate_optimization_metric(m, fishers, events)
    masked_results = filter!(!isnan, results) 

    return mean(masked_results)
end



function evaluate_optimization_metric(::SingleEventTypeTotalRelativeResolution, fishers, events)
    covs = invert_fishers(fishers)
    valid = isposdef.(covs)

    results = zeros(sum(valid))
    for (i, (c, e)) in enumerate(zip(covs[valid], events[valid]))
        event_props = get_event_props(e)
        rel_sigma = diag(c) ./ event_props .^2
        results[i] = sqrt(sum(rel_sigma))
    end

    return mean(results)
end

function evaluate_optimization_metric(::SingleEventTypePerEventRelativeResolution, fishers, events)
    covs = invert_fishers(fishers)
    valid = isposdef.(covs)
    results = zeros(sum(valid))
    for (i, (c, e)) in enumerate(zip(covs[valid], events[valid]))
        event_props = get_event_props(e)
        rel_sigma = diag(c) ./ event_props .^2
        results[i] = sqrt(sum(rel_sigma))
    end

    return results
end


function evaluate_optimization_metric(::SingleEventTypeResolution, fishers, events)
    covs = invert_fishers(fishers)
    valid = isposdef.(covs)
    mean_cov = mean(covs[valid])
    sigmasq = diag(mean_cov)
    return sqrt.(sigmasq)
end



function evaluate_optimization_metric(::SingleEventTypeTotalResolutionNoVertex, fishers, events)
    fisher_no_pos = fishers[:, 1:3, 1:3]
    cov = mean(invert_fishers(fisher_no_pos))
    met = sqrt(sum(diag(cov)))
    return met
end

function evaluate_optimization_metric(::SingleEventTypeDOptimal, fishers, events)
    cov = mean(invert_fishers(fishers))
    cov = 0.5 * (cov + cov')
    met = det(cov)
    return met
end

function evaluate_optimization_metric(::SingleEventTypeDNoVertex, fishers, events)
    fisher_no_pos = fishers[:, 1:3, 1:3]
    cov = mean(invert_fishers(fisher_no_pos))
    cov = 0.5 * (cov + cov')
    met = det(cov)
    return met
end


function evaluate_optimization_metric(xy, detector, m::OptimizationMetric, events; abs_scale, sca_scale)

    targets = get_detector_modules(detector)
    event_mask = get_events_in_range(m.events, targets, m.fisher_model)

    masked_m = mask_events_in_metric(m, event_mask)

    x, y = xy
    new_targets = new_line(x, y)
    new_det = add_line(detector, new_targets)
    met = evaluate_optimization_metric(new_det, masked_m, abs_scale=abs_scale, sca_scale=sca_scale)
    return met
end

function evaluate_optimization_metric(xy, detector, m::MultiEventTypeTotalResolution, events; abs_scale, sca_scale)
    return sum(evaluate_optimization_metric(xy, detector, m, events, abs_scale=abs_scale, sca_scale=sca_scale) for am in m.metrics)
end