
using PhotonSurrogateModel
using StatsBase
using PhysicsTools
using LinearAlgebra
using Distributions


using ..EventGeneration

export OptimizationMetric, AngularResolutionMetric, AOptimalityMetric, AOptimalityMetricDetEff, AngResDetEff, AngResDetEffUnweighted




function get_directional_uncertainty(dir_sph, cov)
    dir_cart = sph_to_cart(dir_sph)

    dist = MvNormal(dir_sph, cov)
    rdirs = rand(dist, 200)

    dangles = rad2deg.(acos.(dot.(sph_to_cart.(rdirs[1, :], rdirs[2, :]), Ref(dir_cart))))
    return mean(dangles)
end


function calc_total_fisher_matrix(layout::OptimizationLayout{T}, surrogate::AnalyticSurrogateModel, event::Event, diff_res; per_module_threshold=3) where {T}

    fisher_matrix = zeros(T, 6, 6)

    p = first(event[:particles])
    n_above = zero(T)
    total_pred = zero(T)
    for mod_pos in layout
        _, pred_permod = calculate_fisher_matrix!(surrogate, fisher_matrix, mod_pos, layout.pmt_positions, p.position, p.direction, log10(p.energy), diff_res)
        total_pred += pred_permod
        #n_above += soft_cut(pred_permod, 10, 3)
        n_above += poisson_atleast_k(pred_permod, per_module_threshold)
    end

    return fisher_matrix, total_pred, n_above
end


abstract type OptimizationMetric end

struct AngularResolutionMetric <: OptimizationMetric end

struct AOptimalityMetric <: OptimizationMetric end

struct AOptimalityMetricDetEff <: OptimizationMetric 
    n_events::Int64
    gen_volume::Float64
    per_module_threshold::Int64
    n_module_threshold::Int64    
end

struct AngResDetEff <: OptimizationMetric 
    n_events::Int64
    gen_volume::Float64
    per_module_threshold::Int64
    n_module_threshold::Int64
    spectral_index::Float64
    flux_norm::Float64
    e_range::Tuple{Float64, Float64}
end

struct AngResDetEffUnweighted <: OptimizationMetric 
    n_events::Int64
    gen_volume::Float64
    per_module_threshold::Int64
    n_module_threshold::Int64
    flux_norm::Float64
    e_range::Tuple{Float64, Float64}
end


function (metric::OptimizationMetric)(sr::SRSurrogateModel, layout::OptimizationLayout{T}, events) where {T}
    metric_val = zero(T)
    valid = 0

    diff_res = create_diff_result(sr, T)
    for ev in events
        m = eval_metric(metric, sr, layout, ev, diff_res)
        if isfinite(m)
            metric += m
            valid += 1
        end
    end
    return metric / valid
end


function eval_metric(::AngularResolutionMetric, sr::SRSurrogateModel, layout::OptimizationLayout{T}, event, diff_res) where {T}
    fisher_matrix, total_pred, n_above = calc_total_fisher_matrix(layout, sr, event, diff_res)

    sym = 0.5 * (fisher_matrix + transpose(fisher_matrix))
    if !isapprox(det(sym), 0)

        cov = inv(sym)
        cov_sym = 0.5 * (cov + LinearAlgebra.transpose(cov))
        dir_cov = cov_sym[4:5, 4:5]


        dir_sph = cart_to_sph(first(event[:particles]).direction)
        
        if isposdef(dir_cov)
            metric = get_directional_uncertainty(dir_sph, dir_cov)
        else
            metric = NaN
        end
    else
        metric = NaN
    end

    return metric
end


function eval_metric(::AOptimalityMetric, sr::SRSurrogateModel, layout::OptimizationLayout{T}, event, diff_res) where {T}

    fisher_matrix, total_pred, n_above = calc_total_fisher_matrix(layout, sr, event, diff_res)

    sym = 0.5 * (fisher_matrix + transpose(fisher_matrix))
    if !isapprox(det(sym), 0)
        metric_sq = tr(inv(sym))
        metric = sqrt(metric_sq)
    else
        metric = NaN
    end
    return metric
end


function (metric::AOptimalityMetricDetEff)(sr::SRSurrogateModel, layout::OptimizationLayout{T}, events; detailed=false) where {T}
    metric_val = zero(T)
    n_detected = zero(T)

    n_valid = 0
    diff_res = create_diff_result(sr, T)
    for event in events
        fisher_matrix, total_pred, n_above = calc_total_fisher_matrix(layout, sr, event, diff_res, per_module_threshold=metric.per_module_threshold)

        sym = 0.5 * (fisher_matrix + transpose(fisher_matrix))
        if !isapprox(det(sym), 0)
            metric_sq = tr(inv(sym))
            metric_val += sqrt(metric_sq)
            n_valid += 1
        end

        n_detected  += poisson_atleast_k(n_above, metric.n_module_threshold)

    end

    eff_volume = n_detected / metric.n_events * metric.gen_volume

    if detailed
        return metric_val / n_valid, eff_volume
    end
    return metric_val / n_valid - eff_volume
end

function (metric::AngResDetEff)(sr::SRSurrogateModel, layout::OptimizationLayout{T}, events) where {T}
    
    ang_res_val = zero(T)
    n_detected = zero(T)
    e_min, e_max = metric.e_range

    event_rate_contrib_sig = zero(T)
    event_rate_contrib_bg = zero(T)

    sum_w = zero(T)

    met_ratio = zero(T)
    

    diff_res = create_diff_result(sr, T)
    
    if metric.spectral_index == -1
        spectral_norm = (log(e_max) - log(e_min)) 
    else
        spectral_norm = (e_max^(metric.spectral_index+1) / (metric.spectral_index+1) - e_min^(metric.spectral_index+1) / (metric.spectral_index+1)) 
    end

    for event in events
        fisher_matrix, total_pred, n_above = calc_total_fisher_matrix(layout, sr, event, diff_res, per_module_threshold=metric.per_module_threshold)

        detection_prob = poisson_atleast_k(n_above, metric.n_module_threshold)

        dir_uncert = 90

        sym = 0.5 * (fisher_matrix + LinearAlgebra.transpose(fisher_matrix))
        if !isapprox(det(sym), 0) .&& all(isfinite, sym)
            cov = inv(sym)
            cov_sym = 0.5 * (cov + LinearAlgebra.transpose(cov))
            dir_cov = cov_sym[4:5, 4:5]

            dir_sph = cart_to_sph(first(event[:particles]).direction)
            
            if isposdef(dir_cov) 
                uncert = get_directional_uncertainty(dir_sph, dir_cov) 
                
                if isfinite(uncert)
                    dir_uncert = uncert
                end
            end
        end

        detection_prob = poisson_atleast_k(n_above, metric.n_module_threshold)
        n_detected  += detection_prob 

        
        event_energy = first(event[:particles]).energy
        generation_weight = 1 / (event_energy ^ -1 / (log(e_max)- log(e_min)))
        
        
        flux_weight_sig =event_energy ^ metric.spectral_index 
        flux_weight_bg = event_energy ^ (-3.7)

        #sum_w += generation_weight * detection_prob * flux_weight_sig
        
        #xsec_CC = 5.53E-36 * event_energy ^ 0.363 # cm^2
        #number_density_water = 33.3679E21 # cm^-3	

        xsec_weight = event_energy ^ 0.363
        eff_volume_contrib = detection_prob / metric.n_events * metric.gen_volume 
       
        ang_res_val += dir_uncert * detection_prob * generation_weight * flux_weight_sig

        event_rate_contrib_sig +=  xsec_weight * flux_weight_sig * generation_weight * eff_volume_contrib
        event_rate_contrib_bg +=   xsec_weight * flux_weight_bg * generation_weight * eff_volume_contrib

        sum_w +=  detection_prob * generation_weight * flux_weight_sig
       #met_ratio +=  dir_uncert * sqrt(flux_weight_bg / (eff_volume_contrib  * xsec_weight)) / flux_weight_sig

    end

    # Flux averaged angular resolution
    ang_res_val /= sum_w

    #return ang_res_val  / sqrt(event_rate_contrib)
    return metric.flux_norm * ang_res_val * sqrt(event_rate_contrib_bg) / event_rate_contrib_sig
end

function (metric::AngResDetEffUnweighted)(sr::SRSurrogateModel, layout::OptimizationLayout{T}, events) where {T}
    
    ang_res_val = zero(T)
    n_detected = zero(T)
    e_min, e_max = metric.e_range

    event_rate_contrib = zero(T)

    sum_w = zero(T)

    diff_res = create_diff_result(sr, T)


   
    for event in events
        fisher_matrix, total_pred, n_above = calc_total_fisher_matrix(layout, sr, event, diff_res, per_module_threshold=metric.per_module_threshold)

        detection_prob = poisson_atleast_k(n_above, metric.n_module_threshold)

        dir_uncert = 90

        sym = 0.5 * (fisher_matrix + LinearAlgebra.transpose(fisher_matrix))
        if !isapprox(det(sym), 0) .&& all(isfinite, sym)
            cov = inv(sym)
            cov_sym = 0.5 * (cov + LinearAlgebra.transpose(cov))
            dir_cov = cov_sym[4:5, 4:5]

            dir_sph = cart_to_sph(first(event[:particles]).direction)
            
            if isposdef(dir_cov) 
                uncert = get_directional_uncertainty(dir_sph, dir_cov) 
                
                if isfinite(uncert)
                    dir_uncert = uncert
                end
            end
        end

        detection_prob = poisson_atleast_k(n_above, metric.n_module_threshold)
        n_detected  += detection_prob 

        
        event_energy = first(event[:particles]).energy
        generation_weight = 1 / (event_energy ^ -1 / (log(e_max)- log(e_min)))
        
        #sum_w += generation_weight * detection_prob * flux_weight_sig
        
        #xsec_CC = 5.53E-36 * event_energy ^ 0.363 # cm^2
        #number_density_water = 33.3679E21 # cm^-3	

        xsec_weight = event_energy ^ 0.363
        eff_volume_contrib = detection_prob / metric.n_events * metric.gen_volume 
       
        ang_res_val += dir_uncert * detection_prob * generation_weight
        event_rate_contrib += xsec_weight * generation_weight * eff_volume_contrib

        sum_w +=  detection_prob * generation_weight 

       


    end

    ang_res_val /= sum_w




    return metric.flux_norm * ang_res_val  / sqrt(event_rate_contrib)

end