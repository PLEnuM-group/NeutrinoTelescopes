

export ModelParam, Model
export make_event_fit_model
export make_obj_func_cascade, make_obj_func_track

mutable struct ModelParam{T<:Number,LB<:Number,UB<:Number}
    name::String
    value::T
    active::Bool
    bounds::Tuple{LB,UB}
end

mutable struct Model
    params::Vector{ModelParam}
end

params(m::Model) = m.params
free_params(m::Model) = [p for p in params(m) if p.active]
get_lower_bounds(m::Model) = [p.bounds[1] for p in free_params(m)]
get_upper_bounds(m::Model) = [p.bounds[2] for p in free_params(m)]
get_seeds(m::Model) = [p.value for p in free_params(m)]

function get_param(m::Model, name::String)
    fp = nothing
    for p in params(m)
        if p.name == name
            fp = p
        end
    end

    if isnothing(fp)
        error("Parameter $name not found in model")
    end

    return fp
end

function set_inactive!(m::Model, name::String)
    p = get_param(m, name)
    p.active = false
end

function set_active!(m::Model, name::String)
    p = get_param(m, name)
    p.active = true
end

function parse_model_params(x, model)

    @assert length(x) == length(free_params(model))

    fix = 1
    mparams = params(model)

    parsed = []
    for par in mparams
        if mparams[fix].active
            val = x[fix]
            fix += 1
        else
            val = par.value
        end

        push!(parsed, val)
    end
    return parsed
end

function make_event_fit_model(; seed_log_energy=3.0, seed_zenith=0.5, seed_azimuth=0.5, seed_x=0.0, seed_y=0.0, seed_z=0.0, seed_time=0.0)
    logenergy = ModelParam("logenergy", seed_log_energy, true, (2.0, 5.0))
    zenith = ModelParam("zenith", seed_zenith, true, (0.0, π))
    azimuth = ModelParam("azimuth", seed_azimuth, true, (seed_azimuth - π, seed_azimuth + π))
    pos_x = ModelParam("pos_x", seed_x, true, (-500.0, 500.0))
    pos_y = ModelParam("pos_y", seed_y, true, (-500.0, 500.0))
    pos_z = ModelParam("pos_z", seed_z, true, (-1000.0, 100.0))
    time = ModelParam("time", seed_time, true, (-50.0, 100.0))
    return Model([logenergy, zenith, azimuth, pos_x, pos_y, pos_z, time])
end


function minimize_model(fit_model, obj_func; strategy=:cg)
    lower = get_lower_bounds(fit_model)
    upper = get_upper_bounds(fit_model)
    seeds = get_seeds(fit_model)

    if strategy == :cg
        inner_optimizer = ConjugateGradient()
        algo = Fminbox(inner_optimizer)
        results = optimize(obj_func, lower, upper, seeds, algo; autodiff=:forward)
    elseif strategy == :annealing
        algo = SAMIN(rt=0.5)
        opt = Optim.Options(iterations=10^5)
        results = optimize(obj_func, lower, upper, seeds, algo, opt)
    else
        error("Strategy $strategy unknown.")
    end

    return results
end

function make_obj_func_cascade(fit_model; data, targets, model, tf_vec, c_n, use_feat_buffer=true)

    feat_buffer = use_feat_buffer ? zeros(9, get_pmt_count(eltype(targets)) * length(targets)) : nothing

    function obj_func(x)
        logenergy, theta, phi, pos_x, pos_y, pos_z, time = parse_model_params(x, fit_model)
        fval = -single_cascade_likelihood(logenergy, theta, phi, SA[pos_x, pos_y, pos_z], time; data=data, targets=targets, model=model, tf_vec=tf_vec, c_n=c_n, feat_buffer=feat_buffer)
        return fval
    end
    return obj_func
end

function make_obj_func_track(fit_model; data, losses, muon_energy, targets, model, tf_vec, c_n, use_feat_buffer=true)

    feat_buffer = use_feat_buffer ? zeros(9, get_pmt_count(eltype(targets)) * length(targets) * length(losses)) : nothing

    function obj_func(x)
        logenergy, theta, phi, pos_x, pos_y, pos_z, time = parse_model_params(x, fit_model)
        fval = -track_likelihood_fixed_losses(logenergy, theta, phi, SA[pos_x, pos_y, pos_z], time; losses=losses_filt, muon_energy=muon_energy, data=data, targets=targets_range, model=model, tf_vec=tf_vec, c_n=c_n, feat_buffer=feat_buffer)
        return fval
    end
    return obj_func
end
