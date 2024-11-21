using SpecialFunctions
using StatsBase
using LossFunctions
import LossFunctions: SupervisedLoss
using StaticArrays
using ProgressLogging
using SymbolicUtils


function transform_input_old(pmt_pos, particle_pos, particle_dir)

    rot = calc_rot_matrix(pmt_pos, [0, 0, 1])

    part_pos_rot = rot * particle_pos
    part_dir_rot = rot * particle_dir
    
    part_pos_rot_sph = cart_to_sph(part_pos_rot ./ norm(part_pos_rot))
    part_dir_rot_sph = cart_to_sph(part_dir_rot)
    delta_phi = part_pos_rot_sph[2] - part_dir_rot_sph[2]

    return part_pos_rot_sph[1], part_dir_rot_sph[1], delta_phi
end

function transform_input(pmt_pos, particle_pos, particle_dir)

    # Rotate the particle position and direction such that the PMT is at the origin and the z-axis is aligned with the PMT
    rot = calc_rot_matrix(pmt_pos, [0, 0, 1])

    part_pos_rot = rot * particle_pos
    part_dir_rot = rot * particle_dir

    # Convert the rotated particle position to cylindrical coordinates
    pos_cyl = cart_to_cyl(part_pos_rot)

    # Calculate Rotation matrix that rotates the particle direction to the xz-plane
    rotm = RotZ(-pos_cyl[2])

    # Apply the rotation matrix to the particle direction
    part_dir_rot_xz = rotm * part_dir_rot
    part_dir_rot_xz_sph = cart_to_sph(part_dir_rot_xz)

    # We dont have to apply the rotation matrix to the particle position as we are only interested in the zenith angle
    part_pos_rot_sph = cart_to_sph(part_pos_rot ./ norm(part_pos_rot))

    return part_pos_rot_sph[1], part_dir_rot_xz_sph[1], part_dir_rot_xz_sph[2]
end


function _read_single_hd5_file(fname; fit_gamma)

    target = POM(SA_F32[0, 0, 0], 1)
    r = RotMatrix3(I)
    pmt_positions = get_pmt_positions(target, r)

    params = []

    fid = h5open(fname)

    datasets = keys( fid["pmt_hits"])
    
    @progress name="datasets" for ds in datasets
        hits = fid["pmt_hits/$ds"][:, :]
        attributes = attrs(fid["pmt_hits/$ds"])

        pos = attributes["distance"] * sph_to_cart(attributes["pos_theta"], attributes["pos_phi"])
        dir = sph_to_cart(attributes["dir_theta"], attributes["dir_phi"])
        particle = Particle(pos, dir, 0., attributes["energy"], 0, PEMinus)

        if length(hits) == 0
            continue
        end

        if size(hits, 2) != 3
            @show fname, ds
            continue
        end

        df = DataFrame(hits, [:time, :pmt_ix, :weight])
        df[!, :time] .+= randn(length(df.time))*2

        for pmt_ix in 1:16

            pmt_pos = pmt_positions[pmt_ix]

            theta_pos, theta_dir, phi_dir = transform_input(pmt_pos, pos, dir)

            sel = df[df.pmt_ix .== pmt_ix, :]

            tshift = NaN
            gfit_alpha = NaN
            gfit_theta = NaN

            if (nrow(sel) > 10) && fit_gamma
                tshift = minimum(sel[:, :time])
                #@show sel.time .- tshift .+ 1E-10
                gfit = fit_mle(Gamma, sel.time .- tshift .+ 1E-10, sel.weight)
                gfit_alpha = gfit.α
                gfit_theta = gfit.θ
            end

            push!(
                params,
                (
                    nhits = sum(sel.weight, init=0),
                    variance = sum(sel.weight .^2, init=0),
                    pmt_ix=pmt_ix,
                    pos=pos,
                    dir=dir,
                    tshift=tshift,
                    gfit_alpha=gfit_alpha,
                    gfit_theta=gfit_theta,
                    energy=particle.energy,
                    theta_pos=theta_pos,
                    theta_dir=theta_dir,
                    phi_dir=phi_dir,
                    abs_scale=attributes["abs_scale"],
                    sca_scale=attributes["sca_scale"],
                    distance=attributes["distance"],
                )
            )
        
        end
    end

    return DataFrame(params)
end


function load_data(files, fit_gamma=false)
    params = []

    for fname in files
        df = _read_single_hd5_file(fname, fit_gamma=fit_gamma)
        push!(params, df)            
    end

    params = reduce(vcat, params)
    return params
end


struct PoissonLoss <: SupervisedLoss end

function (loss::PoissonLoss)(prediction, target)
    if prediction <= 0
        return Inf
    end

    log_likelihood = -prediction + target * log(prediction) - loggamma(target + 1)
    return -log_likelihood
end

struct LogLPLoss{P} <: SupervisedLoss end

LogLPLoss(p::Number) = LogLPLoss{p}()

function (loss::LogLPLoss{P})(prediction, target) where {P}
    if prediction <= 0
        return Inf
    end

    return (abs(log(prediction+1) - log(target+1)))^P
end

const LogL1Loss = LogLPLoss{1}
const LogL2Loss = LogLPLoss{2}


struct Chi2Loss <: SupervisedLoss end

function (loss::Chi2Loss)(prediction, target)
    return ((prediction - target)^2 / abs(prediction+eps()))
end

struct LogChi2Loss <: SupervisedLoss end

function (loss::LogChi2Loss)(prediction, target)
    return ((log(prediction+1) - log(target+1))^2 / log(prediction+1))
end


function select_loss_func(loss_name)
    if loss_name == "poisson"
        loss_func = PoissonLoss()
    elseif loss_name == "logl2"
        loss_func = LogL2Loss()
    elseif loss_name == "logl1"
        loss_func = LogL1Loss()
    elseif loss_name == "l1"
        loss_func = L1DistLoss()
    elseif loss_name == "l2"
        loss_func = L2DistLoss()
    elseif loss_name == "chi2"
        loss_func = Chi2Loss()
    elseif loss_name == "logchi2"
        loss_func = LogChi2Loss()
    else
        error("Unknown loss function $(loss_name)")
    end
    return loss_func
end


square(x) = x^2
exp_minus(x) = exp(-x)
one_over_square(x) = x^(-2)

function apply_selection(df, selection)
    mask = ones(Bool, nrow(df))
    for sel in selection
        range = sel["range"]
        name = sel["name"]
        mask .&= (df[:, Symbol(name)] .> range[1]) .&& (df[:, Symbol(name)] .< range[2])
    end
    dsel = df[mask, :]
    return dsel
end

function weighted_subsample(df, n, weighting_func)

    weights = weighting_func(df)
    ixs = 1:nrow(df)

    selected_ixs = StatsBase.sample(ixs, Weights(weights), n, replace=false)

    return df[selected_ixs, :]
    
end

function make_weight_func(full_df)

    m1 = full_df.energy .< 1E5
    m2 = (full_df.energy .>= 1E5) .& (full_df.energy .<= 5E6)
    m3 = full_df.energy .> 5E6

    cnt_1 = sum(m1)
    cnt_2 = sum(m2)
    cnt_3 = sum(m3)

    function wfunc(df)

        m1 = df.energy .< 1E5
        m2 = (df.energy .>= 1E5) .& (df.energy .<= 5E6)
        m3 = df.energy .> 5E6


        weight = zeros(nrow(df))

        #weight[m1] .= 1 / ((log10(1E5)-log10(1E2)) )
        #weight[m2] .= 1 / ( (log10(5E6)-log10(1E5)) )
        #weight[m3] .= 1 / ( log10(5E7)-log10(5E6)) 

        weight[m1] .= 1 / cnt_1
        weight[m2] .= 1 / cnt_2
        weight[m3] .= 1 / cnt_3 
    
    
        #weight .*= df.energy .* df.distance
    
        return weight
    end

    return wfunc

end


function plot_variable_slice(X, y, func, fixed_vars, plot_variable, slice_variable, ax=nothing)
    if isnothing(ax)
        fig = Figure()
        ax = Axis(fig[1, 1], yscale=log10, xscale=log10, xlabel="Variable", ylabel="Number of Hits")
    end

    slice_var_ix, slice_points, use_log = slice_variable
    plot_var_ix, plot_var_points = plot_variable

    cmap = cgrad(:viridis)
    get_color_slice(val) = cmap[(val-minimum(slice_points)) / (maximum(slice_points) - minimum(slice_points))]
    
    tf = use_log ? log10 : identity
    itf = use_log ? x -> 10^x : identity

    scatter!(ax, X[plot_var_ix, :], y[:], color=get_color_slice.(tf.(X[slice_var_ix, :])))

    xeval = zeros(size(X, 1), length(plot_var_points))

    for (i, val) in fixed_vars
        xeval[i, :] .= val
    end

    xeval[plot_var_ix, :] .= plot_var_points

    for sp in slice_points
        
        xeval[slice_var_ix, :] .= itf(sp)
        ys = func(xeval)

        lines!(ax, plot_var_points, ys, color=:white, linewidth=5)
        lines!(ax, plot_var_points, ys, color=get_color_slice(sp), linewidth=3)
    end
    ylims!(ax, 1E-3, 1E6)

    return ax
end


function plot_ratio(X, y, func, fixed_vars, plot_variable_x, plot_variable_y, ax=nothing)
    if isnothing(ax)
        fig = Figure()
        ax = Axis(fig[1, 1], yscale=log10, xscale=log10, xlabel="Variable 1", ylabel="Variable 2")
    end

    plot_var_x_ix, plot_var_x_bin_edges = plot_variable_x
    plot_var_y_ix, plot_var_y_bin_edges = plot_variable_y


    hw = Hist2D((X[plot_var_x_ix, :], X[plot_var_y_ix, :]), weights=y, binedges=(plot_var_x_bin_edges, plot_var_y_bin_edges))
    hu = Hist2D((X[plot_var_x_ix, :], X[plot_var_y_ix, :]), binedges=(plot_var_x_bin_edges, plot_var_y_bin_edges))
    h_avg = hw ./ hu

    bc_x = 0.5 * (plot_var_x_bin_edges[1:end-1] .+ plot_var_x_bin_edges[2:end])
    bc_y = 0.5 * (plot_var_y_bin_edges[1:end-1] .+ plot_var_y_bin_edges[2:end])

    xeval = zeros(size(X, 1), length(bc_x)*length(bc_y))

    for (i, val) in fixed_vars
        xeval[i, :] .= val
    end

    lix = LinearIndices((length(bc_x), length(bc_y)))

    for i in eachindex(bc_x)
        for j in eachindex(bc_y)
            flat_x = lix[i, j]
            xeval[plot_var_x_ix, flat_x] = bc_x[i]
            xeval[plot_var_y_ix, flat_x] = bc_y[j]
        end
    end


    yeval = func(xeval)
    yeval = reshape(yeval, length(bc_x), length(bc_y))

    ratio = (h_avg.bincounts .- yeval) ./ yeval
  
    hm = heatmap!(ax, plot_var_x_bin_edges, plot_var_y_bin_edges, ratio, colorrange=(-1, 1), colormap=:RdBu_11)
    return ax, hm
end


function plot_pred_target(X, y, func, ax=nothing)
    
    if isnothing(ax)
        fig = Figure()
        ax = Axis(fig[2, 2], yscale=log10, xscale=log10, xlabel="Truth", ylabel="Prediction")
    end
    yeval = func(X)
    h2d = Hist2D((y, yeval), binedges=(10 .^ (-3:0.1:7), 10 .^ (-3:0.1:7)))
    h2d.bincounts ./= sum(h2d.bincounts, dims=2)
    hm = plot!(ax, h2d, colorscale=log10)
    xs = 10 .^ (-2:0.1:6)
    lines!(ax, xs, xs, color=:black)
    
    #olorbar(fig[2, 1], hm, vertical=false)
    return ax
end


function make_callable(tree, sr_opt)
    return (x) -> eval_tree_array(tree, x, sr_opt)[1]
end

function prepare_data(cfile)

    config = TOML.parsefile(cfile)
    config_name = splitext(basename(cfile))[1]

    dataset_dir = "/home/wecapstor3/capn/capn100h/symbolic_regression/datasets/"
    test_dataset = JLD2.load(joinpath(dataset_dir, config_name, "test.jld2"))["test"]

    X = Matrix(test_dataset[:, Symbol.(config["input"]["fit_variables"])])'
    y = test_dataset.nhits
    w = test_dataset.variance

    # limit to 1E6 samples

    if length(y) > 1000000
        sel = 1:1000000
        X = X[:, sel]
        y = y[sel]
        w = w[sel]
    end

    sel = y .> 1E-3

    ys = y[sel]
    ws = w[sel]
    Xs = X[:, sel]
    weights = ys ./ sqrt.(ws)

    return Xs, ys, weights
end

function read_results(result_dir, verbose=false)


    run_config = jldopen(joinpath(result_dir, "config.jld2"))["config"]
    cfile = run_config["config"]
    config = TOML.parsefile(cfile)
   
    sr_result = load(joinpath(result_dir, "sr_state.jld2"))

    hof = sr_result["hof"]
    sr_opt = sr_result["options"]
    dominating = calculate_pareto_frontier(hof)

    loss_func = select_loss_func(run_config["loss"])

    Xs, ys, weights = prepare_data(cfile)

    sr_summary = []

    for member in dominating
        complexity = compute_complexity(member, sr_opt)
        loss = member.loss
        yeval, did_succeed = eval_tree_array(member.tree, Xs, sr_opt)
       
        equation = make_callable(member.tree, sr_opt)

        eq_string = string_tree(member.tree, sr_opt, variable_names=config["input"]["fit_variables"])
        
        eq_sym = node_to_symbolic(member.tree, sr_opt, variable_names=config["input"]["fit_variables"])

        eq_latex = latexify(string(SymbolicUtils.simplify(eq_sym)))

        push!(sr_summary, (val_loss=sum(loss_func, yeval, ys, weights), train_loss=loss, complexity=complexity, equation=equation, eq_str=eq_string, eq_sym=eq_sym, eq_latex=eq_latex, sucess=did_succeed))
    end

    sr_summary = DataFrame(sr_summary)
    best_val = argmin(sr_summary.val_loss)

    if verbose
        println("Summary of run $(basename(result_dir)):")
        println("Variables: $(config["input"]["fit_variables"])")
        println("Loss: $(run_config["loss"])")
        println("Weights: $(run_config["use_weights"])")
        println("Best Complexity: $(sr_summary.complexity[best_val]) (index: $best_val)")
    end

    return Xs, ys, weights, sr_summary
end

function literaltoreal(x)
    if SymbolicUtils.issym(x)
        return SymbolicUtils.Sym{Real}(nameof(x))
    elseif istree(x) && SymbolicUtils.symtype(x) <: LiteralReal
        return similarterm(x, operation(x), arguments(x), Real)
    else
        return x
    end
end