using SymbolicRegression
using MLUtils
using HDF5
using DataFrames
using ClusterManagers
using Distributed
using JLD2
using Dates
using ArgParse
using Logging
using TOML
using PoissonRandom


s = ArgParseSettings()
@add_arg_table s begin
    "--continue"
    help = "Continue from previous state"
    "--config"
    help = "Config file"
    required=true
    "--multiproc"
    action = :store_true
    help = "Use multiprocessing"
end

parsed_args = parse_args(ARGS, s)

if parsed_args["multiproc"]

    n_workers = parse(Int64, ENV["SLURM_CPUS_ON_NODE"]) * parse(Int64, ENV["SLURM_JOB_NUM_NODES"])
    this_node = gethostname()

    @info "Head worker running on $(this_node)"
    @info "Adding $n_workers workers"
    addprocs_slurm(n_workers)

    hostname_worker = Dict()

    for w in workers()
        hname = fetch(@spawnat w gethostname())
        if !haskey(hostname_worker, hname)
            hostname_worker[hname] = []
        end
        push!(hostname_worker[hname], w)
    end


    removed_worker = pop!(hostname_worker[this_node])
    @info "Removing worker $(removed_worker)"
    rmprocs(removed_worker)
else
    n_workers = Threads.nthreads()
end

@everywhere using SymbolicRegression, LoopVectorization, SpecialFunctions, Bumper, SymbolicUtils
@everywhere include("utils.jl")

config = TOML.parsefile(parsed_args["config"])

outdir = config["output"]["dir"]

if !isdir(outdir)
    mkdir(outdir)
end

if isfile(joinpath(outdir, "dataset.jld2"))
    dsel = JLD2.load(joinpath(outdir, "dataset.jld2"))["data"]
else
    data = JLD2.load(config["input"]["filename"])["data"]
    dsel = apply_selection(data, config["selection"])
    jldsave(joinpath(outdir, "dataset.jld2"), data=dsel)
end

X = Matrix(dsel[:, Symbol.(config["run"]["variables"])])'
y = dsel.nhits
y_resampled = pois_rand.(y)

Xs, ys = shuffleobs((X, y_resampled))

@everywhere Xs = $Xs
@everywhere ys = $ys

unary_operators = [exp, log, exp_minus, one_over_square, square]
nested_constraints = [
    log => [log => 0, exp => 0],
    exp => [exp => 0, log => 0, exp_minus => 0],
    exp_minus => [exp_minus => 0, exp => 0],
    one_over_square => [one_over_square => 0]
]


run_config = get(config, "run", Dict())

if get(run_config, "use_trig", false)
    push!(unary_operators, cos)
    push!(unary_operators, tan)
    push!(unary_operators, tanh)

    push!(nested_constraints, cos => [cos => 0, tan => 0])
    push!(nested_constraints, tan => [tan => 0, cos => 0])
end



opt = Options(
    binary_operators=[+, *, /, -, ^],
    unary_operators=unary_operators,
    populations=3*n_workers,
    population_size=100,
    batching=true,
    batch_size=500,
    elementwise_loss=poisson_loss,
    #elementwise_loss=HuberLoss(1),
    maxsize=25,
    #maxdepth=5,
    turbo = true,
    nested_constraints = nested_constraints,
    ncycles_per_iteration=800,
    should_simplify=true,
    mutation_weights = MutationWeights(optimize=0.001),
    bumper = true,
    #warmup_maxsize_by=0.1,
    complexity_of_constants=2,
    complexity_of_variables=2,
    complexity_of_operators = [
        (^) => 2        
    ],
    output_file = joinpath(outdir, "hof_$(now).csv")

)

state = nothing

if !isnothing(parsed_args["continue"])
    f = load(parsed_args["continue"])
    state = (f["state"], f["hof"])
end


if parsed_args["multiproc"]

    state, hof = equation_search(
        Xs,
        ys,
        niterations=500,
        options=opt,
        parallelism=:multiprocessing,
        procs = workers(),
        #numprocs=40,
        #variable_names=["energy", "distance", "theta_pos", "theta_dir", "delta_phi", "abs_scale", "sca_scale"],
        variable_names=["energy", "distance"],
        #X_units=["Constants.GeV", "m", "", "", "", "", ""],
        runtests=false,
        return_state=true,
        verbosity=2,
        saved_state=state
    )

else
    state, hof = equation_search(
        Xs,
        ys,
        niterations=500,
        options=opt,
        parallelism=:multithreading,
        #variable_names=["energy", "distance", "theta_pos", "theta_dir", "delta_phi", "abs_scale", "sca_scale"],
        variable_names=["energy", "distance", "abs_scale"],
        #X_units=["Constants.GeV", "m", "", "", "", "", ""],
        runtests=false,
        return_state=true,
        verbosity=2,
        saved_state=state
    )
end



jldsave(joinpath(outdir), "sr_state_$(now()).jld2", state=state, hof=hof, options=opt)