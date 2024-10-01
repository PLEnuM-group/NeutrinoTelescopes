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
using SHA
using Random

loss_functions = ["poisson", "l2", "l1", "logl1", "logl2"]

s = ArgParseSettings()
@add_arg_table s begin
    "--continue"
    help = "Continue from previous state"
    "--config"
    help = "Config file"
    required=true
    "--train"
    help = "Training dataset"
    required=true
    "--output"
    help = "Output dir"
    required=true
    "--multiproc"
    action = :store_true
    help = "Use multiprocessing"
    "--use_trig"
    action = :store_true
    help = "Use trig function"
    "--use_weights"
    action = :store_true
    help = "Use weighted loss"
    "--loss"
    help = "Loss functions to use"
    required=true
    range_tester = in(loss_functions)
    "--niterations"
    help = "Number of SR iterations"
    required=true
    arg_type = Int64
    "--maxsize"
    help = "Max equation size"
    required=true
    arg_type = Int64
    "--verbose"
    action = :store_true
    help = "Show progress"
    "--use_dim_constraints"
    action = :store_true
    help = "Use dimensional constraints"
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

chash = bytes2hex(sha256(read(parsed_args["config"])))
config = TOML.parsefile(parsed_args["config"])

outdir = parsed_args["output"]

if !isdir(outdir)
    mkdir(outdir)
end


if isfile(parsed_args["train"])
    train = JLD2.load(parsed_args["train"])["train"]
else
    error("Training dataset not found.")
end

run_config = get(config, "run", Dict())

X = Matrix(train[:, Symbol.(config["input"]["fit_variables"])])'
y = train.nhits
w = train.variance

loss_func = select_loss_func(parsed_args["loss"])

if parsed_args["loss"] == "poisson"
    y = pois_rand.(y)
end
sel = y .> 1E-3
X = X[:, sel]
y = y[sel]
w = w[sel]

if any(.!isfinite.(X)) || any(.!isfinite.(y)) || any(.!isfinite.(w))
    error("Non-finite values in training data")
end


Xs, ys, ws = shuffleobs((X, y, w))

@everywhere Xs = $Xs
@everywhere ys = $ys
@everywhere ws = $ws


if parsed_args["use_trig"]
    unary_operators = [exp, log, exp_minus, one_over_square, square, cos, tan, sqrt]
    nested_constraints = [
        log => [log => 0, exp => 0,],
        exp => [exp => 0, log => 0, exp_minus => 0,],
        exp_minus => [exp_minus => 0, exp => 0, log => 0],
        one_over_square => [one_over_square => 0, square => 0],
        square => [exp => 0, exp_minus => 0, log => 0, one_over_square => 0],
        sqrt => [sqrt => 1, exp => 0, exp_minus => 0, log => 0, one_over_square => 0],
        cos => [cos => 0, tan => 0, exp => 0, exp_minus => 0, square => 0, log => 0],
        tan => [tan => 0, cos => 0, exp => 0, exp_minus => 0, square => 0, log => 0],
        square => [exp => 0, exp_minus => 0, log => 0, one_over_square => 0, square => 1],
        (^) => [exp => 0, exp_minus => 0, log => 0]
    ]
else
    unary_operators = [exp, log, exp_minus, one_over_square, square, sqrt]
    nested_constraints = [
        log => [log => 0, exp => 0],
        exp => [exp => 0, log => 0, exp_minus => 0],
        exp_minus => [exp_minus => 0, exp => 0, log => 0],
        one_over_square => [one_over_square => 0, square => 0],
        square => [exp => 0, exp_minus => 0, log => 0, one_over_square => 0, square => 1],
        sqrt => [sqrt => 1, exp => 0, exp_minus => 0, log => 0, one_over_square => 0],
        (^) => [exp => 0, exp_minus => 0, log => 0]
    ]
end

X_units = nothing
dimensional_constraint_penalty = nothing
if parsed_args["use_dim_constraints"]
    X_units = config["input"]["units"]
    dimensional_constraint_penalty = 10^5
end

opt = Options(
    binary_operators=[+, *, /, -, ^],
    unary_operators=unary_operators,
    populations=3*n_workers,
    population_size=150,
    batching=true,
    batch_size=1000,
    elementwise_loss=loss_func,
    #elementwise_loss=HuberLoss(1),
    maxsize=parsed_args["maxsize"],
    #maxdepth=7,
    turbo = true,
    nested_constraints = nested_constraints,
    ncycles_per_iteration=750,
    should_simplify=true,
    mutation_weights = MutationWeights(optimize=0.005),
    bumper = true,
    warmup_maxsize_by=0.3,
    complexity_of_constants=1,
    complexity_of_variables=1,
    complexity_of_operators = [
        (^) => 2,
    ],
    parsimony = 0.002,
    adaptive_parsimony_scaling=100,
    output_file = joinpath(outdir, "hof_$(now()).csv"),
    progress=parsed_args["verbose"],
    fraction_replaced_hof=0.1,
    dimensional_constraint_penalty=dimensional_constraint_penalty

)

state = nothing

if !isnothing(parsed_args["continue"])
    f = load(parsed_args["continue"])
    state = (f["state"], f["hof"])
end

weights = nothing
if parsed_args["use_weights"]
    weights = ys ./ sqrt.(ws)
    if any(.!isfinite.(weights))
        error("Non-finite weights")
    end
end





jldsave(joinpath(parsed_args["output"], "config.jld2"), config=parsed_args)



if parsed_args["multiproc"]

    state, hof = equation_search(
        Xs,
        ys,
        weights=weights,
        niterations=parsed_args["niterations"],
        options=opt,
        parallelism=:multiprocessing,
        procs = workers(),
        variable_names=config["input"]["fit_variables"],
        X_units=X_units,
        runtests=false,
        return_state=true,
        saved_state=state
    )

else
    state, hof = equation_search(
        Xs,
        ys,
        weights=weights,
        niterations=parsed_args["niterations"],
        options=opt,
        parallelism=:multithreading,
        variable_names=config["input"]["fit_variables"],
        X_units=X_units,
        runtests=false,
        return_state=true,
        saved_state=state
    )
end

jldsave(joinpath(parsed_args["output"], "sr_state.jld2"), state=state, hof=hof, options=opt)