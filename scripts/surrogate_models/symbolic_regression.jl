using SymbolicRegression
using MLUtils
using HDF5
using DataFrames
using ClusterManagers
using Distributed
using JLD2

addprocs_slurm(40)

@everywhere using SymbolicRegression, LoopVectorization, SpecialFunctions, Bumper, SymbolicUtils

@everywhere function poisson_loss(prediction, target)
    if prediction < 0
        return Inf
    end

    log_likelihood = -prediction + target * log(prediction + 1e-20) - loggamma(target + 1)

    return -log_likelihood
end


data = JLD2.load("/home/wecapstor3/capn/capn100h/sr_dataset.jld2")

X = data["X"]
y = data["y"]

Xs, ys = shuffleobs((X, y))

@everywhere Xs = $Xs
@everywhere ys = $ys


workers()

opt = Options(
    binary_operators=[+, *, /, -, ^],
    unary_operators=[exp, abs, log, cos, tan],
    populations=120,
    population_size=100,
    batching=true,
    batch_size=3000,
    elementwise_loss=poisson_loss,
    #elementwise_loss=HuberLoss(1),
    maxsize=40,
    turbo = true,
    nested_constraints = [
        abs => [abs => 0, exp => 0,],
        log => [log => 0, exp => 0],
        exp => [exp => 0, log => 0],
        cos => [cos => 0],
        tan => [tan => 0]],
    ncycles_per_iteration=5000,
    should_simplify=true,
    mutation_weights = MutationWeights(optimize=0.001),
    bumper = true
)


state, hof = equation_search(
    Xs,
    ys,
    niterations=100,
    options=opt,
    parallelism=:multiprocessing,
    procs = workers(),
    #numprocs=40,
    variable_names=["energy", "distance", "theta_pos", "theta_dir", "delta_phi"],
    #X_units=["Constants.GeV", "m", "", "", ""],
    runtests=false,
    return_state=true
)

jldsave("/home/wecapstor3/capn/capn100h/sr_state_$(now()).jld2", state=state, hof=hof)


