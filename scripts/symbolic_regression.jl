using SymbolicRegression
using SymbolicUtils
using NeutrinoTelescopes
using Random
using SpecialFunctions
using CairoMakie
using HDF5


fname = "/home/saturn/capn/capn100h/snakemake/photon_tables/lightsabre/hits/photon_table_lightsabre_0_hits.hd5"



features, nhits = h5open(fname, "r") do fid

    datasets = keys(fid["pmt_hits"])

    features = Matrix{Float64}(undef, 8, length(datasets))
    nhits = Matrix{Float64}(undef, 16, length(datasets))

    for (i, grpn) in enumerate(datasets)
        grp = fid["pmt_hits"][grpn]
        f,h = ExtendedCascadeModel.count_hit_per_pmt(grp)

        nhits[:, i] .= h
        features[:, i] .= f
    end

    return features, nhits
end

#lossf(log_lambda, n) = exp(-(n * log_lambda - exp(log_lambda))) #- loggamma(n + 1.0)

options = Options(
    binary_operators=[+, *, /, -, ^],
    unary_operators=[cos, exp, log, sin, tan],
    populations=100,
    population_size=50,
    batching=true,
    maxsize=40,
    #elementwise_loss=lossf,
    should_optimize_constants=true,
    annealing=true,
    nested_constraints = [cos => [cos => 0], sin => [sin => 0], tan => [tan => 0],
        log => [exp => 0]],
    complexity_of_operators=[cos => 2, exp => 2, log => 2, sin => 2, tan => 2]
)


y = nhits[1:1, :]
X = features[:, :]


hall_of_fame = equation_search(
    X, y, niterations=50, options=options,
    parallelism=:multiprocessing
)

dominating = calculate_pareto_frontier(hall_of_fame)

eqn = node_to_symbolic(dominating[end].tree, options)


eval_tree_array(dominating[end].tree, X, options)[1]
y


exp.(eval_tree_array(dominating[end].tree, X, options)[1])


fig, ax = scatter(X[1, :], eval_tree_array(dominating[end].tree, X, options)[1])
scatter!(ax, X[1, :], y[1, :])
fig


eqn_callable
println("Complexity\tMSE\tEquation")

for member in dominating
    complexity = compute_complexity(member, options)
    loss = member.loss
    string = string_tree(member.tree, options)

    println("$(complexity)\t$(loss)\t$(string)")
end