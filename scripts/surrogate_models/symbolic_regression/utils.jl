using SpecialFunctions

function poisson_loss(prediction, target)
    if prediction <= 0
        return Inf
    end

    log_likelihood = -prediction + target * log(prediction) - loggamma(target + 1)

    return -log_likelihood
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