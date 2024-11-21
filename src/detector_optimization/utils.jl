
export poisson_atleast_one, poisson_atleast_k

function poisson_atleast_one(mu)
    return 1 - exp(-mu)
end

function poisson_atleast_k(mu, k)
    return 1 - sum(exp(-mu) * mu^i / factorial(i) for i in 0:k)
end