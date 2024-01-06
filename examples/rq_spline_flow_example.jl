using NeutrinoTelescopes
using CairoMakie

num_bins = 5
x = -5:0.1:5
params = repeat(randn(3 * num_bins + 1), 1, length(x))
x_pos, y_pos, knot_slopes = constrain_spline_params(params, -5, 5)

normal_logpdf = -0.5 .* (x .^ 2 .+ log(2 * pi))

fig = Figure()
ax = Axis(fig[1,1], xlabel="x [y]", ylabel="Density")
ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y")
lines!(ax, x, exp.(normal_logpdf), label=L"\pi(x)")
xrt, logdet_inv = inv_rqs_univariate(x_pos, y_pos, knot_slopes, x)
lines!(ax, x, exp.(normal_logpdf .+ logdet_inv), label=L"p(y)")
lines!(ax2, x, xrt, label=L"f^{-1}(x)")
CairoMakie.scatter!(ax2, y_pos[:, 1], x_pos[:, 1], label="Knots")
axislegend(ax)
axislegend(ax2, position=:lt)

fig


lines!(ax, x, exp.(normal_logpdf))

x_pos



normal_logpdf = -log.(scale) .- 0.5 .* (log(2 * π) .+ ((x .- shift) ./ scale) .^ 2)










y, logdet = rqs_univariate(x_pos, y_pos, knot_slopes, x)
xrt, logdet_inv = inv_rqs_univariate(x_pos, y_pos, knot_slopes, y)