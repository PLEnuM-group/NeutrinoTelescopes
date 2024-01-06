
using CSV
using DataFrames
using BSplineKit
using Polynomials

df = DataFrame(CSV.File(joinpath(@__DIR__, "../assets/duran_1mm.csv")))
df2 = DataFrame(CSV.File(joinpath(@__DIR__, "../assets/duran_2mm.csv")))
df3 = DataFrame(CSV.File(joinpath(@__DIR__, "../assets/duran_8mm.csv")))

function make_interp(df)
    sargs = sortperm(df[:, :wavelength])
    y = df[sargs, :transmissivity]
    y[y .== 0] .= 1E-7
    y = log.(y ./ 100)
    itp = interpolate(df[sargs, :wavelength], y, BSplineOrder(3))
    return itp
end


itp = make_interp(df)
itp2 = make_interp(df2)
itp3 = make_interp(df3)

wl_eval = 250:1:1000

ys = [itp.(wl_eval), itp2.(wl_eval), itp3.(wl_eval)]
ys = reduce(hcat, ys)
xs = [1, 2, 8]

polys = Polynomials.fit.(Ref(xs), eachrow(ys), 1) 

p_eval = [p(14) for p in polys]

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Wavelength (nm)", ylabel="Transmissivity %")

lines!(ax, wl_eval, exp.(itp.(wl_eval)), label="1mm")
lines!(ax, wl_eval, exp.(itp2.(wl_eval)), label="2mm")
lines!(ax, wl_eval, exp.(itp3.(wl_eval)), label="8mm")
lines!(ax, wl_eval, exp.(p_eval), label="14mm (extp)")

pmt_acc = DataFrame(CSV.File(joinpath(@__DIR__, "../assets/PMTAcc.csv"), header=["wavelength", "acceptance"]))

lines!(ax, pmt_acc[:, :wavelength], pmt_acc[:, :acceptance] , label="PMT")
Legend(fig[1, 2], ax)
fig

wl_ts = DataFrame(wavelength=wl_eval, transmissivity=exp.(p_eval))
wl_ts[!, :lambda_abs] .= -14E-3 ./ log.(wl_ts[!, :transmissivity])


CSV.write(joinpath(@__DIR__, "../assets/duran_wl_acc_14mm.csv"), wl_ts)




xs = 250:1:800
fig, ax = lines(df[sargs, :wavelength], df[sargs, :transmissivity])
lines!(ax, xs, itp.(xs))
xlims!(ax, 200, 1000)
fig

xs = 250:1:800
sargs = sortperm(df2[:, :wavelength])
fig, ax = lines(df2[sargs, :wavelength], df2[sargs, :transmissivity])
lines!(ax, xs, itp2.(xs))
xlims!(ax, 200, 1000)
fig

