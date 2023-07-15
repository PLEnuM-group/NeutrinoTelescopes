module Weighting

export get_total_prob, get_transmission_prob, get_interaction_prob, get_xsec
export WeighterPySpline

using PyCall

const np = PyNULL()
const pickle = PyNULL()


function __init__()
    np_tmp = pyimport("numpy")
    pickle_tmp = pyimport("pickle")

    copy!(np, np_tmp)
    copy!(pickle, pickle_tmp)
end

abstract type Weighter end

struct WeighterPySpline <: Weighter
    tprob::PyObject
    xsec::PyObject
end

function WeighterPySpline(fname::String)
    splines = pickle.load(py"open"(fname, "rb"))

    tprob = splines["transmission_prob"]
    xsec = splines["xsec"]

    return WeighterPySpline(tprob, xsec)
end

get_xsec(w::Weighter, log10_energy) = 10^(first(w.xsec(log10_energy))) * 1E-4 # m^


function get_interaction_prob(w::Weighter, log10_energy, length)
    molar_mass_water = 18.01528 # g / mol
    density = 1000 # kg / m^3
    na = 6.02214076E23 # 1 / mol
    n_nucleons = 18 * density / (molar_mass_water / 1000) * na 

    xs = get_xsec(w, log10_energy)

    interaction_coeff = xs * n_nucleons

    #taylor: 1 - exp(-x) ~ x
    int_prob = interaction_coeff * length
    return int_prob
end

get_transmission_prob(w, log10_energy, cos_theta) = first(w.tprob(cos_theta, log10_energy, grid=false))
get_total_prob(w, log10_energy, cos_theta, length) = get_transmission_prob(w, log10_energy, cos_theta) * get_interaction_prob(w, log10_energy, length)
    


end
