module Weighting

export get_total_prob, get_transmission_prob, get_interaction_prob, get_xsec
export WeighterPySpline

using PyCall
using PhysicsTools

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
    tprob_nue::PyObject
    tprob_numu::PyObject
    tprob_nutau::PyObject
    tprob_nuebar::PyObject
    tprob_numubar::PyObject
    tprob_nutaubar::PyObject
    xsec_nu_CC::PyObject
    xsec_nu_NC::PyObject
    xsec_nubar_CC::PyObject
    xsec_nubar_NC::PyObject
    total_xsec_nu_CC::PyObject
    total_xsec_nu_NC::PyObject
    total_xsec_nubar_CC::PyObject
    total_xsec_nubar_NC::PyObject
end

function WeighterPySpline(fname::String)
    splines = pickle.load(py"open"(fname, "rb"))

    tprob = splines["transmission_prob"]
    xsec = splines["xsec"]

    return WeighterPySpline(
        tprob["nue"], tprob["numu"], tprob["nutau"],
        tprob["nuebar"], tprob["numubar"], tprob["nutaubar"],
        xsec["nu_cc"], xsec["nu_nc"], xsec["nubar_cc"], xsec["nubar_nc"],
        xsec["nu_cc_total"], xsec["nu_nc_total"], xsec["nubar_cc_total"], xsec["nubar_nc_total"])
end

function get_total_xsec(w::Weighter, int_type::Symbol, log10_energy)
    if int_type == :NU_CC
        xsec = w.total_xsec_nu_CC
    elseif int_type == :NU_NC
        xsec = w.total_xsec_nu_NC
    elseif int_type == :NUBAR_CC
        xsec = w.total_xsec_nubar_CC
    elseif int_type == :NUBAR_NC
        xsec = w.total_xsec_nuvar_NC
    else
        error("Unknown interaction type $int_type")
    end

    return 10^(first(xsec(log10_energy))) * 1E-4 # m^2
end


function get_interaction_prob(w::Weighter, int_type::Symbol, log10_energy, length)
    molar_mass_water = 18.01528 # g / mol
    density = 1000 # kg / m^3
    na = 6.02214076E23 # 1 / mol
    n_nucleons = 18 * density / (molar_mass_water / 1000) * na 

    xs = get_total_xsec(w, int_type, log10_energy)

    interaction_coeff = xs * n_nucleons

    #taylor: 1 - exp(-x) ~ x
    int_prob = interaction_coeff * length
    return int_prob
end

function get_transmission_prob(w::Weighter, ::T, log10_energy, cos_theta) where {T <:ParticleType}
    if T <: PNuE
        tprob = w.tprob_nue
    elseif T <: PNuEBar
        tprob = w.tprob_nuebar
    elseif T <: PNuMu
        tprob = w.tprob_numu
    elseif T <: PNuMuBar
        tprob = w.tprob_numubar
    elseif T <: PNuTau
        tprob = w.tprob_nutau
    elseif T <: PNuTauBar
        tprob = w.tprob_nutaubar
    else
        error("Unknown particle type $T")
    end

    return first(tprob(cos_theta, log10_energy, grid=false))
end

get_total_prob(w, ptype, int_type, log10_energy, cos_theta, length)  = get_transmission_prob(w, ptype, log10_energy, cos_theta) * get_interaction_prob(w, int_type, log10_energy, length)
    


end
