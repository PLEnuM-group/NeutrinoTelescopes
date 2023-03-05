using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
using CairoMakie
using Distributions
using Random
using BSON
using Flux
using StaticArrays
using DataStructures
using JSON

BSON.@load joinpath(@__DIR__, "../data/extended_cascade_2_FNL.bson") model tf_vec


targets_single = [make_pone_module(@SVector[-25., 0., -450.], 1)]
targets_line = make_detector_line(@SVector[-25., 0.0, 0.0], 20, 50, 1)
targets_three_l = [
    make_detector_line(@SVector[-25., 0.0, 0.0], 20, 50, 1)
    make_detector_line(@SVector[25., 0.0, 0.0], 20, 50, 21)
    make_detector_line(@SVector[0., sqrt(50^2-25^2), 0.0], 20, 50, 41)]
targets_hex = make_hex_detector(3, 50, 20, 50, truncate=1)

medium = make_cascadia_medium_properties(0.99)
d = Detector(targets_hex, medium)

cylinder = get_bounding_cylinder(d)
pdist = CategoricalSetDistribution(OrderedSet([PEMinus, PEPlus]), [0.5, 0.5])
edist = Pareto(1, 1E4) + 1E4
ang_dist = UniformAngularDistribution()
length_dist = Dirac(0.)
time_dist = Dirac(0.)
inj = VolumeInjector(cylinder, edist, pdist, ang_dist,length_dist, time_dist)

event = rand(inj)
hit_generator = SurrogateModelHitGenerator(model, tf_vec, 200., d)

generate_hit_times!(event, d, hit_generator)
