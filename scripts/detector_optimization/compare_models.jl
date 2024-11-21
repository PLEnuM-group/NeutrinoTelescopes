using Random
using StaticArrays
using PhotonSurrogateModel
using PhotonPropagation
using BSON
using NeutrinoTelescopes
using Rotations
using LinearAlgebra
using PhysicsTools
using CairoMakie
using DataFrames
using HDF5
using JLD2
using CSV
using Distributions
using Format


model_path = "/home/wecapstor3/capn/capn100h/snakemake/time_surrogate_perturb/extended/amplitude_2_FNL.bson"
model = BSON.load(model_path)[:model]

input_size = 10
feat_buffer = create_input_buffer(input_size, 16, 1);


buffer_cpu, buffer_gpu = make_hit_buffers();


mean_sca_angle = 0.95f0
medium = CascadiaMediumProperties(mean_sca_angle, 1f0, 1.0f0)

wl_range = (300f0, 800f0)
spectrum = make_cherenkov_spectrum(wl_range, medium)
#source = FastLightsabreMuonEmitter(p, medium, spectrum)


target = POM(SA[0., 0., 0.], 1)
pmt_positions = get_pmt_positions(target, RotMatrix3(I))


function eval_models(params)

    amplitudes = DataFrame()
    for p in eachrow(params)

        particle_pos = p.distance * sph_to_cart(p.pos_theta, p.pos_phi)
        particle_dir = sph_to_cart(p.dir_theta, p.dir_phi)
        particle_energy = p.energy
        
        create_model_input!(
            particle_pos,
            particle_dir,
            particle_energy,
            target.shape.position, 
            feat_buffer,
            model.transformations,
            abs_scale=1.0,
            sca_scale=1.0)
    
        #feat_buffer[:, 1] .= fourier_input_mapping(feat_buffer[1:10, 1], model.embedding_matrix * hparams.fourier_gaussian_scale)
    
    
        amps_nn = sum(exp.(model(feat_buffer[:, 1])))
    
        amp_sr = 0
        #@show cart_to_sph(particle_dir)
        for ppos in pmt_positions
            this_amp = surrogate(target.shape.position, ppos, particle_pos, particle_dir, particle_energy)
            #@show this_amp
            amp_sr += this_amp
        end
    
        p = Particle(particle_pos, particle_dir, 0, particle_energy, 0., PEPlus)
        p = convert(Particle{Float32}, p)
        source = ExtendedCherenkovEmitter(p, medium, spectrum)
    
    
        setup = PhotonPropSetup([source], [target], medium, spectrum, 1, spectrum_interp_steps=500)
    
        photons = propagate_photons(setup, buffer_cpu, buffer_gpu, 15, copy_output=true)
        hits = make_hits_from_photons(photons, setup, RotMatrix3(I));
        calc_pe_weight!(hits, [target])
    
        amps_prop = sum(hits.total_weight)
    
        push!(amplitudes, (sr = amp_sr, nn=amps_nn, prop=amps_prop))
    end

    return hcat(params, amplitudes)
end

surrogate = SRSurrogateModel()

n_events = 9

params = DataFrame()
for i in 1:n_events
    pos_theta = acos(rand(Uniform(-1, 1)))
    dir_theta = acos(rand(Uniform(-1, 1)))
    pos_phi = rand(Uniform(0, 2π))
    energy = 10 ^ rand(Uniform(2.5, 4.3))
    distance = 10 ^ rand(Uniform(1, 2))


    for phi in range(0, 2π, 15)
        push!(params, (pos_theta=pos_theta, pos_phi=pos_phi, dir_theta=dir_theta, dir_phi=phi, energy=energy, distance=distance, ev_ix=i))
    end
end


amplitudes = eval_models(params)



   
fig = Figure(size=(1200, 1200))

ev_grped = groupby(amplitudes, :ev_ix)

for (groupn, group) in pairs(ev_grped)

    row, col = divrem(groupn.ev_ix-1, 3)
    
    ax = Axis(fig[row+1, col+1], xlabel="Dir Phi", ylabel="Number of Hits",
        title=format("E: {:.2f}, d: {:.2f}, p: ({:.2f}, {:.2f}), t: {:.2f}",
             group.energy[1], group.distance[1], rad2deg(group.pos_theta[1]), rad2deg(group.pos_phi[1]), rad2deg(group.dir_theta[1])))

    lines!(ax, group.dir_phi, group.nn, label="MLP")
    lines!(ax, group.dir_phi, group.sr, label="SR")
    lines!(ax, group.dir_phi, group.prop, label="PhotonProp")
    axislegend(position=:lt)
end
fig

CSV.write("amplitudes.csv", amplitudes)

amplitudes

pos_theta = 0.3
pos_phi = 2.5
distance = 25
pos = sph_to_cart(pos_theta, pos_phi) * distance
particle_zenith = 1.5
particle_energy = 1E4
particle_azimuth = 1.3
particle_dir = sph_to_cart(particle_zenith, particle_azimuth)

target = POM(SA_F32[0., 0., 0.], 1)




pmt_pos = pmt_positions[1]

rot_m = calc_rot_matrix(pmt_pos, [0, 0, 1])

rot_pos = rot_m * pos
rot_dir = rot_m * particle_dir


for phi in 0:0.5:2π

    rot_phi = RotZ(phi)
    rot_pos_phi = rot_phi * rot_pos
    rot_dir_phi = rot_phi * rot_dir

    p = Particle(rot_pos_phi, rot_dir_phi, 0, particle_energy, 0., PEPlus)
    p = convert(Particle{Float32}, p)
    source = ExtendedCherenkovEmitter(p, medium, spectrum)

    setup = PhotonPropSetup([source], [target], medium, spectrum, 1, spectrum_interp_steps=500)

    photons = propagate_photons(setup, buffer_cpu, buffer_gpu, 15, copy_output=true)
    hits = make_hits_from_photons(photons, setup, RotMatrix3(I), false);
    calc_pe_weight!(hits, [target])

    @show sum(hits.total_weight)
end






fid = h5open("/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/photon_table_hits_extended_dmin_1_dmax_200_emin_100_emax_100000.0_5.hd5")

keys(fid)

grp = fid["pmt_hits/dataset_845"]
grp[:, :]

desc = attrs(grp)
desc
particle_pos = desc["distance"] * sph_to_cart(desc["pos_theta"], desc["pos_phi"])
particle_dir =  sph_to_cart(desc["dir_theta"], desc["dir_phi"])
particle_energy = desc["energy"]
p = Particle(particle_pos, particle_dir, 0, particle_energy, 0., PEPlus)
p = convert(Particle{Float32}, p)
target = POM(SA_F32[0., 0., 0.], 1)
source = ExtendedCherenkovEmitter(p, medium, spectrum)
setup = PhotonPropSetup([source], [target], medium, spectrum, 1, spectrum_interp_steps=500)
photons = propagate_photons(setup, buffer_cpu, buffer_gpu, 15, copy_output=true)
hits = make_hits_from_photons(photons, setup, RotMatrix3(I), false);
calc_pe_weight!(hits, [target])
sum(grp[:, 3])
sum(hits.total_weight)

amps = Float64[]
amps_sr = Float64[]
particle_azimuth = 2.5
zeniths = 0:0.01:pi
for zenith in zeniths

    particle_dir = sph_to_cart(zenith, particle_azimuth)
    create_model_input!(
        particle_pos,
        particle_dir,
        particle_energy,
        target.shape.position, 
        feat_buffer,
        model.transformations,
        abs_scale=1.0,
        sca_scale=1.0)

    #feat_buffer[:, 1] .= fourier_input_mapping(feat_buffer[1:10, 1], model.embedding_matrix * hparams.fourier_gaussian_scale)


    push!(amps, sum(exp.(model(feat_buffer[1:10, 1]))))

    amp_sr = 0
    for ppos in pmt_positions
        amp_sr += surrogate(target.shape.position, ppos, particle_pos, particle_dir, particle_energy)
    end

    push!(amps_sr, amp_sr)
end

fig, ax, l = CairoMakie.lines(zeniths, amps, label="Model")
lines!(ax, zeniths, amps_sr, label="Surrogate")
fig