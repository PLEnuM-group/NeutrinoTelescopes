using NeutrinoTelescopes
using Enzyme
using PhotonPropagation
using PhysicsTools
using Flux
using BSON: @load
using Random
using StaticArrays
using LinearAlgebra
using Base.Iterators

@load joinpath(ENV["WORK"], "time_surrogate/extended/amplitude_1_FNL.bson") model hparams tf_vec


medium = make_cascadia_medium_properties(0.95f0)

c_n = c_at_wl(800.0f0, medium)
rng = MersenneTwister(31338)

pos = SA[-25.0, 5.0, -460]
dir_theta = 0.1
dir_phi = 0.3
dir = sph_to_cart(dir_theta, dir_phi)

p = Particle(pos, dir, 0.0, 1E4, 0.0, PEMinus)

t = POM(SA[0., 0., 0.], 1)

output = zeros(24, 16)


flow_out = ExtendedCascadeModel.calc_flow_input!([p], [t], tf_vec, output)

function wrapped(pos_x, pos_y, pos_z, output, tf_vec)
    dir_theta = 0.1
    dir_phi = 0.3
    dir = sph_to_cart(dir_theta, dir_phi)
    p = Particle(SA[pos_x, pos_y, pos_z], dir, 0.0, 1E4, 0.0, PEMinus)
    t = POM(SA[0., 0., 0.], 1)
    
    if size(output, 1) != 8 + 16
        error("Output buffer gas wrong size")
    end

    targets = [t]
    particles = [p]

    n_pmt = get_pmt_count(eltype(targets))
    out_ix = LinearIndices((1:n_pmt, eachindex(particles), eachindex(targets)))
    
    for (p_ix, t_ix) in product(eachindex(particles), eachindex(targets))
        particle = particles[p_ix]
        target = targets[t_ix]
        rel_pos = particle.position .- target.shape.position
        # TODO: Remove hardcoded max distance
        dist = clamp(norm(rel_pos), 0., 200.)
        normed_rel_pos = rel_pos ./ dist
      
        @inbounds for pmt_ix in 1:n_pmt

            ix = out_ix[pmt_ix, p_ix, t_ix]

            output[1, ix] = log(dist)
            output[2, ix] = log(particle.energy)
            output[3:5, ix] = particle.direction
            output[6:8, ix] = normed_rel_pos
            output[9:24, ix] = (pmt_ix .== 1:16)

        end
    end

    return output
end

@code_llvm wrapped(0., 0., 1., output, tf_vec)

autodiff(Reverse, wrapped, Active, Active(0), Active(1), Active(2), Duplicated(output, output), Const(tf_vec))