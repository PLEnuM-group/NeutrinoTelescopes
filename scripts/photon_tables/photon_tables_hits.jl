using NeutrinoTelescopes
using Random
using DataFrames
using ProgressLogging
using Rotations
using LinearAlgebra
using ArgParse
using PhysicsTools
using PhotonPropagation
using HDF5
using StaticArrays

include("utils.jl")

function resample_dataset(fname)

    group = "photons"
    
    fid = h5open(fname, "r")
    datasets = keys(fid[group])
    close(fid)

    for ds in datasets
    
        fid = h5open(fname)
        photons = DataFrame(
            fid[group][ds],
            [:tres, :pos_x, :pos_y, :pos_z, :total_weight]
        )

        ats = attrs(fid[group][ds])

        direction::SVector{3,Float32} = sph_to_cart(acos(ats["dir_costheta"]), ats["dir_phi"])
        ppos =  JSON.read(ats["source_pos"], SVector{3, Float32})

        setup = make_setup(
            Symbol(ats["mode"]),
            ppos,
            direction,
            ats["energy"],
            ats["seed"];
            g=Float32(ats["g"])
        )
        
        close(fid)

        for _ in 1:n_resample
            #=
            PMT positions are defined in a standard upright coordinate system centeres at the module
            Sample a random rotation matrix and rotate the pmts on the module accordingly.
            =#
            orientation = rand(RotMatrix3)
            hits = make_hits_from_photons(photons, setup, orientation)

            if nrow(hits) < 10
                continue
            end

            #=
            Rotating the module (active rotation) is equivalent to rotating the coordinate system
            (passive rotation). Hence rotate the position and the direction of the light source with the
            inverse rotation matrix to obtain a description in which the module axis is again aligned with ez
            =#
            direction_rot = orientation' * direction
            position_rot = orientation' * ppos

            position_rot_normed = position_rot ./ norm(position_rot)
            dir_theta, dir_phi = cart_to_sph(direction_rot)
            pos_theta, pos_phi = cart_to_sph(position_rot_normed)

            #= Sanity check:
            if !((dot(ppos / norm(ppos), direction) ≈ dot(position_rot_normed, direction_rot)))
                error("Relative angle not preserved: $(dot(ppos / norm(ppos), direction)) vs. $(dot(position_rot_normed, direction_rot))")
            end
            =#

            sim_attrs["dir_theta"] = dir_theta
            sim_attrs["dir_phi"] = dir_phi
            sim_attrs["pos_theta"] = pos_theta
            sim_attrs["pos_phi"] = pos_phi

            save_hdf!(
                fname,
                "pmt_hits",
                Matrix{Float64}(hits[:, [:tres, :pmt_id]]),
                sim_attrs)
        end
    
    
    end
end


