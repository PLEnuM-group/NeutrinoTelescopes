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
using ArgParse
using JSON3

include("utils.jl")

function resample_dataset(infile, outfile, n_resample)

    group = "photons"
    
    fid = h5open(infile, "r")
    datasets = keys(fid[group])
    close(fid)

    for ds in datasets
    
        fid = h5open(infile)
        photons = DataFrame(
            read(fid[group][ds]),
            [:tres, :pos_x, :pos_y, :pos_z, :dir_x, :dir_y, :dir_z, :total_weight, :module_id, :wavelength]
        )


        sim_attrs = Dict(attrs(fid[group][ds]))

        direction::SVector{3,Float32} = sph_to_cart(acos(sim_attrs["dir_costheta"]), sim_attrs["dir_phi"])
        ppos =  JSON3.read(sim_attrs["source_pos"], SVector{3, Float32})

        target = POM(SA_F32[0, 0, 0], UInt16(1))
        
        close(fid)

        for _ in 1:n_resample
            #=
            PMT positions are defined in a standard upright coordinate system centeres at the module
            Sample a random rotation matrix and rotate the pmts on the module accordingly.
            =#
            orientation = rand(RotMatrix3)
            hits = make_hits_from_photons(photons, [target], orientation)

           

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

            if nrow(hits) == 0
                out_mat = Matrix{Float64}(undef, (0, 2))        
            else
                out_mat = Matrix{Float64}(hits[:, [:tres, :pmt_id]])
            end

            save_hdf!(
                outfile,
                "pmt_hits",
                out_mat,
                sim_attrs)
        end
    end
end


s = ArgParseSettings()

@add_arg_table s begin
    "--infile"
    help = "Input filename"
    arg_type = String
    required = true
    "--outfile"
    help = "Output filename"
    arg_type = String
    required = true
    "--resample"
    help = "Number of resamples"
    arg_type = Int64
    default = 100
end

parsed_args = parse_args(ARGS, s)

resample_dataset(parsed_args["infile"], parsed_args["outfile"], parsed_args["resample"])


