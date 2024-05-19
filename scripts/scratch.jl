using JLD2
using DataFrames
using HDF5
using NeutrinoTelescopes
using StatsBase

fname = "/home/wecapstor3/capn/capn100h/snakemake/training_inputs/time_input__perturb_extended.jld2"

data = jldopen(fname)

data["hits"]
data["features"]

out = h5open("/home/wecapstor3/capn/capn100h/snakemake/training_inputs/time_input__perturb_extended.hd5", "w")
out["hits"] = data["hits"]
out["features"] = data["features"]
close(out)
close(data)

f = h5open("/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/photon_table_hits_extended_dmin_1_dmax_200_emin_100000.0_emax_5000000.0_0.hd5")

mkdir("/home/wecapstor3/capn/capn100h/share_for_rasmus/")


h5open("/home/wecapstor3/capn/capn100h/share_for_rasmus/test.hdf5", "w") do fout
    for groupn in keys(f["pmt_hits"])
        grp = f["pmt_hits"][groupn]

        sumw = sum(grp[:, 3])
        weights = FrequencyWeights(grp[:, 3], sumw)
        grplen = size(grp, 1)

        outlen = min(grplen, sumw)
        sampled = sample(1:grplen, weights, ceil(Int64, outlen), replace=true)
        resampled_hits = grp[:, :][sampled, 1:2]
        out_vector = zeros(Float32, 26)
        PhotonSurrogates._convert_grp_attrs_to_features!(attrs(grp), out_vector)

        fout[groupn] = resampled_hits

        out_attrs = Dict(attrs(grp))
        out_attrs["transformed_features"] = out_vector
        out_attrs["total_hits"] = sumw

        f_attrs = HDF5.attrs(fout["dataset_1"])

        for (k, v) in pairs(out_attrs)
            f_attrs[k] = v
        end
    end
end



