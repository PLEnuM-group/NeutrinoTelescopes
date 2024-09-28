using JLD2
using DataFrames
using HDF5
using NeutrinoTelescopes
using StatsBase
using PhotonSurrogateModel
using Glob

files = glob("*.hd5", "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/lightsabre/hits/")
files[1]
outdir = "/home/wecapstor3/capn/capn100h/share_for_rasmus/"
#mkdir(outdir, exist_ok=true)
for fname in files
    f = h5open(fname)

    outfile_name = joinpath(outdir, basename(fname))

    h5open(outfile_name, "w") do fout
        for groupn in keys(f["pmt_hits"])
            grp = f["pmt_hits"][groupn]

            sumw = sum(grp[:, 3])
            weights = FrequencyWeights(grp[:, 3], sumw)
            grplen = size(grp, 1)

            outlen = min(grplen, sumw)
            sampled = sample(1:grplen, weights, ceil(Int64, outlen), replace=true)
            resampled_hits = grp[:, :][sampled, 1:2]
            out_vector = zeros(Float32, 26)
            PhotonSurrogateModel._convert_grp_attrs_to_features!(attrs(grp), out_vector)

            fout[groupn] = resampled_hits

            out_attrs = Dict(attrs(grp))
            out_attrs["transformed_features"] = out_vector
            out_attrs["total_hits"] = sumw

            f_attrs = HDF5.attrs(fout[groupn])

            for (k, v) in pairs(out_attrs)
                f_attrs[k] = v
            end
        end
    end

    close(f)


end