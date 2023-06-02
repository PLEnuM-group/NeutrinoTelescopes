module IO

using HDF5
using Formatting
export save_hdf!

"""
save_hdf!(
    fname::AbstractString,
    group::AbstractString,
    dataset::Matrix,
    attributes::Dict)

Save a matrix dataset to an HDF5 file with the specified filename, group, and attributes.

Datasets are appended to the group with incrementing dataset identifier, the first being `dataset_1`

### Arguments
- `fname::AbstractString`: The filename of the HDF5 file to save the dataset.
- `group::AbstractString`: The group within the HDF5 file to store the dataset.
- `dataset::Matrix`: The matrix dataset to be saved.
- `attributes::Dict`: A dictionary of attributes to be associated with the dataset.

"""
function save_hdf!(
    fname::AbstractString,
    group::AbstractString,
    dataset::Matrix,
    attributes::Dict)


    if isfile(fname)
        fid = h5open(fname, "r+")
    else
        fid = h5open(fname, "w")
    end

    if !haskey(fid, group)
        g = create_group(fid, group)
        HDF5.attrs(g)["nsims"] = 0
    else
        g = fid[group]
    end


    offset = HDF5.read_attribute(g, "nsims") + 1
    ds_name = format("dataset_{:d}", offset)

    g[ds_name] = dataset# Matrix{Float64}(res.hits[:, [:tres, :pmt_id]])
    f_attrs = HDF5.attrs(g[ds_name])
    for (k, v) in attributes
        f_attrs[k] = v
    end

    HDF5.attrs(g)["nsims"] = offset

    close(fid)
end








end
