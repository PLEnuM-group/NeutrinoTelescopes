using ArgParse
using JLD2
using DataFrames
using NeutrinoTelescopes
using StaticArrays
using PhysicsTools

function main(args)
    data = []
    re = r".*muon_eff_area_([0-9]+)_([0-9]+)_"
    jldopen(args[:infile]) do file
        events = file["events"]
        event_ids = keys(events)
        m = match(re, args[:infile])
        vert_spacing, hor_spacing = m.captures
        for event_ids in event_ids
            event = events[event_ids]
            if haskey(event, :dir_uncert)
                @show event[:dir_uncert]
                dir_uncert = event[:dir_uncert]
            else
                dir_uncert = NaN
            end
            push!(data,
                (e_entry=event[:e_entry],
                 generation_area=event[:generation_area],
                 spectrum_weight=event[:spectrum_weight],
                 area_weight=event[:area_weight],
                 triggered_ls_20=length(event[Symbol("module_triggers_lightsabre_hits20")]) > 1,
                 triggered_full_20=length(event[Symbol("module_triggers_hits20")]) > 1,
                 cos_zen=event[:muon_at_entry].direction[3],
                 vert_spacing=vert_spacing,
                 hor_spacing=hor_spacing,
                 dir_uncert=dir_uncert
                 ))
        end
    end
    data = DataFrame(data)

    println("Saving to $(args[:outfile])")
    jldsave(args[:outfile], data=data)
end

s = ArgParseSettings()
@add_arg_table s begin
    "--infile"
    help = "Input file"
    required = true
    "--outfile"
    help = "Output file"
    required = true
end
parsed_args = parse_args(ARGS, s; as_symbols=true)
main(parsed_args)