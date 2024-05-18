module Triggering

using DataFrames

export LCTrigger, ModuleCoincTrigger
export lc_trigger, calc_coincs_from_trigger, count_coinc_in_tw, module_trigger


struct LCTrigger
    t_start::Float64
    time_window::Float64
    pmt_ids::Set{Int64}
    module_id::Int64
end


abstract type DetectorTrigger end

struct ModuleCoincTrigger <: DetectorTrigger
    t_start::Float64
    time_window::Float64
    module_ids::Set{Int64}
end


"""
    lc_trigger(sorted_hits::AbstractDataFrame; time_window=10.)

Calculate local-coincidence triggers for `sorted_hits` in `time_window`.

The algorithm loops through all hits ``h_i``. When the next hit ``h_j`` is closer than the
time window ``\\delta`` a new trigger is started. The trigger will accumulate all hits
``h_j`` that are within ``h_i + \\delta``. Finally, a trigger is emitted when it includes
hits on at least two different PMTs.

## Arguments
- `sorted_hits`: An abstract dataframe of sorted hits.
- `time_window`: The time window (in nanoseconds) within which coincidences are considered. Default value is 10.

Returns a Vector of hit-time vectors.
"""
function lc_trigger(sorted_hits::AbstractDataFrame; time_window=10.)

    triggers = Vector{LCTrigger}()
    i = 1
    while i < nrow(sorted_hits)

        lc_flag = false

        j = i + 1
        unique_pmts = Set{Int64}([sorted_hits[i, :pmt_id]])
        while j <= nrow(sorted_hits)
            if (sorted_hits[j, :time] - sorted_hits[i, :time]) <= time_window
                lc_flag = true
                push!(unique_pmts, sorted_hits[j, :pmt_id])
            else
                break
            end
            j += 1
        end

        if !lc_flag
            i = j
            continue
        end

        if length(unique_pmts) >= 2
            t_start = sorted_hits[i, :time]
            trigger = LCTrigger(t_start, time_window, unique_pmts, sorted_hits[i, :module_id])
            push!(triggers, trigger)
        end

        #=
        if length(unique(sorted_hits[i:(j-1), :pmt_id])) >= 2
            push!(triggers, sorted_hits[i:(j-1), :])
        end
        =#

        i = j
    end
    return triggers
end


"""
    module_trigger(lc_triggers::AbstractVector{<:LCTrigger}; time_window=1E3, lc_level=2)

This function takes a vector of `LCTrigger` objects and returns a vector of `ModuleCoincTrigger` objects. 
It performs triggering by identifying time coincidences between `LCTrigger` objects based on their start times and module IDs.

## Arguments
- `lc_triggers`: An abstract vector of `LCTrigger` objects.
- `time_window`: The time window (in nanoseconds) within which coincidences are considered. Default value is 1E3.
- `lc_level`: The minimum number of unique PMTs required for an `LCTrigger` object to be considered. Default value is 2.

## Returns
- `triggers`: A vector of `ModuleCoincTrigger` objects representing the identified coincidences.

"""
function module_trigger(lc_triggers::AbstractVector{<:LCTrigger}; time_window=1E3, lc_level=2)
    triggers = Vector{ModuleCoincTrigger}()
    i = 1
    
    lc_triggers = sort(filter(x->length(x.pmt_ids) >=lc_level, lc_triggers), by=x->x.t_start)

    while i < length(lc_triggers)

        coinc_flag = false
        current_lc = lc_triggers[i]
        j = i + 1

        unique_modules = Set([current_lc.module_id])
        while j <= length(lc_triggers)

            compare_lc = lc_triggers[j]

            if current_lc.t_start + time_window >= compare_lc.t_start
                coinc_flag = true
                push!(unique_modules, compare_lc.module_id)
            else
                break
            end
            j += 1
        end

        if !coinc_flag
            i = j
            continue
        end

        if length(unique_modules) >= 2
            t_start = lc_triggers[i].t_start
            trigger = ModuleCoincTrigger(t_start, time_window, unique_modules)
            push!(triggers, trigger)
        end
        i = j
    end
    return triggers
end


function calc_coincs_from_trigger(sorted_hits, timewindow)

    triggers = lc_trigger(sorted_hits, timewindow)
    coincs = Vector{Int64}()
    for trigger in triggers
        push!(coincs, length(unique(trigger[:, :pmt_id])))
    end
    return coincs
end


function count_coinc_in_tw(sorted_hits, time_window)

    t_start = sorted_hits[1, :time]

    window_cnt = Dict{Int64,Set{Int64}}()

    for i in 1:nrow(sorted_hits)
        window_id = div((sorted_hits[i, :time] - t_start), time_window)
        if !haskey(window_cnt, window_id)
            window_cnt[window_id] = Set([])
        end

        push!(window_cnt[window_id], sorted_hits[i, :pmt_id])
    end

    return length.(values(window_cnt))
end





end
