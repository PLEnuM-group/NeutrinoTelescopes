module Triggering

using DataFrames

export lc_trigger, calc_coincs_from_trigger, count_coinc_in_tw

"""
    lc_trigger(sorted_hits::AbstractDataFrame, time_window)

Calculate local-coincidence triggers for `sorted_hits` in `time_window`.

The algorithm loops through all hits ``h_i``. When the next hit ``h_j`` is closer than the
time window ``\\delta`` a new trigger is started. The trigger will accumulate all hits
``h_j`` that are within ``h_i + \\delta``. Finally, a trigger is emitted when it includes
hits on at least two different PMTs.

Returns a Vector of hit-time vectors.
"""
function lc_trigger(sorted_hits::AbstractDataFrame, time_window)

    triggers = []
    i = 1
    while i < nrow(sorted_hits)

        lc_flag = false

        j = i + 1
        while j <= nrow(sorted_hits)
            if (sorted_hits[j, :time] - sorted_hits[i, :time]) <= time_window
                lc_flag = true
            else
                break
            end
            j += 1
        end

        if !lc_flag
            i = j
            continue
        end

        if length(unique(sorted_hits[i:(j-1), :pmt_id])) >= 2
            push!(triggers, sorted_hits[i:(j-1), :])
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
