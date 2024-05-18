

triggered = [length(e[:module_triggers]) > 0 for e in event_collection]
triggered_events = event_collection[triggered]

event_energies = [first(e[:particles]).energy for e in triggered_events]
event_weights = [first(e[:one_weight]) for e in triggered_events]
bins = 2:0.3:7

eff_area = fit(Histogram, log10.(event_energies), Weights(event_weights), bins)

eff_area_y = eff_area.weights ./ (diff(10 .^ bins) * 1E4 * (4*pi) * length(event_collection)) .* length(injector)

stairs(bins, [eff_area_y; eff_area_y[end]])