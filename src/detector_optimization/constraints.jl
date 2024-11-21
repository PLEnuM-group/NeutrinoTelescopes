export OptimisationConstraint, CenterOfMassConstraint, SimulationBoundaryConstraint


abstract type OptimisationConstraint end

struct CenterOfMassConstraint <: OptimisationConstraint
    penalty_scale::Float64
end

(f::CenterOfMassConstraint)(layout::OptimizationLayout{T}) where {T} = begin
    return f.penalty_scale *norm(get_center_of_mass(layout))
end


struct SimulationBoundaryConstraint <: OptimisationConstraint
    max_radius::Float64
    penalty_scale::Float64
end

function (f::SimulationBoundaryConstraint)(layout::OptimizationLayout{T}) where {T} 
    penalty = 0.0
    for pos in layout
        npos = sqrt(pos[1]^2 + pos[2]^2)
        if npos > f.max_radius
            penalty += npos - f.max_radius
        end
    end
    return penalty * f.penalty_scale
end

