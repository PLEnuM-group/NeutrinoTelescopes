module ProposalInterface
using PyCall

export proposal_secondary_to_particle, propagate_muon

pp = pyimport("proposal")
pp.InterpolationSettings.tables_path = joinpath(@__DIR__,"../../assets/proposal_tables")

function proposal_secondary_to_particle(loss)
    energy = loss.energy / 1E3
    pos = SA[loss.position.x / 100, loss.position.y / 100, loss.position.z / 100] 
    dir = SA[loss.direction.x, loss.direction.y, loss.direction.z]
    time = loss.time * 1E9

    return Particle(pos, dir, time, energy, 0., PEMinus)
end


function propagate_muon(particle)

    position = particle.position
    direction = particle.direction
    length = particle.length
    time = particle.time

    if particle.type == PMuMinus
        particle = pp.particle.MuMinusDef()
    elseif particle.type == PMuPlus
        particle = pp.particle.MuPlusDef()
    else
        error("Type $(particle.type) not supported")
    end
    propagator = pp.Propagator(particle, joinpath(@__DIR__,"../../assets/proposal_config.json"))

    initial_state = pp.particle.ParticleState()
    initial_state.energy = energy*1E3
    initial_state.position = pp.Cartesian3D(position[1]*100, position[2]*100, position[3]*100)
    initial_state.direction = pp.Cartesian3D(direction[1], direction[2], direction[3])
    initial_state.time = time / 1E9
    final_state =  propagator.propagate(initial_state, max_distance=length*100)
    stochastic_losses = final_state.stochastic_losses()
    loss_to_particle.(stochastic_losses)
end
end