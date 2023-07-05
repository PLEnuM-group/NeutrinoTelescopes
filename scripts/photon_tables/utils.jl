using StatsBase

function make_setup(
    mode, pos, dir, energy, seed;
    g=0.99f0)

    medium = make_cascadia_medium_properties(g)
    target = POM(SA_F32[0, 0, 0], UInt16(1))
    wl_range = (300.0f0, 800.0f0)

    spectrum = CherenkovSpectrum(wl_range, medium, 30)

    if mode == :extended
        particle = Particle(
            pos,
            dir,
            0.0f0,
            Float32(energy),
            0.0f0,
            PEMinus
        )
        source = ExtendedCherenkovEmitter(particle, medium, wl_range)
    elseif mode == :bare_infinite_track
        length = 400f0
        ppos = pos .- length/2 .* dir
        

        particle = Particle(
            ppos,
            dir,
            0.0f0,
            Float32(energy),
            length,
            PMuMinus
        )

        source = CherenkovTrackEmitter(particle, medium, wl_range)    
    elseif mode == :lightsabre_muon
        length = 400f0
        ppos = pos .- length/2 .* dir
        
        particle = Particle(
            Float32.(ppos),
            Float32.(dir),
            0.0f0,
            Float32(energy),
            length,
            PMuMinus
        )

        source = LightsabreMuonEmitter(particle, medium, wl_range)

    elseif mode == :pointlike_cherenkov
        particle = Particle(
            pos,
            dir,
            0.0f0,
            Float32(energy),
            0.0f0,
            PEMinus)
        source = PointlikeChernekovEmitter(particle, medium, wl_range)
    else
        error("unknown mode $mode")
    end

    setup = PhotonPropSetup(source, target, medium, spectrum, seed)
    return setup

end