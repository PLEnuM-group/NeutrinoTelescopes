import LeptonInjector as LI
from math import pi

from argparse import ArgumentParser
from itertools import product

def select_xs_tag(final1, final2):
    
    particles = [LI.Particle.ParticleType.MuMinus, LI.Particle.ParticleType.EMinus, LI.Particle.ParticleType.TauMinus]

    if final1 in particles:
        sign_tag = "nu"
    else:
        sign_tag = "nubar"

    if final2 == LI.Particle.ParticleType.Hadrons:
        int_type_tag = "CC"
    else:
        int_type_tag = "NC"
    return sign_tag+"_"+int_type_tag


def make_injector(final1, final2, n_events):

    xs_folder = "/home/saturn/capn/capn100h/cross-sections/csms_differential_v1.0"
    is_ranged =  final1 in [LI.Particle.ParticleType.MuMinus, LI.Particle.ParticleType.MuPlus]

    xs_tag = select_xs_tag(final1, final2)
    diff_xs     = xs_folder + f"/dsdxdy_{xs_tag}_iso.fits"
    total_xs    = xs_folder + f"/sigma_{xs_tag}_iso.fits"

    injector = LI.Injector( n_events , final1, final2, diff_xs, total_xs, is_ranged)

    return injector


def run(args):

    
    earthmodel_folder = "/home/hpc/capn/capn100h/repos/LeptonInjector/resources/earthparams/"

    # Now, we'll make a new injector for muon tracks
    n_events    = args.nevents
    if args.mode == "lightsabre":

        finals1 = [LI.Particle.ParticleType.MuMinus, LI.Particle.ParticleType.MuPlus]
        finals2 = [LI.Particle.ParticleType.Hadrons, LI.Particle.ParticleType.Hadrons]

        injectors = [make_injector(final1, final2, n_events) for (final1, final2) in zip(finals1, finals2)]
    
    elif args.mode == "extended":
        #all cascade like ints

        # All electron

        injectors = [
            make_injector(final1, final2, n_events) for (final1, final2) in product([
                LI.Particle.ParticleType.EPlus, LI.Particle.ParticleType.EMinus,
                LI.Particle.ParticleType.NuE, LI.Particle.ParticleType.NuEBar,
                LI.Particle.ParticleType.NuMu, LI.Particle.ParticleType.NuMuBar],
                [LI.Particle.ParticleType.Hadrons])
        ]

    # construct the controller
    controller  = LI.Controller(injectors.pop(0), args.emin, args.emax, args.gamma, 0, 2*pi, 0, pi, args.geo_radius, args.geo_length, args.geo_radius, args.geo_length)
    for inj in injectors:
        controller.AddInjector(inj)

    # specify the output, earth model

    controller.SetEarthModel("Planet", earthmodel_folder)
    controller.NameOutfile(args.outfile)
    controller.NameLicFile(args.outfile_lic)

    # run the simulation
    controller.Execute()

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--emin", type=float, dest="emin", default=1E2)
    parser.add_argument("--emax", type=float, dest="emax", default=1E7)
    parser.add_argument("--gamma", type=float, dest="gamma", default=2)
    parser.add_argument("--geo-radius", type=float, dest="geo_radius", default=1200)
    parser.add_argument("--geo-length", type=float, dest="geo_length", default=1200)
    parser.add_argument("--outfile", type=str, dest="outfile", required=True)
    parser.add_argument("--outfile_lic", type=str, dest="outfile_lic", required=True)
    parser.add_argument("--nevents", type=int, dest="nevents", required=True)
    parser.add_argument("--mode", choices=["lightsabre", "extended"], dest="mode", required=True)

    args = parser.parse_args()

    run(args)
    

