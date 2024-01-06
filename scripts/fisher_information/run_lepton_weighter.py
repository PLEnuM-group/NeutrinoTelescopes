import LeptonWeighter as LW
import h5py
from math import pi
import numpy as np

from argparse import ArgumentParser

def get_weight(weighter, props):
    """
    Accepts the properties list of an event and returns the weight
    """
    LWevent = LW.Event()
    LWevent.energy = props[0]
    LWevent.zenith = props[1]
    LWevent.azimuth = props[2]
    
    LWevent.interaction_x = props[3]
    LWevent.interaction_y = props[4]
    LWevent.final_state_particle_0 = LW.ParticleType( props[5] )
    LWevent.final_state_particle_1 = LW.ParticleType( props[6] )
    LWevent.primary_type = LW.ParticleType( props[7] )
    # Radius is completely irrelevant
    LWevent.radius = 0
    
    LWevent.x = props[8]
    LWevent.y = props[9]
    LWevent.z = props[10]
    LWevent.total_column_depth = props[11]

    weight = weighter.get_oneweight(LWevent)

    # this would alert us that something bad is happening 
    if weight==np.nan:
        raise ValueError("Bad Weight!")

    return weight


def run(args):

    xs_folder = "/home/saturn/capn/capn100h/cross-sections/csms_differential_v1.0"
    earthmodel_folder = "/home/hpc/capn/capn100h/repos/LeptonInjector/resources/earthparams/"

    flux_params={ 'constant': 10**-18, 'index':-2, 'scale':10**5 }

    # Create generator
    #    if there were multiple LIC files, you would instead make a list of Generators
    net_generation = LW.MakeGeneratorsFromLICFile(args.lic_file)
  
    xs = LW.CrossSectionFromSpline(
                        xs_folder+"/dsdxdy_nu_CC_iso.fits",
                        xs_folder+"/dsdxdy_nubar_CC_iso.fits",
                        xs_folder+"/dsdxdy_nu_NC_iso.fits",
                        xs_folder+"/dsdxdy_nubar_NC_iso.fits")
    flux = LW.PowerLawFlux( flux_params['constant'] , flux_params['index'] , flux_params['scale'] )


    weighter = LW.Weighter(flux, xs, net_generation )

    # load data
    data_file = h5py.File(args.li_file, "r+")

    for injector in data_file.keys():
        weights = [get_weight(weighter, data_file[injector]['properties'][event]) for event in range(len( data_file[injector]['properties']))]
        data_file[injector+"/weights"] = weights
    data_file.close()

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--li-file", type=str, dest="li_file", required=True)
    parser.add_argument("--lic-file", type=str, dest="lic_file", required=True)

    args = parser.parse_args()

    run(args)
    

