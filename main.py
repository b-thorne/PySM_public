import ConfigParser, os
import pysm_synchrotron,pysm_thermaldust,pysm_cmb,pysm_spinningdust, pysm_noise, pysm_freefree
from pysm import output, config2list, file_path, smooth_write
import healpy as hp
import numpy as np
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Code to simulate galactic foregrounds.')
    parser.add_argument('config_file', help='Main configuration file.')

##Get the output directory and save the configuration file.
    Config = ConfigParser.ConfigParser()
    Config.read(parser.parse_args().config_file)
    out = output(Config._sections['GlobalParameters'])

    if not os.path.exists(out.output_dir): os.makedirs(out.output_dir)
    with open(out.output_dir+out.output_prefix+'main_config.ini','w') as configfile: Config.write(configfile)

    if out.debug == True:

##Print information about the run:
        print '----------------------------------------------------- \n'
        print ''.join("%s: %s \n" % item   for item in vars(out).items())
        print '-----------------------------------------------------'
        
    sky = np.zeros(hp.nside2npix(out.nside))
    print '----------------------------------------------------- \n'
#Create synchrotron, dust, AME,  and cmb maps at output frequencies then add noise.
    if 'synchrotron' in out.components:
        sky = pysm_synchrotron.main(parser.parse_args().config_file)

    if 'thermaldust' in out.components:
        sky = sky + pysm_thermaldust.main(parser.parse_args().config_file)

    if 'spinningdust' in out.components:
        sky = sky + pysm_spinningdust.main(parser.parse_args().config_file)

    if 'freefree' in out.components:
        sky = sky + pysm_freefree.main(parser.parse_args().config_file)

    if 'cmb' in out.components:
        sky = sky + pysm_cmb.main(parser.parse_args().config_file)

    if out.instrument_noise == True:
        sky = sky + pysm_noise.instrument_noise(parser.parse_args().config_file)

    comps =str()

    if out.instrument_noise: 
        out.components.append('noise')

    sky = np.swapaxes(sky,0,1)

    if out.smoothing:
        print 'Smoothing output maps.'
        print '----------------------------------------------------- \n'

    for i in xrange(len(out.output_frequency)): smooth_write(sky[i,...],out,Config,i)
    
    print '-----------------------------------------------------\n'
    print 'PySM completed successfully. \n' 
    print '-----------------------------------------------------'

