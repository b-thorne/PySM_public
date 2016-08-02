import ConfigParser
import healpy as hp
import numpy as np
from pysm import scale_freqs, output, component, convert_units
import matplotlib.pyplot as plt

def main(fname_config):

#Read in configuration file to classes.
    Config = ConfigParser.ConfigParser()
    Config.read(fname_config)
    out = output(Config._sections['GlobalParameters'])

    Config.read('./ConfigFiles/'+Config.get('SpinningDust','model')+'_config.ini')
    spdust_general = component(Config._sections['General'],out.nside)
    spdust1 = component(Config._sections['SpinningDust1'],out.nside)
    spdust2 = component(Config._sections['SpinningDust2'],out.nside)

    print('Computing spinning dust map.')
    print '----------------------------------------------------- \n'

    if out.debug == True:
        print ''.join("%s: %s \n" % item   for item in vars(spdust1).items())
        print ''.join("%s: %s \n" % item   for item in vars(spdust2).items())
        print '----------------------------------------------------- \n'
    with open(out.output_dir+out.output_prefix+'spdust_config.ini','w') as configfile: Config.write(configfile)
#Compute a map of the polarisation angle from the commander dust map polariationn angle. 
    
    pol_angle = np.arctan2(spdust_general.thermaldust_polu,spdust_general.thermaldust_polq)

#Units to do the scaling in MJysr and then bring the result back to the output units.
    conv1 = convert_units(spdust1.template_units, ['u','K_RJ'], spdust1.freq_ref)
    conv2 = convert_units(spdust2.template_units, ['u','K_RJ'], spdust2.freq_ref)
    conv_end = convert_units(['u','K_RJ'],out.output_units,out.output_frequency)
    unit_conversion1 = conv1*conv_end.reshape((len(out.output_frequency),1))
    unit_conversion2 = conv2*conv_end.reshape((len(out.output_frequency),1))

    scaled_map_spdust = scale_freqs(spdust1,out,pol=False)*spdust1.em_template*unit_conversion1 + scale_freqs(spdust2,out,pol=False)*spdust2.em_template*unit_conversion2
    scaled_map_spdust_pol = scaled_map_spdust[np.newaxis,...]*np.asarray([np.cos(pol_angle),np.sin(pol_angle)])[:,np.newaxis,:]*spdust_general.pol_frac

    if out.debug == True:
        for i in range(0,len(out.output_frequency)):
            hp.write_map(out.output_dir+'spdust_%d.fits'%(out.output_frequency[i]),scaled_map_spdust[i],coord='G',column_units=out.output_units)

    return np.concatenate([scaled_map_spdust[np.newaxis,...],scaled_map_spdust_pol])
