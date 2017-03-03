import numpy as np
import healpy as hp
from pysm import scale_freqs, convert_units, component, output, add_frequency_decorrelation
import ConfigParser

def scale_spdust_pop(i_pop,npop,out,Config) :
    
    if npop==1 : secname='General'
    else : secname='General_%d'%i_pop
    spdust_general = component(Config._sections[secname],out.nside)
    if npop==1 : secname='SpinningDust1'
    else : secname='SpinningDust1_%d'%i_pop
    spdust1 = component(Config._sections[secname],out.nside)
    if npop==1 : secname='SpinningDust2'
    else : secname='SpinningDust2_%d'%i_pop
    spdust2 = component(Config._sections[secname],out.nside)

    print('Computing spinning dust map (%d-th component).'%i_pop)
    print '----------------------------------------------------- \n'

    if out.debug == True:
        print ''.join("%s: %s \n" % item   for item in vars(spdust1).items())
        print ''.join("%s: %s \n" % item   for item in vars(spdust2).items())
        print '----------------------------------------------------- \n'

    #Compute a map of the polarisation angle from the commander dust map polariationn angle. 
    pol_angle = np.arctan2(spdust_general.thermaldust_polu,spdust_general.thermaldust_polq)

    map_i_1=spdust1.em_template*convert_units(spdust1.template_units,['u','K_RJ'],spdust1.freq_ref)
    map_i_2=spdust2.em_template*convert_units(spdust2.template_units,['u','K_RJ'],spdust2.freq_ref)
    maps_constrained_1=np.array([map_i_1,np.zeros_like(map_i_1),np.zeros_like(map_i_1)])
    maps_constrained_2=np.array([map_i_2,np.zeros_like(map_i_2),np.zeros_like(map_i_2)])
    scaled_map_1=add_frequency_decorrelation(out,spdust1,maps_constrained_1,pol=False)
    scaled_map_2=add_frequency_decorrelation(out,spdust2,maps_constrained_2,pol=False)
    scaled_map_spdust=scaled_map_1+scaled_map_2
    scaled_map_spdust[:,1,:]=scaled_map_spdust[:,0,:]*(spdust_general.pol_frac*np.cos(pol_angle))[None,:]
    scaled_map_spdust[:,2,:]=scaled_map_spdust[:,0,:]*(spdust_general.pol_frac*np.sin(pol_angle))[None,:]

    if out.debug == True:
        for i in range(0,len(out.output_frequency)):
            hp.write_map(out.output_dir+out.output_prefix+'spdust%d'%i_pop+'_%d'%(out.output_frequency[i])+'_'+str(out.nside)+'.fits',
                         scaled_map_spdust[i],coord='G',column_units=out.output_units)

    return np.transpose(scaled_map_spdust,axes=[1,0,2])


def main(fname_config):

#Read in configuration file to classes.
    Config = ConfigParser.ConfigParser()
    Config_model = ConfigParser.ConfigParser()

    Config.read(fname_config)
    out = output(Config._sections['GlobalParameters'])

    a=Config_model.read('./ConfigFiles/'+Config.get('SpinningDust','model')+'_config.ini')
    if a==[] :
        print 'Couldn\'t find file '+'./ConfigFiles/'+Config.get('SpinningDust','model')+'_config.ini'
        exit(1)

    npops = len(Config_model.sections())/3

    with open(out.output_dir+out.output_prefix+'spdust_config.ini','w') as configfile: Config_model.write(configfile)

    spdust_out = 0.

    for i_pop in np.arange(npops)+1 :
        spdust_out += scale_spdust_pop(i_pop,npops,out,Config_model)

    return spdust_out
