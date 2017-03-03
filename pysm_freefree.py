import numpy as np
import healpy as hp
from pysm import scale_freqs, convert_units, output, component, add_frequency_decorrelation
import ConfigParser

def scale_ff_pop(i_pop,npop,out,Config) :

	if npop==1 : secname='FreeFree'
	else : secname='FreeFree_%d'%i_pop
        freefree = component(Config._sections[secname],out.nside)
        print('Computing free-free maps (%d-th component).'%i_pop)
        print '----------------------------------------------------- \n'
        if out.debug == True:
                print ''.join("%s: %s \n" % item   for item in vars(freefree).items())
                print '----------------------------------------------------- \n'
        
	maps_constrained=np.array([freefree.em_template*convert_units(freefree.template_units,['u','K_RJ'],freefree.freq_ref),
				   np.zeros(len(freefree.em_template)),
				   np.zeros(len(freefree.em_template))])

	scaled_map_freefree=add_frequency_decorrelation(out,freefree,maps_constrained,pol=False)

        if out.debug == True:
	    for i in range(0,len(out.output_frequency)):
		    hp.write_map(out.output_dir+out.output_prefix+'ff%d'%i_pop+'_%d'%(out.output_frequency[i])+'_'+str(out.nside)+'.fits',
				 scaled_map_freefree[i],coord='G',column_units=out.output_units)

        return np.transpose(scaled_map_freefree,axes=[1,0,2])

def main(fname_config):

#Read configuration into classes
        Config = ConfigParser.ConfigParser()
	Config_model = ConfigParser.ConfigParser()

        Config.read(fname_config)
        out = output(Config._sections['GlobalParameters'])

        a=Config_model.read('./ConfigFiles/'+Config.get('FreeFree','model')+'_config.ini')
	if a==[] :
		print 'Couldn\'t find file '+'./ConfigFiles/'+Config.get('FreeFree','model')+'_config.ini'
		exit(1)

	npops = len(Config_model.sections())

        with open(out.output_dir+out.output_prefix+'freefree_config.ini','w') as configfile: Config_model.write(configfile)

	ff_out = 0.
	
	for i_pop in np.arange(npops)+1 :
		ff_out += scale_ff_pop(i_pop,npops,out,Config_model)

	return ff_out
