import numpy as np
import healpy as hp
from pysm import *
import ConfigParser

def main(fname_config):

#Read config file.
	Config = ConfigParser.ConfigParser()
	Config.read(fname_config)
	out = output(Config._sections['GlobalParameters'])
	Config.read('./ConfigFiles/'+Config.get('CMB','model')+'_config.ini')
	CMB = component(Config._sections['CMB'],out.nside)
	with open(out.output_dir+out.output_prefix+'cmb_config.ini','w') as configfile: Config.write(configfile)

	print 'Computing CMB maps.'
	print '----------------------------------------------------- \n'
	if out.debug:
		print ''.join("%s: %s \n" % item   for item in vars(CMB).items())
		print '----------------------------------------------------- \n'
#This code is edited from taylens code: Naess, S. K. and Louis, T. 2013 'Lensing simulations by Taylor expansion - not so inefficient after all'  Journal of Cosmology and Astroparticle Physics September 2013
#Available at: https://github.com/amaurea/taylens
	
	if CMB.compute_lensed_cmb:
		
		print('Using taylens to compute temperature map.')
		print '----------------------------------------------------- \n'
		synlmax = 8*out.nside #this used to be user-defined.
		data = np.transpose(np.loadtxt(CMB.specs))
		lmax_cl = len(data[0])+1
		l = np.arange(int(lmax_cl+1))
		synlmax = min(synlmax, l[-1])

#Reading input spectra in CAMB format.  CAMB outputs l(l+1)/2pi hence the corrections.

		cl_tebp_arr=np.zeros([10,lmax_cl+1])
		cl_tebp_arr[0,2:]=2*np.pi*data[1]/(l[2:]*(l[2:]+1))      #TT
		cl_tebp_arr[1,2:]=2*np.pi*data[2]/(l[2:]*(l[2:]+1))      #EE
		cl_tebp_arr[2,2:]=2*np.pi*data[3]/(l[2:]*(l[2:]+1))      #BB
		cl_tebp_arr[4,2:]=2*np.pi*data[4]/(l[2:]*(l[2:]+1))      #TE
                cl_tebp_arr[5,:] =np.zeros(lmax_cl+1)                    #EB
		cl_tebp_arr[7,:] =np.zeros(lmax_cl+1)                    #TB

		if CMB.delens:

			cl_tebp_arr[3,2:]=2*np.pi*data[5]*CMB.delensing_ells[1]/(l[2:]*(l[2:]+1))**2   #PP
			cl_tebp_arr[6,:] =np.zeros(lmax_cl+1)                    #BP
			cl_tebp_arr[8,2:]=2*np.pi*data[7]*np.sqrt(CMB.delensing_ells[1])/(l[2:]*(l[2:]+1))**1.5 #EP
			cl_tebp_arr[9,2:]=2*np.pi*data[6]*np.sqrt(CMB.delensing_ells[1])/(l[2:]*(l[2:]+1))**1.5 #TP
		else: 
			cl_tebp_arr[3,2:]=2*np.pi*data[5]/(l[2:]*(l[2:]+1))**2   #PP
			cl_tebp_arr[6,:] =np.zeros(lmax_cl+1)                    #BP
			cl_tebp_arr[8,2:]=2*np.pi*data[7]/(l[2:]*(l[2:]+1))**1.5 #EP
			cl_tebp_arr[9,2:]=2*np.pi*data[6]/(l[2:]*(l[2:]+1))**1.5 #TP

# Coordinates of healpix pixel centers
		ipos = np.array(hp.pix2ang(out.nside, np.arange(12*(out.nside**2))))

# Simulate a CMB and lensing field
		cmb, aphi = simulate_tebp_correlated(cl_tebp_arr,out.nside,synlmax,CMB.cmb_seed)
		
		if cmb.ndim == 1: cmb = np.reshape(cmb, [1,cmb.size])

# Compute the offset positions
		phi, phi_dtheta, phi_dphi = hp.alm2map_der1(aphi,out.nside,lmax=synlmax)

		del aphi
		
		opos, rot = offset_pos(ipos, phi_dtheta, phi_dphi, pol=True, geodesic=False) #geodesic used to be used defined.
		del phi, phi_dtheta, phi_dphi

# Interpolate maps one at a time
		maps  = []
		for comp in cmb:
			for m in taylor_interpol_iter(comp, opos, 3, verbose=False, lmax=None): #lmax here needs to be fixed. order of taylor expansion is fixed to 3.
				pass
			maps.append(m)
		del opos, cmb
#save the map computed for future referemce.
		rm = apply_rotation(maps, rot)
	else:
#option to use an already-computed lensed cmb map.	

		print('Scaling a lensed cmb temperature map.')
		print '----------------------------------------------------- \n'
		rm = np.array([i for i in CMB.lensed_cmb])
		
	if out.debug == True:
		map_cmb = rm[:,np.newaxis,:]*scale_freqs(CMB,out)
		for i in range(len(out.output_frequency)):
			hp.write_map(out.output_dir+out.output_prefix+'lensed_cmb_%d'%(out.output_frequency[i])+'_'+str(out.nside)+'.fits',map_cmb[:,i,:],coord='G',column_units=out.output_units)

	return rm[:,np.newaxis,:]*scale_freqs(CMB,out)


