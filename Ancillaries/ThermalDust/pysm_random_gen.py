import healpy as hp
import numpy as np
from scipy.optimize import minimize 
from scipy.stats import norm

def chi_sq(s,mean,npix,sigma,theta):
    rand_map = norm.rvs(size=npix)*s+mean
    cmap = hp.smoothing(rand_map,fwhm=(np.pi/180.)*theta,verbose=False)
    chi = sigma**2-np.var(cmap)
    return chi**2

"""OPTIONS:

ouput_dir : Output directory
theta : scale of smoothing of final map (FWHM in degrees)
mean : mean of final map
sigma : rms of final map
nside : nside of final map

"""

output_dir = './model2/'
theta = 1.
mean = 1.59
sigma = 0.2
nside = 512

"""CODE"""

npix = hp.nside2npix(nside)
res = minimize(chi_sq,0.2,args=(mean,npix,sigma,theta),method='Powell')
cmap = hp.smoothing(mean+res.x*norm.rvs(size=npix),fwhm=(np.pi/180.)*theta,verbose=False)

suff = str(mean).replace('.','p')+'_'+str(sigma).replace('.','p')
hp.write_map(output_dir+'pysm_beta_rms_'+suff+'.fits',cmap)




