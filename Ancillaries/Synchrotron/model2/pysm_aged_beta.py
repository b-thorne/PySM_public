import healpy as hp
import numpy as np

nside = 256

#Give theta measured from north pole.
theta, phi = hp.pix2ang(nside,np.arange(hp.nside2npix(nside)))

#Changes theta to spherical coordinates.
theta_spherical = 90. - (180./np.pi)*theta

#Gives beta_s smoothly varying from plane to poles.
beta_s = 0.3*np.cos(theta_spherical*(np.pi/180.))-3.3

hp.write_map('model2_synch_beta.fits',beta_s)
