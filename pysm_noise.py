import healpy as hp
import numpy as np
import ConfigParser
from pysm import output,convert_units

def instrument_noise(fname_config):

    Config = ConfigParser.ConfigParser()
    Config.read(fname_config)
    out = output(Config._sections['GlobalParameters'])

    print('Adding instrument noise.')
    print '----------------------------------------------------- \n'
 
    npix = hp.nside2npix(out.nside)

    #Convert noise to sigma per pixel.
    fsky_pix = 1./npix
    pix_ster = 4.*np.pi*fsky_pix
    pix_amin2 = pix_ster*(180.*60./np.pi)**2  #converts size of pixel from steradians to square arcminutes

    sigma_pix_I = np.sqrt(out.instrument_noise_i**2/pix_amin2)
    sigma_pix_pol = np.sqrt(out.instrument_noise_pol**2/pix_amin2)

    #Generate noise as gaussian with variances above:
    np.random.seed(out.instrument_noise_seed)
    instrument_noise = np.random.randn(3,np.asarray(out.output_frequency).size,npix)

    #standard_normal*sigma+mu = N(mu,sigma)
    instrument_noise[0,...]=sigma_pix_I[np.newaxis,:,np.newaxis]*instrument_noise[0,...]
    instrument_noise[1,...]=sigma_pix_pol[np.newaxis,:,np.newaxis]*instrument_noise[1,...]
    instrument_noise[2,...]=sigma_pix_pol[np.newaxis,:,np.newaxis]*instrument_noise[2,...]

    return instrument_noise*convert_units(['u','K_CMB'],out.output_units,out.output_frequency)[np.newaxis,:,np.newaxis]


