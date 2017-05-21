import unittest
import numpy as np
import healpy as hp
import scipy.constants as constants
import pysm
from pysm.nominal import models, template
import os
from subprocess import call

class BandpassTests(unittest.TestCase):
    def setUp(self):
        self.frequencies = np.linspace(0, 1, 100000)
        self.weights = np.ones_like(self.frequencies)
        self.x = lambda x: x
        self.expected_result_1 = 0.5 
        self.sin = lambda x: np.sin(x)
        self.expected_result_2 = (np.cos(0) - np.cos(1))

    def tearDown(self):
        self.uneven_frequencies = None
        self.weights = None
        self.integrated_map = None
        self.analytic_I = None
        self.analytic_Q = None
        self.analytic_U = None
        
    def test_bandpass_1(self):
        np.testing.assert_almost_equal(pysm.pysm.bandpass(self.frequencies, self.weights, self.x) / self.expected_result_1, 1., decimal = 6)

    def test_bandpass_2(self):
        np.testing.assert_almost_equal(pysm.pysm.bandpass(self.frequencies, self.weights, self.sin) / self.expected_result_2, 1., decimal = 6)

class testCheck_Bandpass_Frequencies(unittest.TestCase):
    def setUp(self):
        self.frequencies_uneven = np.logspace(2, 3, 50)
        self.frequencies_even = np.linspace(100, 130, 30)
        return

    def teatDown(self):
        self.freuqencies_uneven = None
        self.frequencies_even = None
        return

    def test_check_bandpass_frequencies(self):
        with self.assertRaises(SystemExit):
            pysm.pysm.check_bpass_frequencies(self.frequencies_uneven)
        
class TestNoise(unittest.TestCase):
    def setUp(self):
        nside = 1024
        sigma_T = 4.
        sigma_P = np.sqrt(2.) * sigma_T
        instrument_config = {
                'frequencies' : np.array([23.]),
                'sens_I' : np.array([sigma_T]),
                'sens_P' : np.array([sigma_P]),
                'nside' : nside,
                'noise_seed' : 1234,
                'use_bandpass' : False,
                'add_noise' : True,
                'output_units' : 'uK_CMB',
                'use_smoothing' : False,
                'output_directory' : os.path.dirname(os.path.abspath(__file__)),
                'output_prefix' : 'test',
            }
        s1 = models("s1", nside)
        s1[0]['A_I'] = np.zeros(hp.nside2npix(nside))
        s1[0]['A_Q'] = np.zeros(hp.nside2npix(nside))
        s1[0]['A_U'] = np.zeros(hp.nside2npix(nside))
        sky_config = {'synchrotron' : s1}
        sky = pysm.Sky(sky_config)
        instrument = pysm.Instrument(instrument_config)
        instrument.observe(sky)
        self.test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_nu0023p00GHz_noise_nside%04d.fits"%nside)
        T, Q, U = pysm.read_map(self.test_file, nside, field = (0, 1, 2))
        T_std = np.std(T)
        Q_std = np.std(Q)
        U_std = np.std(U)
        
        pix2amin = np.sqrt(4. * np.pi * (180. / np.pi * 60.) ** 2 / float(hp.nside2npix(nside)))
        
        self.check_T = T_std * pix2amin / sigma_T
        self.check_Q = Q_std * pix2amin / sigma_P
        self.check_U = U_std * pix2amin / sigma_P

    def tearDown(self):
        os.system("rm %s"%self.test_file)
        
    def test_noise(self):
        np.testing.assert_almost_equal(self.check_T, 1., decimal = 3)
        np.testing.assert_almost_equal(self.check_Q, 1., decimal = 3)
        np.testing.assert_almost_equal(self.check_U, 1., decimal = 3)
        
        
def main():
    unittest.main()

if __name__ == "__main__":
    main()


