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
        np.testing.assert_almost_equal(pysm.pysm.bandpass(self.frequencies, self.weights, self.x) / self.expected_result_1, 1., decimal = 5)

    def test_bandpass_2(self):
        np.testing.assert_almost_equal(pysm.pysm.bandpass(self.frequencies, self.weights, self.sin) / self.expected_result_2, 1., decimal = 5)

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
        self.nside = 1024
        sigma_T = 4.
        sigma_P = np.sqrt(2.) * sigma_T
        self.instrument_config = {
                'frequencies' : np.array([23.]),
                'sens_I' : np.array([sigma_T]),
                'sens_P' : np.array([sigma_P]),
                'nside' : self.nside,
                'noise_seed' : 1234,
                'use_bandpass' : False,
                'add_noise' : True,
                'output_units' : 'uK_CMB',
                'use_smoothing' : False,
                'output_directory' : os.path.dirname(os.path.abspath(__file__)),
                'output_prefix' : 'test',
            }
        s1 = models("s1", self.nside)
        s1[0]['A_I'] = np.zeros(hp.nside2npix(self.nside))
        s1[0]['A_Q'] = np.zeros(hp.nside2npix(self.nside))
        s1[0]['A_U'] = np.zeros(hp.nside2npix(self.nside))
        sky_config = {'synchrotron' : s1}
        self.sky = pysm.Sky(sky_config)
        
        pix2amin = np.sqrt(4. * np.pi * (180. / np.pi * 60.) ** 2 / float(hp.nside2npix(self.nside)))

        self.expected_T_std = sigma_T / pix2amin
        self.expected_P_std = sigma_P / pix2amin

    def tearDown(self):
        os.remove(self.test_file)
        
    def test_noise(self):
        instrument = pysm.Instrument(self.instrument_config)
        instrument.observe(self.sky)
        self.test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_nu0023p00GHz_noise_nside%04d.fits"%self.nside)
        T, Q, U = pysm.read_map(self.test_file, self.nside, field = (0, 1, 2))
        T_std = np.std(T)
        Q_std = np.std(Q)
        U_std = np.std(U)

        np.testing.assert_almost_equal(T_std, self.expected_T_std, decimal = 2)
        np.testing.assert_almost_equal(Q_std, self.expected_P_std, decimal = 2)
        np.testing.assert_almost_equal(U_std, self.expected_P_std, decimal = 2)
        
        
def main():
    unittest.main()

if __name__ == "__main__":
    main()


