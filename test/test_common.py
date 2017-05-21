import unittest
import numpy as np
from astropy.io import fits
import scipy.constants as constants
from pysm import common
import os
import matplotlib.pyplot as plt

class CommonTests(unittest.TestCase):

    def testSafeInvert(self):
        np.random.seed(1234)
        l = np.random.randn(3) ** 2
        A = np.random.randn(3, 3)
        # construct positive definite matrix M
        M = np.dot(np.dot(A, np.diag(l)), np.transpose(A))
        # invert M with invert_safe
        M_inv = common.invert_safe(M)
        ident = np.dot(M_inv, M)
        self.assertAlmostEqual(np.sum(ident), 3.)
        
    def testConvert_Units(self):
        a1 = common.convert_units("K_CMB", "uK_RJ", 300.)
        a2 = common.convert_units("uK_RJ", "K_CMB", 300.)
        self.assertAlmostEqual(1., a1 * a2)
        a1 = common.convert_units("K_CMB", "MJysr", 300.)
        a2 = common.convert_units("MJysr", "K_CMB", 300.)
        self.assertAlmostEqual(1., a1 * a2)
        
        """Validation against ECRSC tables.
        https://irsasupport.ipac.caltech.edu/index.php?/Knowledgebase/
        Article/View/181/20/what-are-the-intensity-units-of-the-planck
        -all-sky-maps-and-how-do-i-convert-between-them
        These tables are based on the following tables: 
        h = 6.626176e-26 erg*s
        k = 1.380662e-16 erg/L
        c = 2.997792458e1- cm/s
        T_CMB = 2.726
        The impact of the incorrect CMB temperature is especially impactful
        and limits some comparison to only ~2/3 s.f.
        """

        uK_CMB_2_K_RJ_30 = 9.77074e-7
        uK_CMB_2_K_RJ_143 = 6.04833e-7
        uK_CMB_2_K_RJ_857 = 6.37740e-11

        self.assertAlmostEqual(uK_CMB_2_K_RJ_30, common.convert_units("uK_CMB", "K_RJ", 30.))
        self.assertAlmostEqual(uK_CMB_2_K_RJ_143, common.convert_units("uK_CMB", "K_RJ", 143.))
        self.assertAlmostEqual(uK_CMB_2_K_RJ_857, common.convert_units("uK_CMB", "K_RJ", 857.))

        K_CMB_2_MJysr_30 = 27.6515 
        K_CMB_2_MJysr_143 = 628.272 
        K_CMB_2_MJysr_857 = 22565.1

        self.assertAlmostEqual(K_CMB_2_MJysr_30 / common.convert_units("K_RJ", "MJysr", 30.), 1., places = 4)
        self.assertAlmostEqual(K_CMB_2_MJysr_143 / common.convert_units("K_RJ", "MJysr", 143.), 1., places = 4)
        self.assertAlmostEqual(K_CMB_2_MJysr_857 / common.convert_units("K_RJ", "MJysr", 857.), 1., places = 4)
        
        uK_CMB_2_MJysr_30 = 2.7e-5
        uK_CMB_2_MJysr_143 = 0.0003800
        uK_CMB_2_MJysr_857 = 1.43907e-6

        #Note that the MJysr definition seems to match comparatively poorly. The
        #definitions of h, k, c in the document linked above are in cgs and differ
        #from those on wikipedia. This may conflict with the scipy constants I use.

        self.assertAlmostEqual(uK_CMB_2_MJysr_30 / common.convert_units("uK_CMB", "MJysr", 30.), 1., places = 2)
        self.assertAlmostEqual(uK_CMB_2_MJysr_143 / common.convert_units("uK_CMB", "MJysr", 143.), 1., places = 2)
        self.assertAlmostEqual(uK_CMB_2_MJysr_857 / common.convert_units("uK_CMB", "MJysr", 857.), 1., places = 2)

class test_Bandpass_Unit_Conversion(unittest.TestCase):
    def setUp(self):
        """To test the bandpass unit conversion we use the Planck detector
        averaged bandpasses provided here:
        https://wiki.cosmos.esa.int/planckpla/index.php/The_RIMO. We
        compute the unit conversion factors for these bandpasses and
        compare them to the official Planck factors provided here:
        https://wiki.cosmos.esa.int/planckpla/index.php/UC_CC_Tables

        """
        # Read in the fits file. This contains only the HFI frequencies 100 -> 857.
        planck_HFI_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data" , "HFI_RIMO_R1.10.fits")
        hdu = fits.open(planck_HFI_file)
        bandpasses = map(lambda i: hdu[i].data, range(2, 8))
        # The table contains 4 lists: wavenumber, transmission, 1-sigma uncertainty, flag.
        # We are only interested in wavenumber and transmission.
        channels = []
        for b in bandpasses:
            wavenumber, transmission, _, _ = zip(*b)
            frequency = 1e-7 * constants.c * np.array(wavenumber)
            # exclude the element frqeuency[0] = 0
            filt = lambda x: (x[0] > 1.) & (x[0] < 1200)
            freqs, weights = zip(*filter(filt, zip(frequency, transmission)))
            channels.append((np.array(freqs), np.array(weights)))
            
        """Planck-provided coefficients for K_CMB to MJysr.
        These should only be taken to the first decimal place.

        """
        self.CMB2MJysr_avg_100_planck = 244.0960
        self.CMB2MJysr_avg_143_planck = 371.7327
        self.CMB2MJysr_avg_217_planck = 483.6874
        self.CMB2MJysr_avg_353_planck = 287.4517
        self.CMB2MJysr_avg_545_planck = 58.0356
        self.CMB2MJysr_avg_857_planck = 2.2681

        """And for MJysr to K_RJ"""
        self.MJysr2KRJ_avg_100_planck = 0.0032548074
        self.MJysr2KRJ_avg_143_planck = 0.0015916707
        self.MJysr2KRJ_avg_217_planck = 0.00069120334
        self.MJysr2KRJ_avg_353_planck = 0.00026120163
        self.MJysr2KRJ_avg_545_planck = 0.00010958025
        self.MJysr2KRJ_avg_857_planck = 4.4316316e-5

        """Do pysm calculation"""
        self.CMB2MJysr_avg_100_pysm = common.bandpass_convert_units("K_CMB", "MJysr", channels[0], nu_c = 100.)
        self.CMB2MJysr_avg_143_pysm = common.bandpass_convert_units("K_CMB", "MJysr", channels[1], nu_c = 143.)
        self.CMB2MJysr_avg_217_pysm = common.bandpass_convert_units("K_CMB", "MJysr", channels[2], nu_c = 217.)
        self.CMB2MJysr_avg_353_pysm = common.bandpass_convert_units("K_CMB", "MJysr", channels[3], nu_c = 353.)
        self.CMB2MJysr_avg_545_pysm = common.bandpass_convert_units("K_CMB", "MJysr", channels[4], nu_c = 545.)
        #self.CMB2MJysr_avg_857_pysm = common.bandpass_convert_units("K_CMB", "MJysr", channels[5], nu_c = 857.)

        self.RJ2MJysr_avg_100_pysm = common.bandpass_convert_units("K_RJ", "MJysr", channels[0], nu_c = 100.)
        self.RJ2MJysr_avg_143_pysm = common.bandpass_convert_units("K_RJ", "MJysr", channels[1], nu_c = 143.)
        self.RJ2MJysr_avg_217_pysm = common.bandpass_convert_units("K_RJ", "MJysr", channels[2], nu_c = 217.)
        self.RJ2MJysr_avg_353_pysm = common.bandpass_convert_units("K_RJ", "MJysr", channels[3], nu_c = 353.)
        self.RJ2MJysr_avg_545_pysm = common.bandpass_convert_units("K_RJ", "MJysr", channels[4], nu_c = 545.)
        #self.RJ2MJysr_avg_857_pysm = common.bandpass_convert_units("K_RJ", "MJysr", channels[5], nu_c = 857.)
        
    def tearDown(self):
        return
    
    # def test_bandpass_unit_conversion_CMB2MJysr(self):
        """Note that the precision is limited by uncertainty on the bandpass central frequency.
        """
        """
        np.testing.assert_almost_equal(self.CMB2MJysr_avg_100_pysm / self.CMB2MJysr_avg_100_planck, 1., decimal = 3)
        np.testing.assert_almost_equal(self.CMB2MJysr_avg_143_pysm / self.CMB2MJysr_avg_143_planck, 1., decimal = 3)
        np.testing.assert_almost_equal(self.CMB2MJysr_avg_217_pysm / self.CMB2MJysr_avg_217_planck, 1., decimal = 3)
        np.testing.assert_almost_equal(self.CMB2MJysr_avg_353_pysm / self.CMB2MJysr_avg_353_planck, 1., decimal = 3)
        np.testing.assert_almost_equal(self.CMB2MJysr_avg_545_pysm / self.CMB2MJysr_avg_545_planck, 1., decimal = 3)
        #np.testing.assert_almost_equal(self.CMB2MJysr_avg_857_pysm / self.CMB2MJysr_avg_857_planck, 1., decimal = 3)

        np.testing.assert_almost_equal(self.RJ2MJysr_avg_100_pysm * self.MJysr2KRJ_avg_100_planck, 1., decimal = 1)
        np.testing.assert_almost_equal(self.RJ2MJysr_avg_143_pysm * self.MJysr2KRJ_avg_143_planck, 1., decimal = 1)
        np.testing.assert_almost_equal(self.RJ2MJysr_avg_217_pysm * self.MJysr2KRJ_avg_217_planck, 1., decimal = 1)
        np.testing.assert_almost_equal(self.RJ2MJysr_avg_353_pysm * self.MJysr2KRJ_avg_353_planck, 1., decimal = 1)
        np.testing.assert_almost_equal(self.RJ2MJysr_avg_545_pysm * self.MJysr2KRJ_avg_545_planck, 1., decimal = 1)
        #np.testing.assert_almost_equal(self.RJ2MJysr_avg_857_pysm * self.MJysr2RJ_avg_857_planck, 1., decimal = 3)
      
        return
        """
    

class test_Check_Lengths(unittest.TestCase):
    def setUp(self):
        self.list1 = [1] * 20
        self.list2 = [2] * 20
        self.list3 = np.ones(20)
        self.list4 = np.ones(40)
        self.list5 = [3] * 40
        self.list6 = [(1, 2), (3, 4), (5, 6), (6, 7), (8, 9)]
        self.list7 = np.random.randn(5) 
        return

    def tearDown(self):
        self.list1 = None
        self.list2 = None
        self.list3 = None
        self.list4 = None
        self.list5 = None
        self.list6 = None
        self.list7 = None
        return

    def test_check_lengths(self):
        self.assertTrue(common.check_lengths(self.list1, self.list2, self.list3))
        self.assertTrue(common.check_lengths(self.list6, self.list7))
        self.assertFalse(common.check_lengths(self.list1, self.list3, self.list4))
        self.assertFalse(common.check_lengths(self.list1, self.list5))
        return
    
def main():
    unittest.main()

if __name__ == '__main__':
    main()
