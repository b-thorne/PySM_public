import unittest, os
from pysm import components, common, read_map, convert_units
from astropy.analytic_functions import blackbody_nu
import numpy as np, healpy as hp
import matplotlib.pyplot as plt
import scipy.constants as constants
from pysm.nominal import models

class ComponentsTests(unittest.TestCase):
    
    def testPower_Law(self):
        scaling1 = components.power_law(120., 30., -0.5)
        scaling2 = components.power_law(1., 27., 1./3.)
        scaling3 = components.power_law(1., 1., 2.)
        self.assertAlmostEqual(scaling1, 0.5, places = 9)
        self.assertAlmostEqual(scaling2, 1. / 3., places = 9)
        self.assertAlmostEqual(scaling3, 1., places = 9)

    def testBlack_Body(self):
        astropy = blackbody_nu(90.e9, 100.) / blackbody_nu(30.e9, 100.)
        pysm = components.black_body(90., 30., 100.)
        self.assertAlmostEqual(astropy, pysm)

class test_Dust(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pysm', 'template'))
        test_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data', 'benchmark'))
    
        d1_config = models("d1", 64)
        dust = components.Dust(d1_config[0])
        signal = dust.signal()

        dust_1_30GHz = read_map(os.path.join(test_data_dir, 'check1therm_30p0_64.fits'), 64, field = (0, 1, 2))
        dust_1_100GHz = read_map(os.path.join(test_data_dir, 'check1therm_100p0_64.fits'), 64, field = (0, 1, 2))
        dust_1_353GHz = read_map(os.path.join(test_data_dir, 'check1therm_353p0_64.fits'), 64, field = (0, 1, 2))

        self.frac_diff_30GHz = (dust_1_30GHz - signal(30.)) / dust_1_30GHz
        self.frac_diff_100GHz = (dust_1_100GHz - signal(100.)) / dust_1_100GHz
        self.frac_diff_353GHz = (dust_1_353GHz - signal(353.)) / dust_1_353GHz

        d2_config = models("d2", 64)
        dust = components.Dust(d2_config[0])
        signal = dust.signal()
        
        dust_2_30GHz = read_map(os.path.join(test_data_dir, 'check6therm_30p0_64.fits'), 64, field = (0, 1, 2)) 
        dust_2_100GHz = read_map(os.path.join(test_data_dir, 'check6therm_100p0_64.fits'), 64, field = (0, 1, 2))
        dust_2_353GHz = read_map(os.path.join(test_data_dir, 'check6therm_353p0_64.fits'), 64, field = (0, 1, 2))

        self.model_2_frac_diff_30GHz = (dust_2_30GHz - signal(30.)) / dust_1_30GHz
        self.model_2_frac_diff_100GHz = (dust_2_100GHz - signal(100.)) / dust_1_100GHz
        self.model_2_frac_diff_353GHz = (dust_2_353GHz - signal(353.)) / dust_1_353GHz
        
        d3_config = models("d3", 64)
        dust = components.Dust(d3_config[0])
        signal = dust.signal()
        
        dust_3_30GHz = read_map(os.path.join(test_data_dir, 'check9therm_30p0_64.fits'), 64, field = (0, 1, 2))
        dust_3_100GHz = read_map(os.path.join(test_data_dir, 'check9therm_100p0_64.fits'), 64, field = (0, 1, 2))
        dust_3_353GHz = read_map(os.path.join(test_data_dir, 'check9therm_353p0_64.fits'), 64, field = (0, 1, 2))
        
        self.model_3_frac_diff_30GHz = (dust_3_30GHz - signal(30.)) / dust_3_30GHz
        self.model_3_frac_diff_100GHz = (dust_3_100GHz - signal(100.)) / dust_3_100GHz
        self.model_3_frac_diff_353GHz = (dust_3_353GHz - signal(353.)) / dust_3_353GHz

    def test_Dust_model_1(self):
        np.testing.assert_array_almost_equal(self.frac_diff_30GHz, np.zeros_like(self.frac_diff_30GHz), decimal = 6)
        np.testing.assert_array_almost_equal(self.frac_diff_100GHz, np.zeros_like(self.frac_diff_30GHz), decimal = 6)
        np.testing.assert_array_almost_equal(self.frac_diff_353GHz, np.zeros_like(self.frac_diff_30GHz), decimal = 6)


    def test_Dust_model_2(self):
        np.testing.assert_array_almost_equal(self.model_2_frac_diff_30GHz, np.zeros_like(self.model_2_frac_diff_30GHz), decimal = 6)
        np.testing.assert_array_almost_equal(self.model_2_frac_diff_100GHz, np.zeros_like(self.model_2_frac_diff_30GHz), decimal = 6)
        np.testing.assert_array_almost_equal(self.model_2_frac_diff_353GHz, np.zeros_like(self.model_2_frac_diff_30GHz), decimal = 6)

    def test_Dust_model_3(self):
        np.testing.assert_array_almost_equal(self.model_3_frac_diff_30GHz, np.zeros_like(self.model_3_frac_diff_30GHz), decimal = 6)
        np.testing.assert_array_almost_equal(self.model_3_frac_diff_100GHz, np.zeros_like(self.model_3_frac_diff_30GHz), decimal = 6)
        np.testing.assert_array_almost_equal(self.model_3_frac_diff_353GHz, np.zeros_like(self.model_3_frac_diff_30GHz), decimal = 6)

class test_Synchrotron(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pysm', 'template'))
        test_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data', 'benchmark'))
        
        s1_config = models("s1", 64)
        synchrotron = components.Synchrotron(s1_config[0])
        signal = synchrotron.signal()

        synch_1_30GHz = read_map(os.path.join(test_data_dir, 'check2synch_30p0_64.fits'), 64, field = (0, 1, 2))
        synch_1_100GHz = read_map(os.path.join(test_data_dir, 'check2synch_100p0_64.fits'), 64, field = (0, 1, 2))
        synch_1_353GHz = read_map(os.path.join(test_data_dir, 'check2synch_353p0_64.fits'), 64, field = (0, 1, 2))

        self.model_1_frac_diff = (synch_1_30GHz - signal(30.)) / synch_1_30GHz
        self.model_1_frac_diff = (synch_1_30GHz - signal(30.)) / synch_1_30GHz
        self.model_1_frac_diff = (synch_1_30GHz - signal(30.)) / synch_1_30GHz

        s2_config = models("s2", 64)
        synchrotron = components.Synchrotron(s2_config[0])
        signal = synchrotron.signal()
        
        synch_1_30GHz = read_map(os.path.join(test_data_dir, 'check7synch_30p0_64.fits'), 64, field = (0, 1, 2))
        synch_1_100GHz = read_map(os.path.join(test_data_dir, 'check7synch_100p0_64.fits'), 64, field = (0, 1, 2))
        synch_1_353GHz = read_map(os.path.join(test_data_dir, 'check7synch_353p0_64.fits'), 64, field = (0, 1, 2))

        self.model_2_frac_diff = (synch_2_30GHz - signal(30.)) / synch_2_30GHz
        self.model_2_frac_diff = (synch_2_30GHz - signal(30.)) / synch_2_30GHz
        self.model_2_frac_diff = (synch_2_30GHz - signal(30.)) / synch_2_30GHz

        s3_config = models("s3", 64)
        synchrotron = components.Synchrotron(s3_config[0])
        signal = synchrotron.signal()
        
        synch_3_30GHz = read_map(os.path.join(test_data_dir, 'check10synch_30p0_64.fits'), 64, field = (0, 1, 2))
        synch_3_100GHz = read_map(os.path.join(test_data_dir, 'check10synch_100p0_64.fits'), 64, field = (0, 1, 2))
        synch_3_353GHz = read_map(os.path.join(test_data_dir, 'check10synch_353p0_64.fits'), 64, field = (0, 1, 2))
        
        self.model_1_frac_diff = (synch_3_30GHz - signal(30.)) / synch_1_30GHz
        self.model_1_frac_diff = (synch_3_30GHz - signal(30.)) / synch_1_30GHz
        self.model_1_frac_diff = (synch_3_30GHz - signal(30.)) / synch_1_30GHz

        def test_Synch_model_1(self):
            np.testing.assert_array_almost_equal(self.model_1_frac_diff_30GHz, np.zeros_like(self.model_1_frac_diff_30GHz), decimal = 6)
            np.testing.assert_array_almost_equal(self.model_1_frac_diff_100GHz, np.zeros_like(self.model_1_frac_diff_30GHz), decimal = 6)
            np.testing.assert_array_almost_equal(self.model_1_frac_diff_353GHz, np.zeros_like(self.model_1_frac_diff_30GHz), decimal = 6)

        def test_Synch_model_2(self):
            np.testing.assert_array_almost_equal(self.model_2_frac_diff_30GHz, np.zeros_like(self.model_2_frac_diff_30GHz), decimal = 6)
            np.testing.assert_array_almost_equal(self.model_2_frac_diff_100GHz, np.zeros_like(self.model_2_frac_diff_30GHz), decimal = 6)
            np.testing.assert_array_almost_equal(self.model_2_frac_diff_353GHz, np.zeros_like(self.model_2_frac_diff_30GHz), decimal = 6)

        def test_Synch_model_3(self):
            np.testing.assert_array_almost_equal(self.model_3_frac_diff_30GHz, np.zeros_like(self.model_3_frac_diff_30GHz), decimal = 6)
            np.testing.assert_array_almost_equal(self.model_3_frac_diff_100GHz, np.zeros_like(self.model_3_frac_diff_30GHz), decimal = 6)
            np.testing.assert_array_almost_equal(self.model_3_frac_diff_353GHz, np.zeros_like(self.model_3_frac_diff_30GHz), decimal = 6)
            
class test_AME(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pysm', 'template'))
        test_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data', 'benchmark'))

        a1_config = models("a1", 64)
        AME1 = components.AME(a1_config[0])
        AME2 = components.AME(a1_config[1])

        signal = lambda nu: AME1.signal()(nu) + AME2.signal()(nu)

        ame_1_30GHz = read_map(os.path.join(test_data_dir, 'check3spinn_30p0_64.fits'), 64, field = (0, 1, 2)) 
        ame_1_100GHz = read_map(os.path.join(test_data_dir, 'check3spinn_100p0_64.fits'), 64, field = (0, 1, 2))
        ame_1_353GHz = read_map(os.path.join(test_data_dir, 'check3spinn_353p0_64.fits'), 64, field = (0, 1, 2))
        
        self.frac_diff_30GHz = (ame_1_30GHz[0] - signal(30.)[0]) / (ame_1_30GHz[0] + 1.e-14)
        self.frac_diff_100GHz = (ame_1_100GHz[0] - signal(100.)[0]) / (ame_1_100GHz[0] + 1e-14)
        self.frac_diff_353GHz = (ame_1_353GHz[0] - signal(353.)[0]) / (ame_1_353GHz[0] +1e-14)

    def tearDown(self):
        self.frac_diff_30GHz = None
        self.frac_diff_100GHz = None
        self.frac_diff_353GHz = None
        
    def test_AME_model_1(self):
        np.testing.assert_array_almost_equal(self.frac_diff_30GHz, np.zeros_like(self.frac_diff_30GHz), decimal = 6)
        np.testing.assert_array_almost_equal(self.frac_diff_100GHz, np.zeros_like(self.frac_diff_30GHz), decimal = 6)
        np.testing.assert_array_almost_equal(self.frac_diff_353GHz, np.zeros_like(self.frac_diff_30GHz), decimal = 6)
                                
class test_Freefree(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pysm', 'template'))
        test_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data', 'benchmark'))
                
        f1_config = models("f1", 64)

        freefree = components.Freefree(f1_config[0])
        signal = freefree.signal()

        freefree_1_30GHz = read_map(os.path.join(test_data_dir, 'check4freef_30p0_64.fits'), 64, field = (0, 1, 2))
        freefree_1_100GHz = read_map(os.path.join(test_data_dir, 'check4freef_100p0_64.fits'), 64, field = (0, 1, 2))
        freefree_1_353GHz = read_map(os.path.join(test_data_dir, 'check4freef_353p0_64.fits'), 64, field = (0, 1, 2))
                        
        self.frac_diff_30GHz = (freefree_1_30GHz[0] - signal(30.)[0]) / (freefree_1_30GHz[0] + 1.e-14)
        self.frac_diff_100GHz = (freefree_1_100GHz[0] - signal(100.)[0]) / (freefree_1_100GHz[0] + 1e-14)
        self.frac_diff_353GHz = (freefree_1_353GHz[0] - signal(353.)[0]) / (freefree_1_353GHz[0] +1e-14)

    def tearDown(self):
        self.frac_diff_30GHz = None
        self.frac_diff_100GHz = None
        self.frac_diff_353GHz = None

    def test_Freefree_model_1(self):
        np.testing.assert_array_almost_equal(self.frac_diff_30GHz, np.zeros_like(self.frac_diff_30GHz), decimal = 6)
        np.testing.assert_array_almost_equal(self.frac_diff_100GHz, np.zeros_like(self.frac_diff_30GHz), decimal = 6)
        np.testing.assert_array_almost_equal(self.frac_diff_353GHz, np.zeros_like(self.frac_diff_30GHz), decimal = 6)

class test_models_partial_sky(unittest.TestCase):
    """All models have same implementation, just testing freefree"""

    def test_partial_freefree(self):
        pixel_indices = np.arange(10000, 11000, dtype=np.int)
        f1_config = models("f1", 64, pixel_indices=pixel_indices)
        freefree = components.Freefree(f1_config[0])
        signal = freefree.signal()
        freefree_30_T = signal(30.)[0]
        assert len(freefree_30_T) == 1000
        test_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data', 'benchmark'))
        freefree_1_30GHz = read_map(os.path.join(test_data_dir, 'check4freef_30p0_64.fits'), 64, field = (0,))
        np.testing.assert_array_almost_equal(freefree_30_T, freefree_1_30GHz[pixel_indices], decimal = 3)


class test_CMB(unittest.TestCase):
        def setUp(self):
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pysm', 'template'))
            test_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data', 'benchmark'))
        
            self.cmb_config_1 = {
                'model' : 'taylens',
                'cmb_specs' : np.loadtxt(os.path.join(data_dir, 'camb_lenspotentialCls.dat'), unpack = True),
                'delens' : False,
                'delensing_ells' : np.loadtxt(os.path.join(data_dir, 'delens_ells.txt')),
                'nside' : 64,
                'cmb_seed' : 1234
            }

            cmb = components.CMB(self.cmb_config_1)
            signal = cmb.signal()

            self.cmb_1_30GHz = read_map(os.path.join(test_data_dir, 'check5cmb_30p0_64.fits'), 64, field = (0, 1, 2))
            cmb_1_100GHz = read_map(os.path.join(test_data_dir, 'check5cmb_100p0_64.fits'), 64, field = (0, 1, 2))
            cmb_1_353GHz = read_map(os.path.join(test_data_dir, 'check5cmb_353p0_64.fits'), 64, field = (0, 1, 2))

            self.frac_diff_30GHz = (self.cmb_1_30GHz - signal(30.)) / self.cmb_1_30GHz
            self.frac_diff_100GHz = (cmb_1_100GHz - signal(100.)) / cmb_1_100GHz
            self.frac_diff_353GHz = (cmb_1_353GHz - signal(353.)) / cmb_1_353GHz

        def tearDown(self):
            self.frac_diff_30GHz = None
            self.frac_diff_100GHz = None
            self.frac_diff_353GHz = None
            
        def test_CMB_model_1(self):
            
            np.testing.assert_array_almost_equal(self.frac_diff_30GHz, np.zeros_like(self.frac_diff_30GHz), decimal = 6)
            np.testing.assert_array_almost_equal(self.frac_diff_100GHz, np.zeros_like(self.frac_diff_30GHz), decimal = 6)
            np.testing.assert_array_almost_equal(self.frac_diff_353GHz, np.zeros_like(self.frac_diff_30GHz), decimal = 6)

        def test_CMB_partial_sky(self):
            cmb_config_partial_sky = self.cmb_config_1.copy()
            pixel_indices = np.arange(10000, 11000, dtype=np.int)
            cmb_config_partial_sky["pixel_indices"] = pixel_indices
            cmb = components.CMB(cmb_config_partial_sky)
            signal = cmb.signal()
            signal_30_T = signal(30.)[0]
            assert len(signal_30_T) == len(pixel_indices)
            np.testing.assert_array_almost_equal(signal_30_T, self.cmb_1_30GHz[0][pixel_indices], decimal=3)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
