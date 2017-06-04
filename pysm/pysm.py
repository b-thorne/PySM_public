"""                                                                                            
.. module:: pysm
   :platform: Unix
   :synopsis: module containing primary use classes Sky and Instrument.

.. moduleauthor: Ben Thorne <ben.thorne@physics.ox.ac.uk> 
"""

import numpy as np
import healpy as hp
import scipy.constants as constants
import scipy.integrate
import os, sys
from components import Dust, Synchrotron, Freefree, AME, CMB
from common import read_key, convert_units, bandpass_convert_units, check_lengths

class Sky(object):
    """Model sky signal of Galactic foregrounds.

    This class combines the contribtions to the Galactic microwave
    foreground from thermal dust, synchrotron, AME, free-free, and CMB
    emissions.

    Is it inistialised using a dictionary. The keys must be 'cmb', 
    'dust', 'synchrotron', 'freefree', 'ame', and the values must be
    dictionaries with the configuration of the named component, e.g.:
    
    cmb_config = { 
    'model' : 'taylens', 
    'cmb_specs' : np.loadtxt('pysm/template/camb_lenspotentialCls.dat', unpack = True), 
    'delens' : False, 
    'delensing_ells' : np.loadtxt('pysm/template/delens_ells.txt'), 
    'nside' : nside,
    'cmb_seed' : 1111 
    }

    dust_config = {
    'model' : 'modified_black_body',
    'nu_0_I' : 545.,
    'nu_0_P' : 353.,
    'A_I' : pysm.read_map('pysm/template/dust_t_new.fits', nside, field = 0),
    'A_Q' : pysm.read_map('pysm/template/dust_q_new.fits', nside, field = 0),
    'A_U' : pysm.read_map('pysm/template/dust_u_new.fits', nside, field = 0),
    'spectral_index' : 1.5,
    'temp' : pysm.read_map('pysm/template/dust_temp.fits', nside, field = 0)
    }

    sky_config = { 
    'cmb' : cmb_config, 
    'dust': dust_config, 
    }

    """

    def __init__(self, config):
        """Read the configuration dict for Sky

        Implement the configuration file for the Sky instance. Then
        define the getattributes corresponding to the requested
        components.
        """
        self.__config = config
        self.__components = config.keys()

        if 'cmb' in self.Components:
            self.cmb = component_adder(CMB, self.Config['cmb'])
        if 'dust' in self.Components:
            self.dust = component_adder(Dust, self.Config['dust'])
        if 'synchrotron' in self.Components:
            self.synchrotron = component_adder(Synchrotron, self.Config['synchrotron'])
        if 'freefree' in self.Components:
            self.freefree = component_adder(Freefree, self.Config['freefree'])
        if 'ame' in self.Components:
            self.ame = component_adder(AME, self.Config['ame'])
        return 

    @property
    def Config(self):
        try:
            return self.__config
        except AttributeError:
            print("Sky attribute 'Config' not set.")
            sys.exit(1)
    
    @property
    def Components(self):
        try:
            return self.__components
        except AttributeError:
            print("Sky attribute 'Components' not set.")
            sys.exit(1)

    def signal(self):
        """Returns the sky as a function of frequency.

        This returns a function which is the sum of all the requested 
        sky components at the given frequency: (T, Q, U)(nu)."""
        def signal(nu):
            sig = 0.
            for component in self.Components:
                sig += getattr(self, component)(nu)
            return sig
        return signal

class Instrument(object):
    """This class contains the attributes and methods required to model
    the instrument observing Sky.

    Instrument contains methods used to perform bandpass integration over an arbitrary bandpass, smooth with a Gaussian beam, and a white Gaussian noise component.
    Instrument is initialised with dictionary, the possible keys are:

    - `frequencies` : frequencies at which to evaluate the Sky model -- numpy.ndarray.
    - `use_smoothing` : whether or not to use smoothing -- bool.
    - `beams` :  Gaussian beam FWHMs in arcmin. Only used if use_smoothing is True. Must be the same length as frequencies.
    - `add_noise` : whether or not to add noise -- bool
    - `sens_I` : sensitivity of intensity in uK_RJamin. Only used if add_noise is True. Must be same length as frequencies -- numpy.ndarray
    - `sens_P` : sensitivity of polarisation in uK_RJamin. Only used if add_noise is True. Must be same length as frequencies -- numpy.ndarray
    - `nside` : nside at which to evaluate maps -- int.
    - `noise_seed` : noise seed -- int.
    - `use_bandpass` : whether or not to use bandpass. If this is True `frequencies` is not required -- bool
    - `channels` : frequencies and weights of channels to be calculated as a list of tuples [(frequencies_1, weights_1), (frequencies_2, weights_2) ...] -- list of tuples
    - `channel_names` : list of names used to label the files to which channel maps are written -- string.
    - `output_directory` : directory to which the files will be written -- str.
    - `output_prefix` : prefix for all output files -- str.
    - `output_units` : output units -- str
    
    The use of Instrument is with the :class:`pysm.pysm.Sky` class. Given an instance of Sky we can use the :meth:`pysm.pysm.Instrument.obseve` to apply instrumental effects:
    >>> sky = pysm.Sky(sky_config)
    >>> instrument = pysm.Instrument(instrument_config)
    >>> instrument.observe(sky)
    """
    def __init__(self, config):
        """Specifies the attributes of the Instrument class."""
        for k in config.keys():
            read_key(self, k, config)
        if not self.Use_Bandpass:
            if not self.Use_Smoothing:
                self.__beams = np.zeros(len(self.Frequencies))
            if not self.Add_Noise:
                self.__sens_I = np.zeros(len(self.Frequencies))
                self.__sens_P = np.zeros(len(self.Frequencies))
        if self.Use_Bandpass:
            if not self.Use_Smoothing:
                self.__beams = np.zeros(len(self.Channels))
            if not self.Add_Noise:
                self.__sens_I = np.zeros(len(self.Channels))
                self.__sens_P = np.zeros(len(self.Channels))
        return

    @property
    def Frequencies(self):
        try:
            return self.__frequencies
        except AttributeError:
            print("Instrument attribute 'Frequencies' not set.")
            sys.exit(1)

    @property
    def Channels(self):
        try:
            return self.__channels
        except AttributeError:
            print("Instrument attribute 'Channels' not set.")
            sys.exit(1)
            
    @property
    def Beams(self):
        try:
            return self.__beams
        except AttributeError:
            print("Instrument attribute 'Beams' not set.")
            sys.exit(1)

    @property
    def Sens_I(self):
        try:
            return self.__sens_I
        except AttributeError:
            print("Instrument attribute 'Sens_I' not set.")
            sys.exit(1)

    @property
    def Sens_P(self):
        try:
            return self.__sens_P
        except AttributeError:
            print("Instrument attribute 'Sens_P' not set.")
            sys.exit(1)

    @property
    def Nside(self):
        try:
            return self.__nside
        except AttributeError:
            print("Instrument attribute 'Nside' not set.")
            sys.exit(1)

    @property
    def Noise_Seed(self):
        try:
            return self.__noise_seed
        except AttributeError:
            print("Instrument attribute 'Noise_Seed' not set.")
            sys.exit(1)

    @property
    def Use_Bandpass(self):
        try:
            return self.__use_bandpass
        except AttributeError:
            print("Instrument attribute 'Use_Bandpass' not set.")
            sys.exit(1)

    @property
    def Output_Prefix(self):
        try:
            return self.__output_prefix
        except AttributeError:
            print("Instrument attribute 'Output_Prefix' not set.")
            sys.exit(1)

    @property
    def Output_Directory(self):
        try:
            return self.__output_directory
        except AttributeError:
            print("Instrument attribute 'Output_Directory' not set.")
            sys.exit(1)
            
    @property
    def Channel_Names(self):
        try:
            return self.__channel_names
        except AttributeError:
            print("Instrument attribute 'Channel_Names' not set.")
            sys.exit(1)

    @property 
    def Write_Components(self):
        try:
            return self.__write_components
        except AttributeError:
            print("Instrument attribute 'Write_Components' not set.")
            sys.exit(1)

    @property
    def Add_Noise(self):
        try:
            return self.__add_noise
        except AttributeError:
            print("Instrument attribute 'Add_Noise' not set.")

    @property
    def Use_Smoothing(self):
        try:
            return self.__use_smoothing
        except AttributeError:
            print("Instrument attribute 'Use_Smoothing' not set.")

    @property
    def Output_Units(self):
        try:
            return self.__output_units
        except AttributeError:
            print("Instrument attribute 'Output_Units not set.'")
            
    def observe(self, Sky):
        """Evaluate and add instrument effects to Sky's signal function.

        This method evaluates the Sky class's signal method at the
        requested frequencies, or over the requested bandpass. Then
        smooths with a Gaussian beam, if requested. Then adds Gaussian
        white noise, if requested. Finally writes the maps to file.

        :param Sky: instance of the :class:`pysm.pysm.Sky` class. 
        :type Sky: class
        :return: no return, writes to file.

        """
        self.print_info()
        
        signal = Sky.signal()
        output = self.apply_bandpass(signal) 
        output = self.smoother(output)
        noise = self.noiser()
        output, noise = self.unit_converter(output, noise)
        self.writer(output, noise)
        return 
        
    def apply_bandpass(self, signal):
        """Function to integrate signal over a bandpass.  Frequencies must be
        evenly spaced, if they are not the function will object. Weights
        must be normalisable.

        :param signal: signal function to be integrated of bandpass
        :type param: function
        :return: maps after bandpass integration shape either (N_freqs, 3, Npix) or (N_channels, 3, Npix) -- numpy.ndarray
        
        """
        if not self.Use_Bandpass:
            return signal(self.Frequencies)
        elif self.Use_Bandpass:
            # convert to MJysr in order to integrate over bandpass
            signal_MJysr = lambda nu: signal(nu) * convert_units("uK_RJ", "Jysr", nu)
            return np.array(map(lambda (f, w): bandpass(f, w, signal_MJysr), self.Channels))
        else:
            print("Please set 'Use_Bandpass' for Instrument object.")
            sys.exit(1)
            
    def smoother(self, map_array):
        """Function to smooth an array of N (T, Q, U) maps with N beams in
        units of arcmin.

        :param map_array:
        :type map_array:
        
        """
        if not self.Use_Smoothing:
            return map_array
        elif self.Use_Smoothing:
            smooth = lambda (m, b): hp.smoothing(m, fwhm = np.pi / 180. * b / 60., verbose = False)
            return np.array(map(smooth, zip(map_array, self.Beams)))
        else:
            print("Please set 'Use_Smoothing' in Instrument object.")
            sys.exit(1)

    def noiser(self):
        """Calculate white noise maps for given sensitivities.  Returns signal
        + noise, and noise maps at the given nside in (T, Q, U). Input
        sensitivities are expected to be in uK_CMB amin for the rest of
        PySM.

        :param map_array: array of maps to which we add noise. 
        :type map_array: numpy.ndarray.
        :return: map plus noise, and noise -- numpy.ndarray

        """
        npix = hp.nside2npix(self.Nside)
        if not self.Add_Noise:
            return np.zeros((len(self.Sens_I), 3, npix))
        elif self.Add_Noise:
            # solid angle per pixel in amin2
            pix_amin2 = 4. * np.pi / float(npix) * (180. * 60. / np.pi) ** 2
            """sigma_pix_I/P is std of noise per pixel. It is an array of length
            equal to the number of input maps."""
            sigma_pix_I = np.sqrt(self.Sens_I ** 2 / pix_amin2)
            sigma_pix_P = np.sqrt(self.Sens_P ** 2 / pix_amin2)
            noise = np.random.randn(len(self.Sens_I), 3, npix)
            noise[:, 0, :] *= sigma_pix_I[:, None]
            noise[:, 1, :] *= sigma_pix_P[:, None]
            noise[:, 2, :] *= sigma_pix_P[:, None]
            return noise
        else:
            print("Please set 'Add_Noise' in Instrument object.")
            sys.exit(1)

    def unit_converter(self, map_array, noise):
        """Function to handle the conversion of units. 

        If using delta bandpasses just evaluate the unit conversion
        factor normally. If using a bandpass we calculate the
        conversion factor following the Planck HFI definitions.

        :param map_array: signal + noise map to convert units of.
        :type map_array: numpy.ndarray
        :param noise: noise map to conver units of.
        :type noise: numpy.ndarray
        :return: signal + noise map converted to output units, noise map converted to output units -- numpy.ndarray
        """
        if not self.Use_Bandpass:
            #If using a delta bandpass just evaluate the standard unit conversion at
            #the frequencies of interest. All the scaling is done in uK_RJ.
            Uc_signal = np.array(convert_units("uK_RJ", self.Output_Units, self.Frequencies))
        elif self.Use_Bandpass:
            # In the case of a given bandpass we calculate the unit conversion as explained in the documentation
            # of bandpass_convert_units.
            Uc_signal = np.array(map(lambda channel: bandpass_convert_units(self.Output_Units, channel), self.Channels))
        if self.Add_Noise:
            # If noise requested also multiple the calculated noise.
            if not self.Use_Bandpass:
                Uc_noise = np.array(convert_units("uK_CMB", self.Output_Units, self.Frequencies))
            elif self.Use_Bandpass:
                # first convert noise to MJysr and then to the output units.
                CMB2MJysr = map(lambda channel: 1. / bandpass_convert_units("uK_CMB", channel), self.Channels)
                Uc_noise = np.array(map(lambda channel: bandpass_convert_units(self.Output_Units, channel), self.Channels))
        elif not self.Add_Noise:
            Uc_noise = np.zeros_like(Uc_signal)
        return Uc_signal[:, None, None] * map_array, Uc_noise[:, None, None] * noise
            
    def file_path(self, channel_name = None, f = None, extra_info = ""):
        """Returns file path for pysm outputs.
        """
        if not self.Use_Bandpass:
            fname = '%s_nu%sGHz_%s_nside%04d.fits'%(self.Output_Prefix, str("%07.2f"%f).replace(".", "p"), extra_info, self.Nside)
        elif self.Use_Bandpass:
            fname = '%s_bandpass_%s_%s_nside%04d.fits'%(self.Output_Prefix, channel_name, extra_info, self.Nside)
        else:
            print("Bandpass set incorrectly.")
            sys.exit(1)
        return os.path.join(self.Output_Directory, fname)

    def writer(self, output, noise):
        """Function to write the total and noise maps to file."""
        if not self.Use_Bandpass:
            if self.Add_Noise:
                for f, o, n in zip(self.Frequencies, output, noise):
                    hp.write_map(self.file_path(f = f, extra_info = "noise"), n)
                    hp.write_map(self.file_path(f = f, extra_info = "total"), o + n)
            elif not self.Add_Noise:
                for f, o in zip(self.Frequencies, output):
                    hp.write_map(self.file_path(f = f, extra_info = "total"), o)
        elif self.Use_Bandpass:
            if self.Add_Noise:
                for c, o, n in zip(self.Channel_Names, output, noise):
                    hp.write_map(self.file_path(channel_name = c, extra_info = "total"), o + n)
                    hp.write_map(self.file_path(channel_name = c, extra_info = "noise"), n)
            elif not self.Add_Noise:
                for c, o in zip(self.Channel_Names, output):
                    hp.write_map(self.file_path(channel_name = c, extra_info = "total"), o)
        return

    def print_info(self):
        """Function to print information about current Instrument
        specifications to screen.

        """
        if not self.Use_Bandpass:
            if not check_lengths(self.Frequencies, self.Beams, self.Sens_I, self.Sens_P):
                print("Check lengths of frequencies, beams, and sensitivities are equal.")
                sys.exit(1)

            print("nu (GHz) | sigma_I (uK_CMB amin) | sigma_P (uK_CMB amin) | FWHM (arcmin) \n")
            for f, s_I, s_P, b in zip(self.Frequencies, self.Sens_I, self.Sens_P, self.Beams):
                print("%07.2f | %05.2f | %05.2f | %05.2f "%(f, s_I, s_P, b))

        elif self.Use_Bandpass:
            print("Channel name | sigma_I (uK_CMB amin) | sigma_P (uK_amin) | FWHM (arcmin) |")
            for cn, s_I, s_P, b in zip(self.Channel_Names, self.Sens_I, self.Sens_P, self.Beams):
                print("%s | %05.2f | %05.2f | %05.2f "%(cn, s_I, s_P, b)) 
        return
    
def bandpass(frequencies, weights, signal):
    """Function to integrate signal over a bandpass.

    Frequencies must be evenly spaced, if they are not the function
    will object. Weights must be able to be normalised to integrate to 1.

    """
    # check that the frequencies are evenly spaced.
    check_bpass_frequencies(frequencies)
    frequency_separation = frequencies[1] - frequencies[0]
    # normalise the weights and check that they integrate to 1.
    weights /= np.sum(weights * frequency_separation)
    check_bpass_weights_normalisation(weights, frequency_separation)
    # define the integration: integrand = signal(nu) * w(nu) * d(nu)
    # signal is already in MJysr.
    return sum(map(lambda (nu, w): signal(nu) * w * frequency_separation, zip(frequencies, weights)))
    
def check_bpass_weights_normalisation(weights, spacing):
    """Function that checks the weights of the bandpass were normalised
    properly.

    """
    try:
        np.testing.assert_almost_equal(np.sum(weights * spacing), 1, decimal = 3)
    except AssertionError:
        print("Bandpass weights can not be normalised.")
        sys.exit(1)
    return

def check_bpass_frequencies(frequencies):
    """Function checking the separation of frequencies are even."""
    frequency_separation = frequencies[1] - frequencies[0]
    number_of_frequencies = frequencies.size
    frequency_range = frequencies[-1] - frequencies[0]
    try:
        np.testing.assert_almost_equal(frequency_separation * (number_of_frequencies - 1)/ frequency_range, 1., decimal = 3)
    except AssertionError:
        print("Bandpass frequencies not evenly spaced.")
        sys.exit(1)
    for i in range(frequencies.size - 1):
        spacing = frequencies[i + 1] - frequencies[i]
        try:
            np.testing.assert_almost_equal(spacing / frequency_range, frequency_separation / frequency_range, decimal = 3)
        except AssertionError:
            print("Bandpass frequencies not evenly spaced.")
            sys.exit(1) 
    return

def component_adder(component_class, dictionary_list):
    """This function adds instances of a component class to a Sky
    attribute for that component, e.g. Sky.Dust, thereby allowing for
    multiple populations of that component to be simulated.

    """
    # need this step in order to avoid calling the setup for
    # each scaling law every time the signal is evaluated.
    # each dictionary is a configuration dict used to
    # instantiate the component's class. We then take the
    # signal produced by that population.
    population_signals = map(lambda dic: component_class(dic).signal(), dictionary_list)
    # sigs is now a list of functions. Each function is the emission
    # due to a population of the component.
    def total_signal(nu):
        total_component_signal = 0
        # now sum up the contributions of each population at
        # frequency nu. 
        for population_signal in population_signals:
            total_component_signal += population_signal(nu)
        return total_component_signal
    # return the total contribution from all populations
    # as a function of frequency nu. 
    return total_signal
