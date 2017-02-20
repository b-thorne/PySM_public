# PySM: Python Sky Model  v-1.0
Most recent version available on [GitHub](https://github.com/bthorne93/PySM_public)

**Authors**: Ben Thorne, David Alonso, Sigurd Naess, Jo Dunkley 

**Contact**: ben.thorne@physics.ox.ac.uk

If you use the code for any publications, please acknowledge it and cite:
[Thorne et al 2016, arxiv: 1608.02841](http://arxiv.org/abs/1608.02841)

------------------------------------------------------------------------
### Outline:
This code generates full-sky simulations of Galactic foregrounds in intensity and
polarization relevant for CMB experiments. The components simulated are: thermal dust,
synchrotron, AME, free-free, and CMB at a given Nside, with an option to integrate over
a top hat bandpass, to add white instrument noise, and to smooth with a given beam.

There is scope for a few options for the model for each component, attempting to
be consistent with current data. The current v-1.0 version has typically two-three options
for each component.

Currently much of the available data is limited in resolution at degree-scale.  We therefore
make efforts to provide reasonable small-scale simulations to extend the data to higher
multipoles.  The details of the procedures developed can be found in the accompanying paper.

This code is based on the large-scale Galactic part of Planck Sky Model code and uses
some of its inputs (http://www.apc.univ-paris7.fr/~delabrou/PSM/psm.html,
[astro-ph/1207.3675](http://arxiv.org/abs/1207.3675)).

-----------------------------------------------------------------------

### Dependencies:
This code uses python, and needs the healpy, numpy, scipy and astropy modules.
Versions of those that it is known to work with are:

    - python 2.7.6
    - healpy 1.9.1
    - numpy 1.8.1
    - scipy 0.14.0
    - astropy 1.1.1

It requires at least:

    - healpy 1.9.1

Note that the healpy.write_map function will not work properly with outdated versions
of healpy.  healpy.write_map will also throw a warning when run with the most recent
versions of healpy and astropy because healpy uses a deprecated astropy function.
This does not affect the outcome of the code.

--------------------------------------------------------------------------
### Running the code
To run the code, in the directory containing main.py run:

    > python main.py main_config.ini

The default outputs are Healpix maps, at the specified frequencies, of the
summed emission of all the chosen components. The default output directory is './Output/'.

To change the parameters of the simulation edit the 'main_config.ini' file (or
create a separate configuration file). The different parameters are described
in the comments of this ini file as well as the individual model config files
in './ConfigFiles/<model>_config.ini'.

--------------------------------------------------------------------------
## Models
### Nominal

**'dust1'** = Thermal dust: Thermal dust is modelled as a single-component modified
 black body (mbb).  We use dust templates for emission at 545 GHz in intensity and
 353 GHz in polarisation from the Planck-2015 analysis, and scale these to different
 frequencies with a mbb spectrum using the spatially varying temperature and spectral
 index obtained from the Planck data using the Commander code (Planck Collaboration
 2015, arXiv:1502.01588). Note that it therefore assumes the same spectral index for
 polarization as for intensity.  The input intensity template at 545 GHz is simply the
 available 2048 product degraded to nside 512.  The polarization templates have been
 smoothed with a Gaussian kernel of FWHM 2.6 degrees, and had small scales added via
 the procedure described in the accompanying paper.

**'synchrotron1'** = Synchrotron:  A power law scaling is used for the synchrotron emission, with
 a spatially varying spectral index.  The emission templates are the Haslam 408 MHz, 57'
 resolution data reprocessed by Remazeilles et al 2015 MNRAS 451, 4311, and the WMAP 9-year
 23 GHz Q/U maps (Bennett, C.L., et.al., 2014, ApJS, 208, 20B). The polarization maps
 have been smoothed with a Gaussian kernel of FWHM 5 degrees and had small scales added.
 The intensity template has had small scales added straight to the template. The
 details of the small scale procedure is outlined in the accompanying paper.
 The spectral index map was derived using a combination of the Haslam 408 MHz data and WMAP 23
 GHz 7-year data (Miville-Deschenes, M.-A. et al., 2008, A&A, 490, 1093). The same scaling
 is used for intensity and polarization.  This is the same prescription as used in the
 Planck Sky Model's v1.7.8 'power law' option (Delabrouille et al. A&A 553, A96, 2013),
 but with the Haslam map updated to the Remazeilles version. A 'curved power law'
 model is also supported with a single isotropic curvature index. The amplitude of this
 curvature is taken from Kogut, A. 2012, ApJ, 753, 110.
 
 **'spdust1'** = Spinning Dust: We model the AME as a sum of two spinning dust populations
 based on the Commander code (Planck Collaboration 2015, arXiv:1502.01588). A component
 is defined by a degree-scale emission template at a reference frequency and a peak frequency
 of the emission law. Both populations have a spatially varying emission template, one
 population has a spatially varying peak frequency, and the other population has a
 spatially constant peak frequency.  The emission law is generated using the SpDust2 code
 [(Ali-Haimoud 2008)](http://arxiv.org/abs/0812.2904). The nominal model is unpolarized. We
 add small scales to the emission maps, the method is outlined in the accompanying paper.

**'freefree1'** = Free-Free: We model the free-free emission using the analytic model
 assumed in the Commander fit to the Planck 2015 data (Draine 2011 'Physics of the
 Interstellar and Intergalactic Medium') to produce a degree-scale map of free-free
 emission at 30 GHz. We add small scales to this using a procedure outlined in the
 accompanying paper.  This map is then scaled in frequency by applying a spatially
 constant power law index of -2.14.

**'cmb1'** = CMB: A lensed CMB realisation is computed using Taylens, a code to compute
 a lensed CMB realisation using nearest-neighbour Taylor interpolation
 (https://github.com/amaurea/taylens; Naess, S. K. and Louis, T. JCAP 09 001, 2013,
 astro-ph/1307.0719). This code takes, as an input, a set of unlensed Cl's generated
 using CAMB (http://www.camb.info/). The params.ini is in the Ancillary directory.
 There is a pre-computed CMB map provided at Nside 512.

### Alternatives

**'dust2'** (**'dust3'**) = emissivity that varies spatially on degree scales, drawn from a Gaussian
with beta=1.59 \pm 0.2 (0.3). A Gaussian variation is not physically motivated, but
amount of variation consistent with Planck.

**'dust4'** = a generalization of model 1 to multiple dust populations.  It has been found that
a two component model is still a good fit to the Planck data.  This option uses the two
component model from Finkbeiner, D. P., Davis, M., & Schlegel, D. J. 1999,Astrophysical Journal,
524, 867.

**'synchrotron2'** = synchrotron index steepens off the Galactic plane, from -3.0 in the
plane to -3.3 off the plane. Consistent with WMAP.

**'synchrotron3'** = a power law with a curved index. The model uses the same index map as
 the nominal model, plus a curvature term.  We use the best-fit curvature amplitude of
 -0.052 found in Kogut, A. 2012, ApJ, 753, 110, pivoted at 23 GHz.

**'spdust2'** = AME has 2% polarization fraction. Polarized maps simulated with thermal
dust angles and nominal AME intensity scaled globally by polarization fraction.
Within WMAP/Planck bounds.

----------------------------------------------------------------------------
