import numpy as np
import scipy as sp
import healpy as hp, sys, time
from scipy.misc import factorial, comb
from scipy.interpolate import interp1d
import os

constants = {
    'T_CMB':2.7255,
    'h':6.62607004e-34,
    'k_B':1.38064852e-23,
    'c':2.99792e8
}

units = {
    'n':1.e-9,
    'u':1.e-6,
    'm':1.e-3,
    '1': 1.,
    'k':1.e3,
    'M':1.e6,
    'G':1.e9,
    'K_RJ': lambda x: 3.072387e4*x**2,
    'K_CMB': lambda x:  (0.017608676*x/(np.exp(0.017608676*x)-1))**2*np.exp(0.017608676*x)*3.072387e4*x**2,
    'Jysr': lambda x: np.ones(x.size)
}

def condense_list(models):
    mod_names =[f[1] for f in models]
    return [(f[0],' '.join(mod_names))]

def file_path(o,j):
    comps = str()
    for k in sorted(o.components): 
        comps=''.join([comps,k[0:5],'_'])
    fname = ''.join([o.output_prefix,comps, str(o.output_frequency[j]).replace('.', 'p'),'_', str(o.nside), '.fits'])
    path = os.path.join(o.output_dir, fname)
    return path

def write_output_single(sky_freq,o,Config,i):
    path = file_path(o,i)
    hp.write_map(path, hp.ud_grade(sky_freq, nside_out=o.nside), coord='G', column_units = ''.join(o.output_units), column_names = None, extra_header = config2list(Config,o,i))

def config2list(config,o,i):
    info = []

    exclude = ['__name__','output_frequency','instrument_noise',
               'instrument_noise_i','instrument_noise_pol','smoothing',
               'fwhm','bandpass','bandpass_widths','instrument_noise_seed','output_dir']

    for f in config.sections(): 
        info += config._sections[f].items()

    info = filter(lambda f: (not f[0] in exclude),info)
    models = filter(lambda f: f[0]=='model',info)
    not_models = filter(lambda f: not f[0]=='model',info)
    models = condense_list(models)
    
    info = not_models+models
    
    info += [('freq',o.output_frequency[i],'GHz')]

    if o.smoothing:
        info += [('fwhm',o.fwhm[i],'degrees')]

    if o.bandpass:
        info += [('bandpass_width',o.bandpass_widths[i],'GHz')]

    if o.instrument_noise:
        info += [('instrument_noise_i',o.instrument_noise_i[i],'uK_CMB amin'),
                 ('instrument_noise_pol',o.instrument_noise_pol[i],'uK_CMB amin'),
                  ('instrument_noise_seed',o.instrument_noise_seed)]
    info = add_hierarch(info)
    return info

def add_hierarch(lis):
    for i, item in enumerate(lis):
        if len(item) == 3:
            lis[i]= ('HIERARCH '+item[0],item[1],item[2])
        else:
            lis[i]= ('HIERARCH '+item[0],item[1])
    return lis

def read_map_wrapped(fname,nside_out,field=0) :
    return hp.ud_grade(np.array(hp.read_map(fname,field=field,verbose=False)),nside_out=nside_out)
# Switch to this if you don't want to ud_grade on input
#    return hp.read_map(fname,field=field,verbose=False)

class component(object):

    def __init__(self,cdict,nside_out):
        keys = cdict.keys()
        if 'pol' in keys:
            self.pol = cdict['pol']
        if 'spectral_model' in keys:
            self.spectral_model = cdict['spectral_model']
        if 'em_template' in keys:
            self.em_template = read_map_wrapped(cdict['em_template'],nside_out)
        if 'beta_template' in keys:
            self.beta_template = read_map_wrapped(cdict['beta_template'],nside_out)
        if 'temp_template' in keys:
            self.temp_template = read_map_wrapped(cdict['temp_template'],nside_out)
        if 'freq_curve' in keys:
            self.freq_curve = float(cdict['freq_curve'])
        if 'beta_curve' in keys:
            self.beta_curve = float(cdict['beta_curve'])
        if 'polq_em_template' in keys: 
            self.polq_em_template = read_map_wrapped(cdict['polq_em_template'],nside_out)
        if 'polu_em_template' in keys:
            self.polu_em_template = read_map_wrapped(cdict['polu_em_template'],nside_out)
        if 'freq_ref' in keys:
            self.freq_ref = float(cdict['freq_ref'])
        if 'pol_freq_ref' in keys:
            self.pol_freq_ref = float(cdict['pol_freq_ref'])
        if 'template_units' in keys:
            self.template_units = [cdict['template_units'][0],cdict['template_units'][1:]]
        if 'output_dir' in keys:
            self.output_dir = cdict['output_dir']
        if 'specs' in keys: 
            self.specs = cdict['specs']
        if 'cmb_seed' in keys:
            self.cmb_seed = int(cdict['cmb_seed'])
        if 'compute_lensed_cmb' in keys:
            self.compute_lensed_cmb = 'True' in cdict['compute_lensed_cmb']
            if self.compute_lensed_cmb == False: self.lensed_cmb = read_map_wrapped(cdict['lensed_cmb'],nside_out,field=(0,1,2))
        if 'emissivity' in keys:
            self.emissivity = np.loadtxt(cdict['emissivity'],unpack=True)
        if 'freq_peak' in keys:
            try: self.freq_peak = float(cdict['freq_peak'])
            except ValueError: self.freq_peak = read_map_wrapped(cdict['freq_peak'],nside_out)
        if 'peak_ref' in keys:
            self.peak_ref = float(cdict['peak_ref'])
        if 'thermaldust_polq' in keys:
            self.thermaldust_polq = read_map_wrapped(cdict['thermaldust_polq'],nside_out)
        if 'thermaldust_polu' in keys:
            self.thermaldust_polu = read_map_wrapped(cdict['thermaldust_polu'],nside_out)
        if 'pol_frac' in keys:
            self.pol_frac = float(cdict['pol_frac'])
        if 'delens' in keys:
            self.delens = 'True' in cdict['delens']
        if 'delensing_ells' in keys:
            self.delensing_ells = np.loadtxt(cdict['delensing_ells'],unpack=True)
        if 'ff_em_temp' in keys:
            self.em = read_map_wrapped(cdict['ff_em_temp'],nside_out)
        if 'ff_te_temp' in keys:
            self.te = read_map_wrapped(cdict['ff_te_temp'],nside_out)
            
class output(object):
    def __init__(self, config_dict):
        self.output_prefix = config_dict['output_prefix']
        if 'debug' in config_dict :
            self.debug = 'True' in config_dict['debug']
        else :
            self.debug = False
        self.components = [i for i in config_dict['components'].split()]
        self.output_frequency = [float(i) for i in config_dict['output_frequency'].split()]
        self.output_units = [config_dict['output_units'][0],config_dict['output_units'][1:]]
        self.nside = int(config_dict['nside'])
        self.output_dir = config_dict['output_dir']
        self.bandpass = 'True' in config_dict['bandpass']
        self.bandpass_widths = [float(i) for i in config_dict['bandpass_widths'].split()]
        self.instrument_noise = 'True' in config_dict['instrument_noise']
        if config_dict['instrument_noise_seed'] == 'None': 
            self.instrument_noise_seed = None
        else:
            self.instrument_noise_seed = int(config_dict['instrument_noise_seed'])
        self.instrument_noise_i = np.asarray([float(i) for i in config_dict['instrument_noise_i'].split()])
        self.instrument_noise_pol = np.asarray([float(i) for i in config_dict['instrument_noise_pol'].split()])
        self.smoothing = 'True' in config_dict['smoothing']
        self.fwhm = [float(i) for i in config_dict['fwhm'].split()]


def convert_units(u_from, u_to, freq): #freq in GHz

    if u_from[0] not in units.keys():
        if u_to[0] not in units.keys(): return units[u_from[0]+u_from[1]](np.asarray(freq))/units[u_to[0]+u_to[1]](np.asarray(freq))
        else: return units[u_from[0]+u_from[1]](np.asarray(freq))/(units[u_to[0]]*units[u_to[1]](np.asarray(freq)))

    else: 
        if u_to[0] not in units.keys(): return units[u_from[0]]*units[u_from[1]](np.asarray(freq))/units[u_to[0]+u_to[1]](np.asarray(freq))
        else: return units[u_from[0]]*units[u_from[1]](np.asarray(freq))/(units[u_to[0]]*units[u_to[1]](np.asarray(freq)))




def scale_freqs(c, o, pol=None, samples=10.):

#All scalings, other than the CMB, are done Rayleigh-Jeans units.
    
     freq = np.asarray(np.copy(o.output_frequency))

     if pol == False: 
         freq_ref = np.copy(c.freq_ref)
     if pol == True: 
         freq_ref = np.copy(c.pol_freq_ref)

     if o.bandpass: 

         widths = np.asarray([np.linspace(-(samples-1.)*w/(samples*2.),(samples-1)*w/(samples*2.),num=samples) for w in o.bandpass_widths])
         freq = freq[...,np.newaxis]+widths
         freq_cen = np.asarray(np.copy(o.output_frequency))



#Note that the frequencies within a bandwidth are stored in the second dimension of the frequency array.  When we sum over the final produce of each power law we therefore specify the ndim(freq)-1 dimension. Freq has two dimensions at this point and so ndim(freq)-1 = 1.  Axis indexing starts at 0. So this gives us the correct summation.

#Note that the bandpass has to be done in non-thermodynamic units, so there are factors of frequency squared inside and outside the integral to account for switching between the two unit systems.

     if c.spectral_model=="curvedpowerlaw": 
         if not o.bandpass: 
             return (freq[...,np.newaxis]/freq_ref)**(c.beta_template+c.beta_curve*np.log(freq[...,np.newaxis]/c.freq_curve))
         else: 
             return (1./freq_cen**2)[...,np.newaxis]*np.sum((freq**2)[...,np.newaxis]*(freq[...,np.newaxis]/c.freq_ref)**(c.beta_template+c.beta_curve*np.log(freq[...,np.newaxis]/c.freq_curve)),axis=np.ndim(freq)-1)/samples


     if c.spectral_model=="powerlaw": 
         if not o.bandpass: 
             return (freq[...,np.newaxis]/freq_ref)**c.beta_template
         else: 
             return (1./freq_cen**2)[...,np.newaxis]*np.sum((freq**2)[...,np.newaxis]*(freq[...,np.newaxis]/freq_ref)**c.beta_template,axis=np.ndim(freq)-1)/samples


     if c.spectral_model=="thermaldust":
        exponent=(constants['h']/constants['k_B'])*(freq[...,np.newaxis]*1.e9/c.temp_template)
        exponent_ref=(constants['h']/constants['k_B'])*(freq_ref*1.e9/c.temp_template)
        if not o.bandpass: 
            return (freq[...,np.newaxis]/freq_ref)**(c.beta_template+1)*((np.exp(exponent_ref)-1.)/(np.exp(exponent)-1.))
        else: 
            return (1./freq_cen**2)[...,np.newaxis]*np.sum((freq**2)[...,np.newaxis]*(freq[...,np.newaxis]/freq_ref)**(c.beta_template+1) * ( (np.exp(exponent_ref)-1.) / (np.exp(exponent)-1.) ) , axis=np.ndim(freq)-1 ) / samples


     if  c.spectral_model=="cmb":
         if o.bandpass == False: return convert_units(['u','K_CMB'],o.output_units,o.output_frequency)[np.newaxis,:,np.newaxis]
         else: return (1./freq_cen**2)[...,np.newaxis]*np.sum((freq**2)[...,np.newaxis]*convert_units(['u','K_CMB'],o.output_units,freq)[...,np.newaxis], axis=np.ndim(freq)-1 ) / samples

     if c.spectral_model=="spdustnum":

         J = interp1d(c.emissivity[0],c.emissivity[1],bounds_error=False,fill_value=0)
         arg1 = freq[...,np.newaxis]*c.peak_ref/c.freq_peak
         arg2 = c.freq_ref*c.peak_ref/c.freq_peak

         if not o.bandpass: 
             return ((c.freq_ref/freq)**2)[...,np.newaxis]*(J(arg1)/J(arg2))
         else: 
             return (1./freq_cen**2)[...,np.newaxis]*np.sum((freq**2)[...,np.newaxis]*((c.freq_ref/freq)**2)[...,np.newaxis] * (J(arg1)/J(arg2)), axis=np.ndim(freq)-1 ) / samples

     if c.spectral_model=="freefree":
         if not o.bandpass: 
             return (freq[...,np.newaxis]/c.freq_ref)**-2.14
         else: 
             return (1/freq_cen**2)[...,np.newaxis]*np.sum((freq**2)[...,np.newaxis]*((c.freq_ref/freq)**2)[...,np.newaxis]*(freq[...,np.newaxis]/c.freq_ref)**-2.14, axis=np.ndim(freq)-1 ) / samples

     else:
        print('No law selected')
        exit()

#The following code is edited from the taylens code: Naess, S. K. and Louis, T. 2013 'Lensing simulations by Taylor expansion -  not so inefficient after all'  Journal of Cosmology and Astroparticle Physics September 2013
#Available at: https://github.com/amaurea/taylens

# This generates correlated T,E,B and Phi maps                                             
def simulate_tebp_correlated(cl_tebp_arr,nside,lmax,seed):
        np.random.seed(seed)
        alms=hp.synalm(cl_tebp_arr,lmax=lmax,new=True)
        aphi=alms[-1]
        acmb=alms[0:-1]
#Set to zero above map resolution to avoid aliasing                                        
        beam_cut=np.ones(3*nside)
        for ac in acmb :
                hp.almxfl(ac,beam_cut,inplace=True)
        cmb=np.array(hp.alm2map(acmb,nside,pol=True,verbose=False))

        return cmb,aphi

# This function is the core of Taylens.                                                              
def taylor_interpol_iter(m, pos, order=3, verbose=False, lmax=None):
        """Given a healpix map m[npix], and a set of positions                                       
        pos[{theta,phi},...], evaluate the values at those positions                                 
        using harmonic Taylor interpolation to the given order (3 by                                 
        default). Successively yields values for each cumulative order                               
        up to the specified one. If verbose is specified, it will print                              
        progress information to stderr."""
        nside = hp.npix2nside(m.size)
        if lmax is None: lmax = 3*nside
        # Find the healpix pixel centers closest to pos,                                             
        # and our deviation from these pixel centers.                                                
        ipos = hp.ang2pix(nside, pos[0], pos[1])
        pos0 = np.array(hp.pix2ang(nside, ipos))
        dpos = pos[:2]-pos0
        # Take wrapping into account                                                                 
        bad = dpos[1]>np.pi
        dpos[1,bad] = dpos[1,bad]-2*np.pi
        bad = dpos[1]<-np.pi
        dpos[1,bad] = dpos[1,bad]+2*np.pi

        # Since healpix' dphi actually returns dphi/sintheta, we choose                              
        # to expand in terms of dphi*sintheta instead.                                               
        dpos[1] *= np.sin(pos0[0])
        del pos0

        # We will now Taylor expand our healpix field to                                             
        # get approximations for the values at our chosen                                            
        # locations. The structure of this section is                                                
        # somewhat complicated by the fact that alm2map_der1 returns                                 
        # two different derivatives at the same time.                                                
        derivs = [[m]]
        res = m[ipos]
        yield res
        for o in range(1,order+1):
                # Compute our derivatives                                                            
                derivs2 = [None for i in range(o+1)]
                used    = [False for i in range(o+1)]
                # Loop through previous level in steps of two (except last)                          
                if verbose: tprint("order %d" % o)
                for i in range(o):
                        # Each alm2map_der1 provides two derivatives, so avoid                       
                        # doing double work.                                                         
                        if i < o-1 and i % 2 == 1:
                                continue
                        a = hp.map2alm(derivs[i], use_weights=True, lmax=lmax, iter=0)
                        derivs[i] = None
                        dtheta, dphi = hp.alm2map_der1(a, nside, lmax=lmax)[-2:]
                        derivs2[i:i+2] = [dtheta,dphi]
                        del a, dtheta, dphi
                        # Use these to compute the next level                                        
                        for j in range(i,min(i+2,o+1)):
                                if used[j]: continue
                                N = comb(o,j)/factorial(o)
                                res += N * derivs2[j][ipos] * dpos[0]**(o-j) * dpos[1]**j
                                used[j] = True
                                # If we are at the last order, we don't need to waste memory         
                                # storing the derivatives any more                                   
                                if o == order: derivs2[j] = None
                derivs = derivs2
                yield res

# The following functions are support routines for reading                                           
# input data and preparing it for being lensed. Most of them                                 
# are only needed to take care of tiny, curvature-related                                            
# effects that can be safely ignored.                                                                
def readspec(fname):
        """Read a power spectrum with columns [l,comp1,comp2,....]                                   
        into a 2d array indexed by l. Entries with missing data are                                  
        filled with 0."""
        tmp = np.loadtxt(fname).T
        l, tmp = tmp[0], tmp[1:]
        res = np.zeros((len(tmp),np.max(l)+1))
        res[:,np.array(l,dtype=int)] = tmp
        return res

def offset_pos(ipos, dtheta, dphi, pol=False, geodesic=False):
        """Offsets positions ipos on the sphere by a unit length step                                
        along the gradient dtheta, dphi/sintheta, taking the curvature                               
        of the sphere into account. If pol is passed, also computes                                  
        the cos and sin of the angle by which (Q,U) must be rotated to                               
        take into account the change in local coordinate system.                                     
                                                                                                     
        If geodesic is passed, a quick and dirty, but quite accurate, approximation                  
        is used.                                                                                     
                                                                                                     
        Uses the memory of 2 maps (4 if pol) (plus that of the input maps)."""
        opos = np.zeros(ipos.shape)
        if pol and not geodesic: orot = np.zeros(ipos.shape)
        else: orot = None
        if not geodesic:
                # Loop over chunks in order to conserve memory                                       
                step = 0x10000
                for i in range(0, ipos.shape[1], step):
                        small_opos, small_orot = offset_pos_helper(ipos[:,i:i+step], dtheta[i:i+step\
], dphi[i:i+step], pol)
                        opos[:,i:i+step] = small_opos
                        if pol: orot[:,i:i+step] = small_orot
        else:
                opos[0] = ipos[0] + dtheta
                opos[1] = ipos[1] + dphi/np.sin(ipos[0])
                opos = fixang(opos)
        return opos, orot

def offset_pos_helper(ipos, dtheta, dphi, pol):
        grad = np.array((dtheta,dphi))
        dtheta, dphi = None, None
        d = np.sum(grad**2,0)**0.5
        grad  /= d
        cosd, sind = np.cos(d), np.sin(d)
        cost, sint = np.cos(ipos[0]), np.sin(ipos[0])
        ocost  = cosd*cost-sind*sint*grad[0]
        osint  = (1-ocost**2)**0.5
        ophi   = ipos[1] + np.arcsin(sind*grad[1]/osint)
        if not pol:
                return np.array([np.arccos(ocost), ophi]), None
        A      = grad[1]/(sind*cost/sint+grad[0]*cosd)
        nom1   = grad[0]+grad[1]*A
        denom  = 1+A**2
        cosgam = 2*nom1**2/denom-1
        singam = 2*nom1*(grad[1]-grad[0]*A)/denom
        return np.array([np.arccos(ocost), ophi]), np.array([cosgam,singam])

def fixang(pos):
        """Handle pole wraparound."""
        a = np.array(pos)
        bad = np.where(a[0] < 0)
        a[0,bad] = -a[0,bad]
        a[1,bad] = a[1,bad]+np.pi
        bad = np.where(a[0] > np.pi)
        a[0,bad] = 2*np.pi-a[0,bad]
        a[1,bad] = a[1,bad]+np.pi
        return a

def apply_rotation(m, rot):
        """Update Q,U components in polarized map by applying                                        
        the rotation rot, representat as [cos2psi,sin2psi] per                                       
        pixel. Rot is one of the outputs from offset_pos."""
        if len(m) < 3: return m
        if rot is None: return m
        m = np.asarray(m)
        res = m.copy()
        res[1] = rot[0]*m[1]-rot[1]*m[2]
        res[2] = rot[1]*m[1]+rot[0]*m[2]
        return m

# Set up progress prints                                                                             
t0 = None
def silent(msg): pass
def tprint(msg):
        global t0
        if t0 is None: t0 = time.time()
        print >> sys.stderr, "%8.2f %s" % (time.time()-t0,msg)

