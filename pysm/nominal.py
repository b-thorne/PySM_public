from __future__ import absolute_import
from .common import read_map
import numpy as np
from healpy import nside2npix
import os

data_dir = os.path.join(os.path.dirname(__file__), 'template')
template = lambda x: os.path.join(data_dir, x)

def models(key, nside, pixel_indices=None):
    model = eval(key)(nside, pixel_indices=pixel_indices)
    model['pixel_indices'] = pixel_indices # include pixel indices in the model dictionary
    return model

def d0(nside, pixel_indices=None):
    return [{
        'model': 'modified_black_body',
        'nu_0_I': 545.,
        'nu_0_P': 353.,
        'A_I': read_map(template('dust_t_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_Q': read_map(template('dust_q_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_U': read_map(template('dust_u_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'spectral_index': np.ones(nside2npix(nside, pixel_indices=None)) * 1.54,
        'temp': np.ones(nside2npix(nside, pixel_indices=None)) * 20.,
        'add_decorrelation': False,
    }]

def d1(nside, pixel_indices=None):
    return [{
        'model': 'modified_black_body',
        'nu_0_I': 545.,
        'nu_0_P': 353.,
        'A_I': read_map(template('dust_t_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_Q': read_map(template('dust_q_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_U': read_map(template('dust_u_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'spectral_index': read_map(template('dust_beta.fits'), nside=nside, field=0, pixel_indices=pixel_indices),
        'temp': read_map(template('dust_temp.fits'), nside, field=0, pixel_indices=pixel_indices),
        'add_decorrelation': False,
    }]

def d2(nside, pixel_indices=None):
    return [{
        'model': 'modified_black_body',
        'nu_0_I': 545.,
        'nu_0_P': 353.,
        'A_I': read_map(template('dust_t_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_Q': read_map(template('dust_q_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_U': read_map(template('dust_u_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'spectral_index': read_map(template('beta_mean1p59_std0p2.fits'), nside, field=0, pixel_indices=pixel_indices),
        'temp': read_map(template('dust_temp.fits'), nside, field=0, pixel_indices=pixel_indices),
        'add_decorrelation': False,
    }]

def d3(nside, pixel_indices=None):
    return [{
        'model': 'modified_black_body',
        'nu_0_I': 545.,
        'nu_0_P': 353.,
        'A_I': read_map(template('dust_t_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_Q': read_map(template('dust_q_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_U': read_map(template('dust_u_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'spectral_index': read_map(template('beta_mean1p59_std0p3.fits'), nside, field=0, pixel_indices=pixel_indices),
        'temp': read_map(template('dust_temp.fits'), nside, field=0, pixel_indices=pixel_indices),
        'add_decorrelation': False,
    }]

def d4(nside, pixel_indices=None):
    return [{
        'model': 'modified_black_body',
        'nu_0_I': 545.,
        'nu_0_P': 353.,
        'A_I': read_map(template('dust2comp_I1_ns512_545.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_Q': read_map(template('dust2comp_Q1_ns512_353.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_U': read_map(template('dust2comp_U1_ns512_353.fits'), nside, field=0, pixel_indices=pixel_indices),
        'spectral_index': read_map(template('dust2comp_beta1_ns512.fits'), nside, field=0, pixel_indices=pixel_indices),
        'temp': read_map(template('dust2comp_temp1_ns512.fits'), nside, field=0, pixel_indices=pixel_indices),
        'add_decorrelation': False,
    }, {
        'model': 'modified_black_body',
        'nu_0_I': 545.,
        'nu_0_P': 353.,
        'A_I': read_map(template('dust2comp_I2_ns512_545.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_Q': read_map(template('dust2comp_Q2_ns512_353.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_U': read_map(template('dust2comp_U2_ns512_353.fits'), nside, field=0, pixel_indices=pixel_indices),
        'spectral_index': read_map(template('dust2comp_beta2_ns512.fits'), nside, field=0, pixel_indices=pixel_indices),
        'temp': read_map(template('dust2comp_temp2_ns512.fits'), nside, field=0, pixel_indices=pixel_indices),
        'add_decorrelation': False,
    }]

def d5(nside, pixel_indices=None):
    return [{
        'model': 'hensley_draine_2017',
        'draw_uval': True,
        'draw_uval_seed': 4632,
        'fcar': 1.,
        'f_fe': 0.,
        'nu_0_I': 545.,
        'nu_0_P': 353.,
        'A_I': read_map(template('dust_t_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_Q': read_map(template('dust_q_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_U': read_map(template('dust_u_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'add_decorrelation': False,
    }]

def d6(nside, pixel_indices=None):
    return [{
        'model': 'modified_black_body',
        'nu_0_I': 545.,
        'nu_0_P': 353.,
        'A_I': read_map(template('dust_t_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_Q': read_map(template('dust_q_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_U': read_map(template('dust_u_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'spectral_index': read_map(template('dust_beta.fits'), nside, field=0, pixel_indices=pixel_indices),
        'temp': read_map(template('dust_temp.fits'), nside, field=0, pixel_indices=pixel_indices),
        'add_decorrelation': True,
        'corr_len': 5.0
    }]

def d7(nside, pixel_indices=None):
    return [{
        'model': 'hensley_draine_2017',
        'draw_uval': True,
        'draw_uval_seed': 4632,
        'fcar': 1.,
        'f_fe': 0.44,
        'nu_0_I': 545.,
        'nu_0_P': 353.,
        'A_I': read_map(template('dust_t_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_Q': read_map(template('dust_q_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_U': read_map(template('dust_u_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'add_decorrelation' : False,
    }]

def d8(nside, pixel_indices=None):
    return [{
        'model': 'hensley_draine_2017',
        'draw_uval': False,
        'uval': 0.2,
        'draw_uval_seed': 4632,
        'fcar': 1.,
        'f_fe': 0.44,
        'nu_0_I': 545.,
        'nu_0_P': 353.,
        'A_I': read_map(template('dust_t_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_Q': read_map(template('dust_q_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_U': read_map(template('dust_u_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'add_decorrelation': False,
    }]

def s0(nside, pixel_indices=None):
    return [{
        'model': 'power_law',
        'nu_0_I': 0.408,
        'nu_0_P': 23.,
        'A_I': read_map(template('synch_t_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_Q': read_map(template('synch_q_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_U': read_map(template('synch_u_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'spectral_index': np.ones(nside2npix(nside, pixel_indices=None)) * -3,
    }]

def s1(nside, pixel_indices=None):
    return [{
        'model': 'power_law',
        'nu_0_I': 0.408,
        'nu_0_P': 23.,
        'A_I': read_map(template('synch_t_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_Q': read_map(template('synch_q_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_U': read_map(template('synch_u_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'spectral_index': read_map(template('synch_beta.fits'), nside, field=0, pixel_indices=pixel_indices),
    }]

def s2(nside, pixel_indices=None):
    return [{
        'model': 'power_law',
        'nu_0_I': 0.408,
        'nu_0_P': 23.,
        'A_I': read_map(template('synch_t_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_Q': read_map(template('synch_q_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_U': read_map(template('synch_u_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'spectral_index': read_map(template('beta_latvar.fits'), nside, field=0, pixel_indices=pixel_indices),
    }]

def s3(nside, pixel_indices=None):
    return [{
        'model': 'curved_power_law',
        'nu_0_I': 0.408,
        'nu_0_P': 23.,
        'A_I': read_map(template('synch_t_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_Q': read_map(template('synch_q_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'A_U': read_map(template('synch_u_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'spectral_index': read_map(template('synch_beta.fits'), nside, field=0, pixel_indices=pixel_indices),
        'spectral_curvature': -0.052,
        'nu_curve': 23.,
    }]

def f1(nside, pixel_indices=None):
    return [{
        'model': 'power_law',
        'nu_0_I': 30.,
        'A_I': read_map(template('ff_t_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'spectral_index': -2.14,
    }]

def c1(nside, pixel_indices=None):
    return [{
        'model': 'taylens',
        'cmb_specs': np.loadtxt(template('camb_lenspotentialCls.dat'), unpack=True),
        'delens': False,
        'delensing_ells': np.loadtxt(template('delens_ells.txt')),
        'nside': nside,
        'cmb_seed': 1111
    }]

def a1(nside, pixel_indices=None):
    return [{
        'model': 'spdust',
        'nu_0_I': 22.8,
        'nu_0_P': 22.8,
        'A_I': read_map(template('ame_t_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'nu_peak_0': 30.,
        'emissivity': np.loadtxt(template('emissivity.txt'), unpack=True),
        'nu_peak': read_map(template('ame_nu_peak_0.fits'), nside, field=0, pixel_indices=pixel_indices),
    }, {
        'model': 'spdust',
        'nu_0_I': 41.0,
        'nu_0_P': 41.0,
        'A_I': read_map(template('ame2_t_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'nu_peak_0': 30.,
        'emissivity': np.loadtxt(template('emissivity.txt'), unpack=True),
        'nu_peak': 33.35
    }]

def a2(nside, pixel_indices=None):
    return [{
        'model': 'spdust_pol',
        'nu_0_I': 22.8,
        'A_I': read_map(template('ame_t_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'nu_peak_0': 30.,
        'emissivity': np.loadtxt(template('emissivity.txt'), unpack=True),
        'nu_peak': read_map(template('ame_nu_peak_0.fits'), nside, field=0, pixel_indices=pixel_indices),
        'pol_frac': 0.02,
        'angle_q': read_map(template('dust_q_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'angle_u': read_map(template('dust_u_new.fits'), nside, field=0, pixel_indices=pixel_indices)
    }, {
        'model': 'spdust_pol',
        'nu_0_I': 41.0,
        'A_I': read_map(template('ame2_t_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'nu_peak_0': 30.,
        'emissivity': np.loadtxt(template('emissivity.txt'), unpack=True),
        'nu_peak': 33.35,
        'pol_frac': 0.02,
        'angle_q': read_map(template('dust_q_new.fits'), nside, field=0, pixel_indices=pixel_indices),
        'angle_u': read_map(template('dust_u_new.fits'), nside, field=0, pixel_indices=pixel_indices)
    }]
