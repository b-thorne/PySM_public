"""Run this test with MPI as:

    mpirun -np 4 python test_mpi.py
"""

import numpy as np
import healpy as hp

import pysm
from pysm.nominal import models

import sys

try:
    from mpi4py import MPI
except ImportError:
    print("Skipping MPI test as mpi4py is missing")
    sys.exit(0)

def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)

def build_sky_config(pysm_model, nside, pixel_indices=None, mpi_comm=None):
    """Build a PySM sky configuration dict from a model string"""

    sky_components = [
        'synchrotron',
        'dust',
        'freefree',
        'cmb',
        'ame',
    ]

    sky_config = dict()
    for component_model in pysm_model.split(','):
        full_component_name = [
            each for each in sky_components
            if each.startswith(component_model[0])][0]
        sky_config[full_component_name] = \
            models(component_model, nside=nside, pixel_indices=pixel_indices, mpi_comm=mpi_comm)
    return sky_config

def test_mpi_read():

    comm = MPI.COMM_WORLD

    assert is_power2(comm.size), "Run with a number of MPI processes which is power of 2"

    nside = 64
    npix = hp.nside2npix(nside)

    num_local_pixels = npix // comm.size

    if comm.size == 1:
        pixel_indices = None
        comm = None
    else:
        pixel_indices = np.arange(comm.rank     * num_local_pixels,
                                  (comm.rank+1) * num_local_pixels,
                                  dtype=np.int)

    pysm_model = "s3,d7,f1,c1,a2"

    sky_config = build_sky_config(pysm_model, nside, pixel_indices, comm)

    sky = pysm.Sky(sky_config, mpi_comm=comm)


    instrument_bpass = {
        'use_smoothing': False,
        'nside': nside,
        'add_noise': False,
        'use_bandpass': True,
        'channels': [(np.linspace(20, 25, 10), np.ones(10))],
        'channel_names': ['channel_1'],
        'output_units': 'uK_RJ',
        'output_directory': './',
        'output_prefix': 'test',
        'noise_seed': 1234,
        'pixel_indices': pixel_indices
    }

    instrument = pysm.Instrument(instrument_bpass)
    local_map = instrument.observe(sky, write_outputs=False)

    # Run PySM again locally on each process on the full map

    sky_config = build_sky_config(pysm_model, nside)

    sky = pysm.Sky(sky_config, mpi_comm=comm)

    instrument_bpass["pixel_indices"] = None

    instrument = pysm.Instrument(instrument_bpass)
    complete_map = instrument.observe(sky, write_outputs=False)

    if pixel_indices is None:
        pixel_indices = np.arange(npix)

    np.testing.assert_array_almost_equal(
            local_map[0],
            complete_map[0][:, :, pixel_indices])

if __name__ == "__main__":
    test_mpi_read()
