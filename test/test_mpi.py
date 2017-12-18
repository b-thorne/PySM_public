"""Run this test with MPI as:

    mpirun -np 4 python test_mpi.py
"""
import numpy as np
import healpy as hp
import pysm
from mpi4py import MPI

def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)

comm = MPI.COMM_WORLD

assert is_power2(comm.size), "Test with a number of MPI processes which is power of 2"

from pysm.nominal import models

nside = 64
npix = hp.nside2npix(nside)

num_local_pixels = npix // comm.size

if comm.size == 1:
    pixel_indices = None
    comm = None
else:
    pixel_indices = np.arange(comm.rank*num_local_pixels, (comm.rank+1)*num_local_pixels,dtype=np.int)

sky_config = {
    'synchrotron' : models("s3", nside, pixel_indices=pixel_indices, mpi_comm=comm),
    'dust' : models("d7", nside, pixel_indices=pixel_indices, mpi_comm=comm),
    'freefree' : models("f1", nside, pixel_indices=pixel_indices, mpi_comm=comm),
    'cmb' : models("c1", nside, pixel_indices=pixel_indices, mpi_comm=comm),
    'ame' : models("a2", nside, pixel_indices=pixel_indices, mpi_comm=comm),
}


sky = pysm.Sky(sky_config, mpi_comm=comm)


instrument_bpass = {
    'use_smoothing' : False,
    'nside' : nside,
    'add_noise' : False,
    'use_bandpass' : True,
    'channels' : [(np.linspace(20, 25, 10), np.ones(10))],
    'channel_names' : ['channel_1'],
    'output_units' : 'uK_RJ',
    'output_directory' : './',
    'output_prefix' : 'test',
    'noise_seed' : 1234,
    'pixel_indices' : pixel_indices
}

instrument = pysm.Instrument(instrument_bpass)
local_map = instrument.observe(sky, write_outputs=False)

# Run PySM again locally on each process on the full map

comm = None

sky_config = {
    'synchrotron' : models("s3", nside, pixel_indices=None, mpi_comm=comm),
    'dust' : models("d7", nside, pixel_indices=None, mpi_comm=comm),
    'freefree' : models("f1", nside, pixel_indices=None, mpi_comm=comm),
    'cmb' : models("c1", nside, pixel_indices=None, mpi_comm=comm),
    'ame' : models("a2", nside, pixel_indices=None, mpi_comm=comm),
}


sky = pysm.Sky(sky_config, mpi_comm=comm)

instrument_bpass["pixel_indices"] = None

instrument = pysm.Instrument(instrument_bpass)
complete_map = instrument.observe(sky, write_outputs=False)

if pixel_indices is None:
    pixel_indices = np.arange(npix)

np.testing.assert_array_almost_equal(
        local_map[0],
        complete_map[0][:, :, pixel_indices])
