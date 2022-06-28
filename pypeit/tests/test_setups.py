"""
Module to run tests on scripts
"""
import os
import glob
import shutil

from IPython import embed

import numpy as np

import pytest
from configobj import ConfigObj

from pypeit.pypmsgs import PypeItError
from pypeit.metadata import PypeItMetaData
from pypeit.par import PypeItPar
from pypeit.par.util import parse_pypeit_file
from pypeit.scripts.setup import Setup
from pypeit.scripts.chk_for_calibs import ChkForCalibs
from pypeit.spectrographs.util import load_spectrograph
from pypeit.tests.tstutils import data_path
from pypeit import pypeit
from pypeit import pypeitsetup


def expected_file_extensions():
    return ['pypeit', 'sorted']


def test_run_setup():
    """ Test the setup script
    """
    # Remove .setup if needed
    sfiles = glob.glob('*.setups')
    for sfile in sfiles:
        os.remove(sfile)
    #
    droot = data_path('b')
    pargs = Setup.parse_args(['-r', droot, '-s', 'shane_kast_blue', '-c=all',
                              '--extension=fits.gz', '--output_path={:s}'.format(data_path(''))])
    Setup.main(pargs)

    #setup_file = glob.glob(data_path('setup_files/shane_kast_blue*.setups'))[0]
    ## Load
    #with open(setup_file, 'r') as infile:
    #    setup_dict = yaml.load(infile)
    ## Test
    #assert '01' in setup_dict['A'].keys()
    #assert setup_dict['A']['--']['disperser']['name'] == '600/4310'
    # Failures
    pargs2 = Setup.parse_args(['-r', droot, '-s', 'shane_kast_blu', '-c=all',
                               '--extension=fits.gz', '--output_path={:s}'.format(data_path(''))])
    with pytest.raises(ValueError):
        Setup.main(pargs2)
    
    # Cleanup
    shutil.rmtree(data_path('setup_files'))


def test_setup_made_pypeit_file():
    """ Test the .pypeit file(s) made by pypeit_setup

    This test depends on the one above
    """
    pypeit_file = data_path('shane_kast_blue_A/shane_kast_blue_A.pypeit')
    cfg_lines, data_files, frametype, usrdata, setups, setup_dict = parse_pypeit_file(pypeit_file)
    # Test
    assert len(data_files) == 8
    assert sorted(frametype['b1.fits.gz'].split(',')) == ['arc', 'tilt']
    assert setups[0] == 'A'

    # Cleanup
    shutil.rmtree(data_path('shane_kast_blue_A'))
