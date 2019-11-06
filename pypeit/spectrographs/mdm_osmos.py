""" Module for Shane/Kast specific codes
"""
import numpy as np

from astropy.io import fits
from pkg_resources import resource_filename

from pypeit import msgs
from pypeit import telescopes
from pypeit.core import framematch
from pypeit.par import pypeitpar
from pypeit.spectrographs import spectrograph
from pypeit.core import parse

from IPython import embed


class MDMOSMOSMDM4KSpectrograph(spectrograph.Spectrograph):
    """
    Child to handle WHT/ISIS blue specific code
    """
    def __init__(self):
        # Get it started
        super(MDMOSMOSMDM4KSpectrograph, self).__init__()
        self.spectrograph = 'mdm_osmos_mdm4k'
        self.telescope = telescopes.MDMTelescopePar()
        #self.camera = 'ISISb'
        self.detector = [
                # Detector 1
                pypeitpar.DetectorPar(
                            dataext         = 1,
                            specaxis        = 1,
                            specflip        = True,
                            xgap            = 0.,
                            ygap            = 0.,
                            ysize           = 1.,
                            platescale      = 0.273,
                            darkcurr        = 0.0,
                            saturation      = 65535.,
                            nonlinear       = 0.86,
                            numamplifiers   = 2,
                            gain            = [2.2, 2.4],
                            ronoise         = [5.0, 5.0],
                            datasec         = ['[:508,33:]', '[508:,33:]'],
                            oscansec        = [':508, 1:32]', '[508:, 1:32]'],
                            suffix          = ''
                            )]
        self.numhead = 1
        # Uses default timeunit
        # Uses default primary_hdrext
        # self.sky_file = ?

    def default_pypeit_par(self):
        """
        Set default parameters for Keck LRISb reductions.
        """
        par = pypeitpar.PypeItPar()
        par['rdx']['spectrograph'] = 'mdm_osmos_mdm4k'

        # Ignore PCA
        par['calibrations']['slitedges']['sync_predict'] = 'nearest'

        # Set pixel flat combination method
        par['calibrations']['pixelflatframe']['process']['combine'] = 'median'
        par['calibrations']['pixelflatframe']['process']['sig_lohi'] = [10.,10.]
        # Change the wavelength calibration method
        par['calibrations']['wavelengths']['method'] = 'holy-grail'
        par['calibrations']['wavelengths']['lamps'] = ['ArI', 'XeI']
        par['calibrations']['wavelengths']['nonlinear_counts'] = self.detector[0]['nonlinear'] * self.detector[0]['saturation']
        par['calibrations']['wavelengths']['sigdetect'] = 10.0
        #par['calibrations']['wavelengths']['wv_cen'] = 4859.0
        #par['calibrations']['wavelengths']['disp'] = 0.2
        # Scienceimage default parameters
        par['scienceimage'] = pypeitpar.ScienceImagePar()
        # Do not flux calibrate
        par['fluxcalib'] = None
        # Always correct for flexure, starting with default parameters
        par['flexure'] = pypeitpar.FlexurePar()
        # Set the default exposure time ranges for the frame typing
        par['calibrations']['biasframe']['exprng'] = [None, 1]
        par['calibrations']['darkframe']['exprng'] = [999999, None]     # No dark frames
        par['calibrations']['pinholeframe']['exprng'] = [999999, None]  # No pinhole frames
        par['calibrations']['arcframe']['exprng'] = [None, 120]
        par['calibrations']['standardframe']['exprng'] = [None, 120]
        par['scienceframe']['exprng'] = [90, None]

        return par

    def config_specific_par(self, scifile, inp_par=None):
        """
        Modify the PypeIt parameters to hard-wired values used for
        specific instrument configurations.

        .. todo::
            Document the changes made!

        Args:
            scifile (str):
                File to use when determining the configuration and how
                to adjust the input parameters.
            inp_par (:class:`pypeit.par.parset.ParSet`, optional):
                Parameter set used for the full run of PypeIt.  If None,
                use :func:`default_pypeit_par`.

        Returns:
            :class:`pypeit.par.parset.ParSet`: The PypeIt paramter set
            adjusted for configuration specific parameter values.
        """
        par = self.default_pypeit_par() if inp_par is None else inp_par

        # Wavelength calibrations
        #if self.get_meta_value(scifile, 'dispname') == 'R1200B':
        #    par['calibrations']['wavelengths']['reid_arxiv'] = 'wht_isis_blue_1200_4800.fits'

        # Return
        return par

    def init_meta(self):
        """
        Generate the meta data dict
        Note that the children can add to this

        Returns:
            self.meta: dict (generated in place)

        """
        meta = {}
        # Required (core)
        meta['ra'] = dict(ext=0, card='RA')
        meta['dec'] = dict(ext=0, card='DEC')
        meta['target'] = dict(ext=0, card='OBJECT')
        meta['decker'] = dict(ext=0, card='SLITID')
        meta['binning'] = dict(card=None, compound=True)

        meta['mjd'] = dict(ext=0, card='MJD')
        meta['exptime'] = dict(ext=0, card='EXPTIME')
        meta['airmass'] = dict(ext=0, card='SECZ')
        # Extras for config and frametyping
        meta['dispname'] = dict(ext=0, card='DISPID')
        #meta['dichroic'] = dict(ext=0, card='ISIDICHR')
        #meta['dispangle'] = dict(ext=0, card='CENWAVE', rtol=1e-3)
        #meta['slitwid'] = dict(ext=0, card='ISISLITW')
        meta['idname'] = dict(ext=0, card='IMAGETYP')
        # Lamps
        meta['lampstat01'] = dict(ext=0, card='LAMPS')

        # Ingest
        self.meta = meta

    def compound_meta(self, headarr, meta_key):
        if meta_key == 'binning':
            binspatial = headarr[0]['CCDXBIN']
            binspec = headarr[0]['CCDYBIN']
            return parse.binning2string(binspec, binspatial)
        else:
            msgs.error("Not ready for this compound meta")

    def configuration_keys(self):
        """
        Return the metadata keys that defines a unique instrument
        configuration.

        This list is used by :class:`pypeit.metadata.PypeItMetaData` to
        identify the unique configurations among the list of frames read
        for a given reduction.

        Returns:
            list: List of keywords of data pulled from meta
        """
        return ['dispname', 'decker', 'binning']

    def pypeit_file_keys(self):
        pypeit_keys = super(MDMOSMOSMDM4KSpectrograph, self).pypeit_file_keys()
        pypeit_keys += ['slitwid']
        return pypeit_keys

    def check_frame_type(self, ftype, fitstbl, exprng=None):
        """
        Check for frames of the provided type.
        """
        good_exp = framematch.check_frame_exptime(fitstbl['exptime'], exprng)
        if ftype in ['science', 'standard']:
            return good_exp & (fitstbl['idname'] == 'OBJECT')
        if ftype == 'bias':
            return good_exp & (fitstbl['idname'] == 'zero')
        if ftype in ['pixelflat', 'trace']:
            return good_exp & (fitstbl['lampstat01'] == 'Flat') & (fitstbl['idname'] == 'FLAT')
        if ftype in ['pinhole', 'dark']:
            # Don't type pinhole or dark frames
            return np.zeros(len(fitstbl), dtype=bool)
        if ftype in ['arc', 'tilt']:
            #embed(header='192 of mdm')
            return good_exp & np.array([ilamp in ['Ar','Xe'] for ilamp in fitstbl['lampstat01']]) & (
                    np.array([iid in ['Ar', 'Xe'] for iid in fitstbl['idname']]))
        msgs.warn('Cannot determine if frames are of type {0}.'.format(ftype))
        return np.zeros(len(fitstbl), dtype=bool)

    def bpm(self, filename=None, det=None, shape=None, msbias=None, **null_kwargs):
        """ Generate a BPM

        Parameters
        ----------
        shape : tuple, REQUIRED
        filename : str, REQUIRED for binning
        det : int, REQUIRED
        **null_kwargs:
           Captured and never used

        Returns
        -------
        badpix : ndarray

        """
        # Get the empty bpm: force is always True
        self.bpm_img = self.empty_bpm(filename, det=det, shape=shape)

