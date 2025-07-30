"""
Module for MMT/BINOSPEC specific methods.

.. include:: ../include/links.rst
"""
import numpy as np
import os

from pypeit import msgs
from pypeit import telescopes
from pypeit import utils
from pypeit import io
from pypeit.core import framematch
from pypeit.spectrographs import spectrograph
from pypeit.core import parse
from pypeit.images import detector_container

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units
from pypeit.spectrographs.slitmask import SlitMask

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython import embed

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np




class MMTBINOSPECSpectrograph(spectrograph.Spectrograph):
    """
    Child to handle MMT/BINOSPEC specific code
    """
    ndet = 2
    name = 'mmt_binospec'
    telescope = telescopes.MMTTelescopePar()
    camera = 'BINOSPEC'
    url = 'https://lweb.cfa.harvard.edu/mmti/binospec.html'
    header_name = 'Binospec'
    supported = True

    def get_detector_par(self, det, hdu=None):
        """
        Return metadata for the selected detector.

        Args:
            det (:obj:`int`):
                1-indexed detector number.
            hdu (`astropy.io.fits.HDUList`_, optional):
                The open fits file with the raw image of interest.  If not
                provided, frame-dependent parameters are set to a default.

        Returns:
            :class:`~pypeit.images.detector_container.DetectorContainer`:
            Object with the detector metadata.
        """
        # Binning
        binning = '1,1' if hdu is None else self.get_meta_value(self.get_headarr(hdu), 'binning')

        # Detector 1
        detector_dict1 = dict(
                            binning         = binning,
                            det             = 1,
                            dataext         = 1,
                            specaxis        = 0,
                            specflip        = False,
                            spatflip        = False,
                            xgap            = 0.,
                            ygap            = 0.,
                            ysize           = 1.,
                            platescale      = 0.24,
                            darkcurr        = 3.6,  #e-/pixel/hour  (=0.001 e-/pixel/s)  --  pulled from the ETC
                            saturation      = 65535.,
                            nonlinear       = 0.95,  #ToDO: To Be update
                            mincounts       = -1e10,
                            numamplifiers   = 4,
                            gain            = np.atleast_1d([1.085,1.046,1.042,0.975]),
                            ronoise         = np.atleast_1d([3.2,3.2,3.2,3.2]),
                            )
        # Detector 2
        detector_dict2 = detector_dict1.copy()
        detector_dict2.update(dict(
            det=2,
            dataext=2,
            gain=np.atleast_1d([1.028,1.115,1.047,1.045]), #ToDo: FW measures 1.115 for amp2 but 1.163 in IDL pipeline
            ronoise=np.atleast_1d([3.6,3.6,3.6,3.6])
        ))

        # Instantiate
        detector_dicts = [detector_dict1, detector_dict2]
        return detector_container.DetectorContainer(**detector_dicts[det-1])

    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        That is, this associates the PypeIt-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        self.meta = {}
        # Required (core)
        self.meta['ra'] = dict(ext=1, card='RA')
        self.meta['dec'] = dict(ext=1, card='DEC')
        self.meta['target'] = dict(ext=1, card='OBJECT')
        self.meta['decker'] = dict(ext=1, card='MASK')

        self.meta['dichroic'] = dict(ext=1, card=None, default='default')
        self.meta['binning'] = dict(ext=1, card='CCDSUM', compound=True)


        self.meta['mjd'] = dict(ext=1, card='MJD')
        self.meta['exptime'] = dict(ext=1, card='EXPTIME')
        self.meta['airmass'] = dict(ext=1, card='AIRMASS')
        # Extras for config and frametyping
        self.meta['dispname'] = dict(ext=1, card='DISPERS1')
        self.meta['idname'] = dict(ext=1, card='IMAGETYP')

        # used for arclamp
        self.meta['lampstat01'] = dict(ext=1, card='HENEAR')
        # used for flatlamp, SCRN is actually telescope status
        self.meta['lampstat02'] = dict(ext=1, card='SCRN')
        self.meta['instrument'] = dict(ext=1, card='INSTRUME')

    def compound_meta(self, headarr, meta_key):
        """
        Methods to generate metadata requiring interpretation of the header
        data, instead of simply reading the value of a header card.

        Args:
            headarr (:obj:`list`):
                List of `astropy.io.fits.Header`_ objects.
            meta_key (:obj:`str`):
                Metadata keyword to construct.

        Returns:
            object: Metadata value read from the header(s).
        """
        if meta_key == 'binning':
            binspatial, binspec = parse.parse_binning(headarr[1]['CCDSUM'])
            binning = parse.binning2string(binspec, binspatial)
            return binning

    def configuration_keys(self):
        """
        Return the metadata keys that define a unique instrument
        configuration.

        This list is used by :class:`~pypeit.metadata.PypeItMetaData` to
        identify the unique configurations among the list of frames read
        for a given reduction.

        Returns:
            :obj:`list`: List of keywords of data pulled from file headers
            and used to constuct the :class:`~pypeit.metadata.PypeItMetaData`
            object.
        """
        return ['dispname']

    def raw_header_cards(self):
        """
        Return additional raw header cards to be propagated in
        downstream output files for configuration identification.

        The list of raw data FITS keywords should be those used to populate
        the :meth:`~pypeit.spectrographs.spectrograph.Spectrograph.configuration_keys`
        or are used in :meth:`~pypeit.spectrographs.spectrograph.Spectrograph.config_specific_par`
        for a particular spectrograph, if different from the name of the
        PypeIt metadata keyword.

        This list is used by :meth:`~pypeit.spectrographs.spectrograph.Spectrograph.subheader_for_spec`
        to include additional FITS keywords in downstream output files.

        Returns:
            :obj:`list`: List of keywords from the raw data files that should
            be propagated in output files.
        """
        return ['DISPERS1']

    @classmethod
    def default_pypeit_par(cls):
        """
        Return the default parameters to use for this instrument.

        Returns:
            :class:`~pypeit.par.pypeitpar.PypeItPar`: Parameters required by
            all of PypeIt methods.
        """
        par = super().default_pypeit_par()

        # Wavelengths
        # 1D wavelength solution
        par['calibrations']['wavelengths']['rms_thresh_frac_fwhm'] = 0.125
        par['calibrations']['wavelengths']['sigdetect'] = 5.
        par['calibrations']['wavelengths']['fwhm']= 4.0
        par['calibrations']['wavelengths']['lamps'] = ['ArI', 'ArII']
        #par['calibrations']['wavelengths']['nonlinear_counts'] = self.detector[0]['nonlinear'] * self.detector[0]['saturation']
        par['calibrations']['wavelengths']['method'] = 'full_template'
        par['calibrations']['wavelengths']['lamps'] = ['HeI', 'NeI', 'ArI', 'ArII']

        # Tilt and slit parameters
        par['calibrations']['tilts']['tracethresh'] =  10.0
        par['calibrations']['tilts']['spat_order'] = 6
        par['calibrations']['tilts']['spec_order'] = 6
        par['calibrations']['slitedges']['sync_predict'] = 'nearest'

        # Processing steps
        turn_off = dict(use_biasimage=False, use_darkimage=False)
        par.reset_all_processimages_par(**turn_off)

        # Extraction
        par['reduce']['skysub']['bspline_spacing'] = 0.8
        par['reduce']['extraction']['sn_gauss'] = 4.0
        ## Do not perform global sky subtraction for standard stars
        par['reduce']['skysub']['global_sky_std']  = False

        # Adjust temporarily to skip over bug in J2232p2930
        par['flexure']['spec_method'] = 'slitcen'
        #par['flexure']['spec_method'] = 'boxcar'

        # cosmic ray rejection parameters for science frames
        par['scienceframe']['process']['sigclip'] = 5.0
        par['scienceframe']['process']['objlim'] = 2.0

        # Set the default exposure time ranges for the frame typing
        par['calibrations']['standardframe']['exprng'] = [None, 100]
        par['calibrations']['arcframe']['exprng'] = [20, None]
        par['calibrations']['darkframe']['exprng'] = [20, None]
        par['scienceframe']['exprng'] = [20, None]

        # Sensitivity function parameters
        par['sensfunc']['polyorder'] = 7
        par['sensfunc']['IR']['telgridfile'] = 'TellPCA_3000_26000_R10000.fits'

        return par

    def config_specific_par(self, scifile, inp_par=None):
        """
        Modify the PypeIt parameters to hard-wired values used for
        specific instrument configurations.

        Args:
            scifile (:obj:`str`):
                File to use when determining the configuration and how
                to adjust the input parameters.
            inp_par (:class:`~pypeit.par.parset.ParSet`, optional):
                Parameter set used for the full run of PypeIt.  If None,
                use :func:`default_pypeit_par`.

        Returns:
            :class:`~pypeit.par.parset.ParSet`: The PypeIt parameter set
            adjusted for configuration specific parameter values.
        """
        par = super().config_specific_par(scifile, inp_par=inp_par)

        headarr = self.get_headarr(scifile)

        if 'spec2d' in scifile:
            decker = headarr[0]['DECKER']
            grating = headarr[0]['DISPNAME']
        else:
            decker = headarr[1]['MASK']
            grating = headarr[1]['DISPERS1']


        if grating == 'x270':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'mmt_binospec_270.fits'

        if grating == 'x600':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'mmt_binospec_600.fits'

        if grating == 'x1000':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'mmt_binospec_1000.fits'


        if 'Longslit' not in decker:

            # Turn on the use of mask design
            par['calibrations']['slitedges']['use_maskdesign'] = True
            # Since we use the slitmask info to find the alignment boxes, I don't need `minimum_slit_length_sci`
            par['calibrations']['slitedges']['minimum_slit_length_sci'] = None
            # Sometime the added missing slits at the edge of the detector are to small to be useful.
            par['calibrations']['slitedges']['minimum_slit_length'] = 3.
            # Since we use the slitmask info to add and remove traces, 'minimum_slit_gap' may undo the matching effort.
            par['calibrations']['slitedges']['minimum_slit_gap'] = 0.
            # Lower edge_thresh works better
            par['calibrations']['slitedges']['edge_thresh'] = 10.
            # use stars in alignment boxes to compute the slitmask offset (this works the best)
            par['reduce']['slitmask']['use_alignbox'] = True
            # Assign RA, DEC, OBJNAME to detected objects
            par['reduce']['slitmask']['assign_obj'] = True
            # force extraction of undetected objects
            par['reduce']['slitmask']['extract_missing_objs'] = True
            # Adjust sky subtraction parameters

            # lower tilts spat_order and higher spec_order for multislits (i.e., generally not very long slits)
            par['calibrations']['tilts']['spat_order'] = 2  # Default: 3
            par['calibrations']['tilts']['spec_order'] = 5  # Default: 4
            # pca
            par['calibrations']['slitedges']['sync_predict'] = 'auto'

            par['coadd2d']['offsets'] = 'maskdef_offsets'



        print("Enter embed in config_specific_par()")

        return par

    def update_edgetracepar(self, par):
        """
        This method is used in :func:`pypeit.edgetrace.EdgeTraceSet.maskdesign_matching`
        to update EdgeTraceSet parameters when the slitmask design matching is not feasible
        because too few slits are present in the detector.

        Args:
            par (:class:`pypeit.par.pypeitpar.EdgeTracePar`):
                The parameters used to guide slit tracing.

        Returns:
            :class:`pypeit.par.pypeitpar.EdgeTracePar`
            The modified parameters used to guide slit tracing.
        """

        par['minimum_slit_gap'] = 0.25
        par['minimum_slit_length_sci'] = 4.5
        return par

    def bpm(self, filename, det, shape=None, msbias=None):
        """
        Generate a default bad-pixel mask.

        Even though they are both optional, either the precise shape for
        the image (``shape``) or an example file that can be read to get
        the shape (``filename`` using :func:`get_image_shape`) *must* be
        provided.

        Args:
            filename (:obj:`str` or None):
                An example file to use to get the image shape.
            det (:obj:`int`):
                1-indexed detector number to use when getting the image
                shape from the example file.
            shape (tuple, optional):
                Processed image shape
                Required if filename is None
                Ignored if filename is not None
            msbias (`numpy.ndarray`_, optional):
                Processed bias frame used to identify bad pixels

        Returns:
            `numpy.ndarray`_: An integer array with a masked value set
            to 1 and an unmasked value set to 0.  All values are set to
            0.
        """
        # Call the base-class method to generate the empty bpm
        bpm_img = super().bpm(filename, det, shape=shape, msbias=msbias)

        if det == 1:
            msgs.info("Using hard-coded BPM for det=1 on BINOSPEC")

            # TODO: Fix this
            # Get the binning
            hdu = io.fits_open(filename)
            binning = hdu[1].header['CCDSUM']
            hdu.close()

            # Apply the mask
            xbin, ybin = int(binning.split(' ')[0]), int(binning.split(' ')[1])
            bpm_img[2447 // xbin, 2056 // ybin:4112 // ybin] = 1
            bpm_img[2111 // xbin, 2056 // ybin:4112 // ybin] = 1

        elif det == 2:
            msgs.info("Using hard-coded BPM for det=2 on BINOSPEC")

            # Get the binning
            hdu = io.fits_open(filename)
            binning = hdu[5].header['CCDSUM']
            hdu.close()

            # Apply the mask
            xbin, ybin = int(binning.split(' ')[0]), int(binning.split(' ')[1])
            #ToDo: Need to double check the  BPM for detector 2
            ## Identified by FW from flat observations
            bpm_img[3336 // xbin, 0:2056 // ybin] = 1
            bpm_img[3337 // xbin, 0:2056 // ybin] = 1
            bpm_img[4056 // xbin, 0:2056 // ybin] = 1
            bpm_img[3011 // xbin, 2057 // ybin:4112 // ybin] = 1
            ## Got from IDL pipeline
            #bpm_img[2378 // xbin, 0:2056 // ybin] = 1
            #bpm_img[2096 // xbin, 2057 // ybin:4112 // ybin] = 1
            #bpm_img[1084 // xbin, 0:2056 // ybin] = 1

        return bpm_img

    def check_frame_type(self, ftype, fitstbl, exprng=None):
        """
        Check for frames of the provided type.

        Args:
            ftype (:obj:`str`):
                Type of frame to check. Must be a valid frame type; see
                frame-type :ref:`frame_type_defs`.
            fitstbl (`astropy.table.Table`_):
                The table with the metadata for one or more frames to check.
            exprng (:obj:`list`, optional):
                Range in the allowed exposure time for a frame of type
                ``ftype``. See
                :func:`pypeit.core.framematch.check_frame_exptime`.

        Returns:
            `numpy.ndarray`_: Boolean array with the flags selecting the
            exposures in ``fitstbl`` that are ``ftype`` type frames.
        """
        good_exp = framematch.check_frame_exptime(fitstbl['exptime'], exprng)
        if ftype == 'science':
            return good_exp & (fitstbl['lampstat01'] == 'off') & (fitstbl['lampstat02'] == 'stowed') & (fitstbl['exptime'] > 100.0)
        if ftype == 'standard':
            return good_exp & (fitstbl['lampstat01'] == 'off') & (fitstbl['lampstat02'] == 'stowed') & (fitstbl['exptime'] <= 100.0)
        if ftype in ['arc', 'tilt']:
            return good_exp & (fitstbl['lampstat01'] == 'on')
        if ftype in ['pixelflat', 'trace', 'illumflat']:
            return good_exp & (fitstbl['lampstat01'] == 'off') & (fitstbl['lampstat02'] == 'deployed')

        msgs.warn('Cannot determine if frames are of type {0}.'.format(ftype))
        return np.zeros(len(fitstbl), dtype=bool)

    def get_slitmask(self, filename, ccdnum=None):
        """
        Parse Binospec slitmask file and construct a SlitMask object with target and slit metadata.

        This function reads a FITS file describing a multi-slit mask for a specified Binospec
        detector, extracts geometric and target information, computes relevant positional offsets,
        and constructs a `SlitMask` object used for downstream processing and plotting.

        Parameters
        ----------
        filename : :obj:`str`
            Path to the slitmask FITS file.
        det : :obj:`int`
            Detector number (1 or 2).

        Returns
        -------
        slitmask : :class:`SlitMask`
            An object containing geometry, target positions, and metadata for each slit, defined
            in PypeIt/pypeit/spectrographs/slitmask.py

        Notes
        -----
        - Target-slit alignment is characterized via distances from slit edges.
        - Slit corners and on-sky positions are stored for each target.
        """

        if ccdnum is None:
            raise ValueError("A valid detector number must be provided.")

        if filename is None:
            raise ValueError("A valid slitmask filename must be provided.")

        # Open the FITS file
        hdu = fits.open(filename)

        # Select appropriate extension for detector 1 or 2
        if ccdnum == 1:
            mask_fits = hdu[9].data[0]
        elif ccdnum == 2:
            mask_fits = hdu[10].data[0]
        else:
            raise ValueError("Not a valid detector number. Try 1 or 2.")


        # Identify TARGET objects and number of slits
        target_type = mask_fits['TARGET_TYPE']
        targ = np.where(target_type == 'TARGET')[0]
        numslits = mask_fits['NTARGETS']

        # Extract slit centers and target positions in arcseconds
        slit_mean_x = np.mean(mask_fits['POLY_X'], axis=0)[targ]
        slit_mean_y = np.mean(mask_fits['POLY_Y'], axis=0)[targ]
        x_targ = mask_fits['SLITX'][targ]
        y_targ = mask_fits['SLITY'][targ]

        # Compute offsets from slit centers
        delta_x = x_targ - slit_mean_x
        delta_y = y_targ - slit_mean_y
        offsets = np.sqrt(delta_x ** 2 + delta_y ** 2)


        slit_length_half = mask_fits['SLIT_LENGTH'][targ] / 2.

        # Calculate projected distances to top/bottom edges of slits
        topdist = np.round(slit_length_half - delta_y, 3)
        botdist = np.round(slit_length_half + delta_y, 3)

        # Extract basic object metadata
        obj_ra = mask_fits['RA'][targ]
        obj_dec = mask_fits['DEC'][targ]
        objname = mask_fits['TARGET_NAME'][targ]
        objid = mask_fits['TARGET_ID'][targ]

        # Assemble object array: [slit_id, id, ra, dec, name, mag, mag_band, top, bot]
        objects = np.array([np.array(mask_fits['SLIT_ID'][targ], dtype=int),
                            objid,
                            obj_ra,
                            obj_dec,
                            objname,
                            np.array(mask_fits['MAG'][targ], dtype=float),
                            ['None'] * mask_fits['SLIT_ID'][targ].size,
                            topdist,
                            botdist
                            ]).T


        # Slitmask pointing coordinates (mask center RA/Dec)
        mask_coord = SkyCoord(mask_fits['CENTERRA'], mask_fits['CENTERDEC'],
                              unit=('hourangle', 'deg'))

        # Position angle corresponding to detector +x axis (spatial direction)
        posx_pa = hdu[1].header['POSANG'] - 180
        if posx_pa < 0:
            posx_pa += 360.

        # Slit position angles and sign of offset (accounting for up/down location)
        slit_pas = posx_pa + np.zeros(numslits)
        off_signs = np.ones_like(slit_pas)
        negy = delta_y < 0.
        off_signs[negy] = -1.


        # Compute slit center RA/Dec via spherical offset from target position
        obj_coord = SkyCoord(ra=obj_ra, dec=obj_dec, unit='deg')
        offsets = np.sqrt(delta_x ** 2 + delta_y ** 2)

        slit_ra, slit_dec = [], []
        for offset, coord, slit_pa, off_sign in zip(offsets, obj_coord, slit_pas, off_signs):
            slit_coord = coord.directional_offset_by(
                slit_pa * units.deg, off_sign * offset * units.arcsec)
            slit_ra.append(slit_coord.ra.deg)
            slit_dec.append(slit_coord.dec.deg)

        # Extract slit corner coordinates
        poly_x_T = np.squeeze(mask_fits['POLY_X'].T)[targ]
        poly_y_T = np.squeeze(mask_fits['POLY_Y'].T)[targ]

        corners = np.stack((poly_x_T, poly_y_T), axis=-1)
        corners = corners[:, [3, 0, 1, 2], :]



        # Construct and return the slitmask object
        self.slitmask = SlitMask(
            np.array(corners),
            slitid=np.array(mask_fits['SLIT_ID'][targ], dtype=int),
            onsky=np.array([
                np.array(slit_ra), np.array(slit_dec),
                np.array(mask_fits['SLIT_LENGTH'][targ], dtype=float),
                np.array(mask_fits['SLIT_WIDTH'][targ], dtype=float),
                slit_pas]).T,
            objects=objects,
            mask_radec=(mask_coord.ra.deg, mask_coord.dec.deg),
            posx_pa=posx_pa
        )

        return self.slitmask


    def bino_get_slit_region(self, filename, ccdnum=None, Nx=4096, Ny=4112, pady=0):
        """
        Compute the pixel-space rectangular regions for each slit in a Binospec mask.

        This function reads the slitmask design from a FITS file (or an already-loaded
        `SlitMask` object), converts slit and object positions from mask coordinates to
        pixel coordinates, and determines the x/y pixel boundaries for each slit on the
        detector. It returns these boundaries along with the updated slitmask object.

        Parameters
        ----------
        filename : :obj:`str`
            Path to the slitmask FITS file. Must be provided unless the slitmask
            is already loaded via `self.get_slitmask`.
        ccdnum : :obj:`int`, optional
            Detector number (1 or 2). Must be specified.
        Nx : :obj:`int`, optional
            Detector size in the x-direction (default: 4096 pixels).
        Ny : :obj:`int`, optional
            Detector size in the y-direction (default: 4112 pixels).
        pady : :obj:`float`, optional
            Additional padding (in pixels) applied to the slit boundaries (default: 0).

        Returns
        -------
        region : :obj:`list`
            A list containing:
            - slit_x_range : array of x-boundaries for each slit [Nslits, 2]
            - slit_y_range : array of y-boundaries for each slit [Nslits, 2]
            - x_slitobj_pix : array of x pixel positions for slit objects
            - y_slitobj_pix : array of y pixel positions for slit objects
        slitmask : :class:`SlitMask`
            The updated `SlitMask` object containing slit geometry and metadata.

        Notes
        -----
        - Converts mask coordinates to pixel coordinates using the appropriate scale factor.
        - Handles detector 2 by reversing slit order and applying a vertical flip.
        - Slit boundaries are clipped to remain within detector dimensions.
        """

        if ccdnum is None:
            raise ValueError("A valid detector number must be provided.")

        # Load slitmask information if a file is provided
        if filename is not None:
            self.get_slitmask(filename, ccdnum)
        else:
            raise ValueError("The name of a science file should be provided")

        if self.slitmask is None:
            raise ValueError("Unable to read slitmask design info. Provide a file.")

        # Open FITS file and read mask data for the correct detector
        hdu = fits.open(filename)
        mask_fits = hdu[9].data[0] if ccdnum == 1 else hdu[10].data[0]
        numslits = len(self.slitmask.slitid)

        # Initialize arrays to hold slit x/y boundaries
        res_x = np.zeros((2, numslits))
        res_y = np.zeros((2, numslits))

        # Extract target distances from slit edges and slit widths
        topdist = np.array(self.slitmask.objects[:, 7], dtype=float)
        botdist = np.array(self.slitmask.objects[:, 8], dtype=float)
        width = np.array(self.slitmask.width)

        # Extract slit center positions in mask coordinates
        x_slits = np.array(self.slitmask.center[:, 0])
        x_obj = x_slits
        y_slits = -np.array(self.slitmask.center[:, 1])

        # Extract slit corner y-coordinates (for top/bottom edges)
        y_slitsh = -np.array(self.slitmask.corners[:, 0, 1])
        y_slitsl = -np.array(self.slitmask.corners[:, 2, 1])

        # Compute object y-position relative to slit center
        y_obj = y_slits + (topdist - botdist) / 2

        # Extract slit lengths and widths (in mask coordinates)
        dx_slits = self.slitmask.length
        dy_slits = width

        # Define scale factor and detector offsets
        dy0 = -200.0
        y_scl = 24.555832 if ccdnum == 1 else 24.548194

        # Extract mask corner reference point
        mask_corners = np.array(mask_fits['MASK_CORNERS'])
        corner_x = mask_corners[0]
        corner_y = mask_corners[1]

        # Convert slit center positions to pixel coordinates
        x_slits_pix = (x_slits - corner_x) * y_scl + Nx / 2.0
        x_slitobj_pix = (x_obj - corner_x) * y_scl + Nx / 2.0
        y_slits_pix = Ny - 1 - ((y_slits - corner_y) * y_scl) + dy0
        y_slitobj_pix = Ny - 1 - ((y_obj - corner_y) * y_scl) + dy0
        y_slitsl_pix = Ny - 1 - ((y_slitsl - corner_y) * y_scl) + dy0
        y_slitsh_pix = Ny - 1 - ((y_slitsh - corner_y) * y_scl) + dy0

        # Convert slit lengths and widths to pixel units
        dx_slits_pix = dx_slits * y_scl
        dy_slits_pix = dy_slits * y_scl

        # Loop through slits to compute pixel-space rectangular boundaries
        for i in range(numslits):
            xmin = round(x_slits_pix[i] - dx_slits_pix[i] / 2.0 - pady)
            xmax = round(x_slits_pix[i] + dx_slits_pix[i] / 2.0 - 1 + pady)
            res_x[0, i] = max(0, xmin)
            res_x[1, i] = min(Ny - 1, xmax)

            ymin = round(y_slits_pix[i] - dy_slits_pix[i] / 2.0 - pady)
            ymax = round(y_slits_pix[i] + dy_slits_pix[i] / 2.0 - 1 + pady)
            res_y[0, i] = max(0, ymin)
            res_y[1, i] = min(Ny - 1, ymax)

        # Handle detector 2: reverse slit order and flip vertically
        if ccdnum == 2:
            res_y = res_y[:, ::-1]

            # Apply vertical flip relative to detector height (Ny) and offset
            res_y_flipped = np.zeros_like(res_y)
            res_y_flipped[0, :] = -1 * (res_y[1, :] - Ny - 14)
            res_y_flipped[1, :] = -1 * (res_y[0, :] - Ny - 14)
            res_y = res_y_flipped

        # Package results and return
        slit_x_range, slit_y_range = res_x.T, res_y.T
        region = [slit_x_range, slit_y_range, x_slitobj_pix, y_slitobj_pix]

        return region, self.slitmask

    def get_maskdef_slitedges(self, ccdnum=None, filename=None, debug=None,
                              trc_path=None, binning=None):
        """
        Determine the slit edges from a Binospec mask design file.

        This function reads the slitmask design for a specified detector, converts
        slit and object positions to pixel coordinates, and determines the top and
        bottom edges (y-direction boundaries) for each slit on the detector. It also
        returns the slitmask object with associated metadata.

        Parameters
        ----------
        ccdnum : :obj:`int`
            Detector number (1 or 2). Must be specified.
        filename : :obj:`str`
            Path to the slitmask FITS file containing mask design information.
        debug : :obj:`bool`, optional
            Flag for enabling debug-level output (default: None).
        trc_path : :obj:`str`, optional
            Path to trace files (not used in this function, included for compatibility).
        binning : :obj:`str`, optional
            Detector binning information (not used in this function, included for compatibility).

        Returns
        -------
        top_edges : :class:`numpy.ndarray`
            Array of y-pixel positions for the upper edge of each slit.
        bot_edges : :class:`numpy.ndarray`
            Array of y-pixel positions for the lower edge of each slit.
        sortindx : :class:`numpy.ndarray`
            Indices that sort the slits by their bottom edge position.
        slitmask : :class:`SlitMask`
            The `SlitMask` object containing slit geometry and metadata.

        Notes
        -----
        - The function relies on `bino_get_slit_region()` to compute slit pixel positions.
        - Edges are sorted by bottom edge y-coordinate to order slits spatially.
        """

        if ccdnum is None:
            raise ValueError("A valid detector number must be provided.")

        if filename is None:
            raise ValueError("A valid slitmask filename must be provided.")

        # Compute slit regions and associated slitmask using mask design info
        region, slitmask = self.bino_get_slit_region(filename, ccdnum)

        # region contains: [slit_x_range, slit_y_range, x_slitobj_pix, y_slitobj_pix]
        slit_x_range, slit_y_range, x_slitobj_pix, y_slitobj_pix = region

        # Extract lower and upper slit edges from the y-range boundaries
        bot_edges = slit_y_range[:, 0]  # Lower edge (minimum y-value)
        top_edges = slit_y_range[:, 1]  # Upper edge (maximum y-value)

        # Sort slits by their bottom edge position in ascending y-coordinate
        sortindx = np.argsort(bot_edges)

        # Return the slit edges, sorted indices, and slitmask object
        return top_edges, bot_edges, sortindx, self.slitmask


    def get_rawimage(self, raw_file, det):
        """
        Read raw images and generate a few other bits and pieces
        that are key for image processing.

        Parameters
        ----------
        raw_file : :obj:`str`
            File to read
        det : :obj:`int`
            1-indexed detector to read

        Returns
        -------
        detector_par : :class:`pypeit.images.detector_container.DetectorContainer`
            Detector metadata parameters.
        raw_img : `numpy.ndarray`_
            Raw image for this detector.
        hdu : `astropy.io.fits.HDUList`_
            Opened fits file
        exptime : :obj:`float`
            Exposure time read from the file header
        rawdatasec_img : `numpy.ndarray`_
            Data (Science) section of the detector as provided by setting the
            (1-indexed) number of the amplifier used to read each detector
            pixel. Pixels unassociated with any amplifier are set to 0.
        oscansec_img : `numpy.ndarray`_
            Overscan section of the detector as provided by setting the
            (1-indexed) number of the amplifier used to read each detector
            pixel. Pixels unassociated with any amplifier are set to 0.
        """
        fil = utils.find_single_file(f'{raw_file}*', required=True)

        # Read
        msgs.info(f'Reading BINOSPEC file: {fil}')
        hdu = io.fits_open(fil)
        head1 = hdu[1].header

        # TOdO Store these parameters in the DetectorPar.
        # Number of amplifiers
        detector_par = self.get_detector_par(det if det is not None else 1, hdu=hdu)
        numamp = detector_par['numamplifiers']

        # get the x and y binning factors...
        binning = head1['CCDSUM']
        xbin, ybin = [int(ibin) for ibin in binning.split(' ')]

        # First read over the header info to determine the size of the output array...
        datasec = head1['DATASEC']
        x1, x2, y1, y2 = np.array(parse.load_sections(datasec, fmt_iraf=False)).flatten()
        nxb = x1 - 1

        # determine the output array size...
        nx = (x2 - x1 + 1) * int(numamp/2) + nxb * int(numamp/2)
        ny = (y2 - y1 + 1) * int(numamp/2)

        #datasize = head1['DETSIZE']
        #_, nx, _, ny = np.array(parse.load_sections(datasize, fmt_iraf=False)).flatten()

        # allocate output array...
        array = np.zeros((nx, ny))
        rawdatasec_img = np.zeros_like(array, dtype=int)
        oscansec_img = np.zeros_like(array, dtype=int)

        if det == 1:  # A DETECTOR
            order = range(1, 5, 1)
        elif det == 2:  # B DETECTOR
            order = range(5, 9, 1)

        # insert extensions into calibration image...
        for kk, jj in enumerate(order):
            # grab complete extension...
            data, overscan, datasec, biassec = binospec_read_amp(hdu, jj)

            # insert components into output array...
            inx = data.shape[0]
            xs = inx * kk
            xe = xs + inx

            iny = data.shape[1]
            ys = iny * kk
            yn = ys + iny

            b1, b2, b3, b4 = np.array(parse.load_sections(biassec, fmt_iraf=False)).flatten()

            if kk == 0:
                array[b2:inx+b2,:iny] = data #*1.028
                rawdatasec_img[b2:inx+b2,:iny] = kk + 1
                array[:b2,:iny] = overscan
                oscansec_img[2:b2,:iny] = kk + 1
            elif kk == 1:
                array[b2+inx:2*inx+b2,:iny] = np.flipud(data) #* 1.115
                rawdatasec_img[b2+inx:2*inx+b2:,:iny] = kk + 1
                array[2*inx+b2:,:iny] = overscan
                oscansec_img[2*inx+b2:,:iny] = kk + 1
            elif kk == 2:
                array[b2+inx:2*inx+b2,iny:] = np.fliplr(np.flipud(data)) #* 1.047
                rawdatasec_img[b2+inx:2*inx+b2,iny:] = kk + 1
                array[2*inx+b2:, iny:] = overscan
                oscansec_img[2*inx+b2:, iny:] = kk + 1
            elif kk == 3:
                array[b2:inx+b2,iny:] = np.fliplr(data) #* 1.045
                rawdatasec_img[b2:inx+b2,iny:] = kk + 1
                array[:b2,iny:] = overscan
                oscansec_img[2:b2,iny:] = kk + 1

        # Need the exposure time
        exptime = hdu[self.meta['exptime']['ext']].header[self.meta['exptime']['card']]
        # Return, transposing array back to orient the overscan properly
        return detector_par, np.fliplr(np.flipud(array)), hdu, exptime, np.fliplr(np.flipud(rawdatasec_img)), \
               np.fliplr(np.flipud(oscansec_img))


    def plot_mask(self, filename, ccdnum=None, save_dir=None):
        """
        Plot the slit mask layout and target positions for one or both detectors.

        This function retrieves slit region data for a given Binospec mask and
        plots the rectangular slit outlines and target positions for detector 1,
        detector 2, or both. It is useful for visually validating mask design and
        target alignment.

        Parameters
        ----------
        filename : :obj:`str`
            Path to the mask design file (e.g., a JSON file containing slit definitions).
        det : :obj:`int` or :obj:`str`
            Specifies which detector(s) to plot. Accepts 1, 2, or 'both'.
        save_dir : :obj:`str`, optional
            If provided, the plot will be saved as a PNG in the given directory.

        Returns
        -------
        region_1 : :obj:`tuple`, optional
            Slit region and target position data for detector 1, if requested.
        region_2 : :obj:`tuple`, optional
            Slit region and target position data for detector 2, if requested.
        """

        if ccdnum is None:
            raise ValueError("A valid detector number must be provided: 1, 2, or 'both'")

        if filename is None:
            raise ValueError("A valid filename must be provided.")

        # Build save filename from FITS header
        hdu = fits.open(filename)
        basename = os.path.basename(filename)
        name = f"plot_mask_{hdu[1].header['MASK']}_{basename}"
        save_filename = os.path.splitext(name)[0] + ".png"

        plt.rcParams.update({"font.size": 20})

        # Load slit regions depending on the selected detector(s)
        if ccdnum == 'both':
            fig, (axA, axB) = plt.subplots(ncols=2, figsize=(16, 16))
            region_1 = self.bino_get_slit_region(filename, 1)[0]
            region_2 = self.bino_get_slit_region(filename, 2)[0]

        elif ccdnum == 1:
            fig, axA = plt.subplots(figsize=(8, 8))
            region_1 = self.bino_get_slit_region(filename, 1)[0]

        elif ccdnum == 2:
            fig, axB = plt.subplots(figsize=(8, 8))
            region_2 = self.bino_get_slit_region(filename, 2)[0]

        else:
            raise ValueError("det must be 1, 2, or 'both'.")

        # Internal helper to draw slits and targets on a given axis
        def plot_region(ax, region, color, side_label):
            num_targets = len(region[0])
            label = f" N = {num_targets}"

            for i in range(len(region[0])):
                slit_x_range = region[0][i]
                slit_y_range = region[1][i]
                width = slit_x_range[1] - slit_x_range[0]
                height = slit_y_range[1] - slit_y_range[0]

                rect = patches.Rectangle(
                    (slit_x_range[0], slit_y_range[0]),
                    width,
                    height,
                    linewidth=1,
                    edgecolor="blue",
                    facecolor="none"
                )
                ax.add_patch(rect)

            ax.scatter(region[2], region[3], s=10, color=color, label=label)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"Detector {side_label}")
            ax.set_aspect("equal")
            ax.grid(True)
            ax.legend()

        # Plot based on detector selection
        if ccdnum == 'both':
            plot_region(axA, region_1, color="red", side_label="1")
            plot_region(axB, region_2, color="green", side_label="2")
        elif ccdnum == 1:
            plot_region(axA, region_1, color="red", side_label="1")
        elif ccdnum == 2:
            plot_region(axB, region_2, color="green", side_label="2")

        # Save to file if directory provided
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, save_filename)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()

        # Return the plotted region data
        if ccdnum == 'both':
            return region_1, region_2
        elif ccdnum == 1:
            return region_1
        elif ccdnum == 2:
            return region_2

    from astropy.coordinates import SkyCoord
    import astropy.units as u
    import numpy as np


    def radec_pixel_offset(self, ra1, dec1, ra2, dec2, filename, ccdnum, Nx=4096, Ny=4112):
        """
        Compute the pixel offset between two sky positions on the Binospec detector.

        Parameters
        ----------
        ra1, dec1 : float
            First coordinate in degrees.
        ra2, dec2 : float
            Second coordinate in degrees.
        filename : str
            Path to slitmask FITS file.
        ccdnum : int
            Detector number (1 or 2).
        Nx, Ny : int
            Detector dimensions (default Binospec size).

        Returns
        -------
        dx_pix, dy_pix : float
            Offset in X and Y (pixels) from (ra1, dec1) to (ra2, dec2)
        dr_pix : float
            Euclidean distance in pixels
        """
        from astropy.io import fits

        # Constants (from your bino_get_slit_region)
        dy0 = -200.0
        y_scl = 24.555832 if ccdnum == 1 else 24.548194  # arcsec to pixels

        # Read slitmask FITS
        hdu = fits.open(filename)
        mask_fits = hdu[9].data[0] if ccdnum == 1 else hdu[10].data[0]

        # Mask center (used as origin in slitmask coordinates)
        mask_center = SkyCoord(mask_fits['CENTERRA'], mask_fits['CENTERDEC'], unit=('hourangle', 'deg'))

        # Position angle of +x detector axis (spatial direction)
        posx_pa = hdu[1].header['POSANG'] - 180.
        if posx_pa < 0:
            posx_pa += 360.

        # Create SkyCoord objects for each position
        coord1 = SkyCoord(ra1, dec1, unit='deg')
        coord2 = SkyCoord(ra2, dec2, unit='deg')

        # Project both coordinates into the slitmask pixel frame
        offset1 = mask_center.spherical_offsets_to(coord1)
        offset2 = mask_center.spherical_offsets_to(coord2)

        # Each offset is (dRA, dDec) in arcsec. Rotate using POSANG to detector frame
        # Convert to arcsec offsets along detector x and y axes
        def project_to_detector_frame(ra_offset, dec_offset, posang):
            theta = np.deg2rad(posang)
            dx = ra_offset * np.cos(theta) + dec_offset * np.sin(theta)
            dy = -ra_offset * np.sin(theta) + dec_offset * np.cos(theta)
            return dx, dy

        dx1, dy1 = project_to_detector_frame(offset1[0].arcsec, offset1[1].arcsec, posx_pa)
        dx2, dy2 = project_to_detector_frame(offset2[0].arcsec, offset2[1].arcsec, posx_pa)

        # Convert to pixel coordinates (centered on Nx/2, Ny-1 with dy0 shift)
        x1_pix = dx1 * y_scl + Nx / 2.0
        x2_pix = dx2 * y_scl + Nx / 2.0

        y1_pix = Ny - 1 - (dy1 * y_scl) + dy0
        y2_pix = Ny - 1 - (dy2 * y_scl) + dy0

        # Compute pixel offsets
        dx_pix = x2_pix - x1_pix
        dy_pix = y2_pix - y1_pix
        dr_pix = np.sqrt(dx_pix ** 2 + dy_pix ** 2)

        return dx_pix, dy_pix, dr_pix

    from astropy.io import fits
    import numpy as np

    from astropy.io import fits
    import numpy as np

    from astropy.io import fits
    import numpy as np

    def get_slit_y_edges(self, filename, ccdnum, slit_index=0, Ny=4112):
        """
        Compute y pixel edges (top and bottom) of a single slit from Binospec
        mask design file, exactly matching PypeIt's get_maskdef_slitedges().

        Parameters
        ----------
        filename : str
            Path to the Binospec slitmask FITS file.
        ccdnum : int
            Detector number (1 or 2).
        slit_index : int
            Index of valid slit (among those where offset <= 2").
        Ny : int
            Image height in pixels.

        Returns
        -------
        bot_edge : float
            Bottom y pixel coordinate of the slit.
        top_edge : float
            Top y pixel coordinate of the slit.
        """

        if filename is None or ccdnum not in [1, 2]:
            raise ValueError("Must provide valid filename and ccdnum")

        hdu = fits.open(filename)
        mask_fits = hdu[9].data[0] if ccdnum == 1 else hdu[10].data[0]
        mask_corners = np.array(mask_fits['MASK_CORNERS'])

        target_type = mask_fits['TARGET_TYPE']
        targ = np.where(target_type == 'TARGET')[0]
        if len(targ) == 0:
            raise ValueError("No slits with TARGET_TYPE='TARGET' found.")

        # Offsets
        poly_x = np.mean(mask_fits['POLY_X'], axis=0)[targ]
        poly_y = np.mean(mask_fits['POLY_Y'], axis=0)[targ]
        delta_x = mask_fits['SLITX'][targ] - poly_x
        delta_y = mask_fits['SLITY'][targ] - poly_y
        offsets = np.sqrt(delta_x ** 2 + delta_y ** 2)
        good = np.where(offsets <= 2)[0]

        if slit_index >= len(good):
            raise IndexError("slit_index out of range of good slits")

        i = good[slit_index]

        # Grab slit corners (Y only)
        corners_y = np.squeeze(mask_fits['POLY_Y'].T)[targ]
        corners_y = corners_y[good]  # only good slits
        corners_y = corners_y[:, [3, 0, 1, 2]]  # reorder to match get_slitmask()

        y_top_arcsec = -corners_y[slit_index, 0]  # top corner (reversed)
        y_bot_arcsec = -corners_y[slit_index, 2]  # bottom corner (reversed)

        # Pixel scale
        y_scl = 24.555832 if ccdnum == 1 else 24.548194
        dy0 = -200.0

        top_edge = Ny - 1 - ((y_top_arcsec - mask_corners[1]) * y_scl) + dy0
        bot_edge = Ny - 1 - ((y_bot_arcsec - mask_corners[1]) * y_scl) + dy0

        # Ensure correct order
        if top_edge < bot_edge:
            top_edge, bot_edge = bot_edge, top_edge

        return bot_edge, top_edge

    def extract_y_slit_info(self, filename, ccdnum, slit_index=0):
        """
        Extracts arcsec positions of top/bottom slit corners and mask center Y value
        from a Binospec slitmask FITS file for computing y-pixel edges.

        This version does NOT filter slits based on target offset.

        Parameters
        ----------
        filename : str
            Path to the slitmask FITS file.
        ccdnum : int
            Detector number (1 or 2).
        slit_index : int
            Index of the slit among all TARGET-type entries (no offset filtering).

        Returns
        -------
        dict
            Dictionary containing:
            - 'y_top_arcsec' : float
            - 'y_bot_arcsec' : float
            - 'mask_y_arcsec' : float
            - 'y_scl' : float
        """

        if filename is None or ccdnum not in [1, 2]:
            raise ValueError("Must provide valid filename and CCD number")

        hdu = fits.open(filename)
        mask_fits = hdu[9].data[0] if ccdnum == 1 else hdu[10].data[0]

        # Identify TARGET-type slits
        target_type = mask_fits['TARGET_TYPE']
        targ = np.where(target_type == 'TARGET')[0]

        if slit_index >= len(targ):
            raise IndexError(f"slit_index={slit_index} exceeds number of TARGET slits ({len(targ)})")

        i = targ[slit_index]  # raw index into full slit list

        # POLY_Y: shape (4, N), each column is a slit
        poly_y = np.squeeze(mask_fits['POLY_Y'].T)  # shape (N, 4)
        corners_y = poly_y[targ]  # only TARGET-type
        corners_y = corners_y[:, [3, 0, 1, 2]]  # reorder to clockwise

        y_top_arcsec = -corners_y[slit_index, 0]
        y_bot_arcsec = -corners_y[slit_index, 2]

        mask_y_arcsec = mask_fits['MASK_CORNERS'][1]
        y_scl = 24.555832 if ccdnum == 1 else 24.548194

        return {
            'poly_y': mask_fits['POLY_Y'].T[slit_index],
            'y_top_arcsec': y_top_arcsec,
            'y_bot_arcsec': y_bot_arcsec,
            'mask_y_arcsec': mask_y_arcsec,
            'y_scl': y_scl
        }


def binospec_read_amp(inp, ext):
    """
    Read one amplifier of an MMT BINOSPEC multi-extension FITS image

    Parameters
    ----------
    inp: tuple
      (str,int) filename, extension
      (hdu,int) FITS hdu, extension

    Returns
    -------
    data
    predata
    postdata
    x1
    y1

    ;------------------------------------------------------------------------
    function lris_read_amp, filename, ext, $
      linebias=linebias, nobias=nobias, $
      predata=predata, postdata=postdata, header=header, $
      x1=x1, x2=x2, y1=y1, y2=y2, GAINDATA=gaindata
    ;------------------------------------------------------------------------
    ; Read one amp from LRIS mHDU image
    ;------------------------------------------------------------------------
    """
    # Parse input
    if isinstance(inp, str):
        hdu = io.fits_open(inp)
    else:
        hdu = inp
    # get entire extension...
    temp = hdu[ext].data.transpose()
    nxt = temp.shape[0]
    nyt = temp.shape[1]

    # parse the DETSEC keyword to determine the size of the array.
    header = hdu[ext].header

    # parse the DATASEC keyword to determine the size of the science region (unbinned)
    datasec = header['DATASEC']
    xdata1, xdata2, ydata1, ydata2 = np.array(parse.load_sections(datasec, fmt_iraf=False)).flatten()
    datasec = '[{:}:{:},{:}:{:}]'.format(xdata1 - 1, xdata2, ydata1-1, ydata2)

    #TODO: Since pypeit can only subtract overscan along one axis, I'm subtract the overscan here using median method.
    # Overscan X-axis
    if xdata1 > 1:
        overscanx = temp[2:xdata1-1, :]
        overscanx_vec = np.median(overscanx, axis=0)
        temp = temp - overscanx_vec[None,:]
    data = temp[xdata1 - 1:xdata2, ydata1 -1 : ydata2]

    ## Overscan Y-axis
    if ydata2<nyt:
        os1, os2 = ydata2+1, nyt-1
        overscany = temp[xdata1 - 1:xdata2, ydata2:os2]
        overscany_vec = np.median(overscany, axis=1)
        data = data -  overscany_vec[:,None]

    # Overscan
    biassec = '[0:{:},{:}:{:}]'.format(xdata1-1, ydata1-1, ydata2)
    xos1, xos2, yos1, yos2 = np.array(parse.load_sections(biassec, fmt_iraf=False)).flatten()
    overscan = np.zeros_like(temp[xos1:xos2, yos1:yos2]) # Give a zero fake overscan at the edge of each amplifiers
    #overscan = temp[xos1:xos2,yos1:yos2]

    return data, overscan, datasec, biassec

