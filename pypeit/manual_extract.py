"""
Implements an object to handle manual object extraction.

.. include:: ../include/links.rst
"""
import inspect

from IPython import embed

import numpy as np

from pypeit import msgs
from pypeit import datamodel
from pypeit.core import parse

class ManualExtractionObj(datamodel.DataContainer):
    """
    A data container holding the arguments for how to perform the
    manual extraction of a spectrum.

    A list of these objects is generated in pypeit.py
    to perform a set of user-defined extractions.

    For an example of how to define a series of manual extractions in
    the pypeit input file, see :ref:`pypeit_file`.

    The datamodel attributes are:

    .. include:: ../include/class_datamodel_manualextractionobj.rst

    Args:
        frame (:obj:`str`):
            The name of the fits file for a manual extraction
        spat (`numpy.ndarray`_): Array of spatial positions to hand extract
        spec (`numpy.ndarray`_): Array of spectral positions to hand extract
        det (`numpy.ndarray`_): Array of detectors for hand extraction. 
            This must be a aligned with spec and spat .
            The values can be negative (for negative images)
        fwhm (`numpy.ndarray`_): Array of FWHM for hand extraction. 
            This must be aligned with spec and spat.
        boxcar_rad (`numpy.ndarray`_, optional): Array of boxcar_radii for hand extraction. 
            This must be aligned with spec and spat.
            It is to be in *pixels*, not arcsec.
            This is only intended for multi-slit reductions (not Echelle)


    """
    version = '1.1.0'

    datamodel = {
        'frame': dict(otype=str,
                    descr='The name of the fits file for a manual extraction'),
        'detname': dict(otype=np.ndarray, atype=str,
                    descr='detectors name for hand extraction.'),
        'spec': dict(otype=np.ndarray, atype=np.floating, 
                    descr='spectral positions to hand extract'),
        'spat': dict(otype=np.ndarray, atype=np.floating, 
                    descr='spatial positions to hand extract'),
        'fwhm': dict(otype=np.ndarray, atype=np.floating, 
                    descr='FWHMs for hand extractions'),
        'neg': dict(otype=np.ndarray, atype=np.bool_,
                     descr='Flags indicating which hand extract is a negative trace'),
        'boxcar_rad': dict(otype=np.ndarray, atype=np.floating, 
                    descr='Boxcar radius for hand extractions (optional)'),
    }

    @classmethod
    def by_fitstbl_input(cls, frame: str, inp: str, spectrograph):
        """Generate the object from an entry in the fitstbl

        Args:
            frame (str):
                filename
            inp (str):
                String specifying the manual aperture: ``det:spat:spec:fwhm``;
                e.g., ``1:1181.8:3820.6:3.``
            spectrograph (:class:`pypeit.spectrographs.spectrograph.Spectrograph`):
                The `Spectrograph` instance that sets the instrument used to
                take the observations.  Used to set check that the input value
                of the mosaic detectors are allowed for this spectrograph

        Returns:
            ManualExtractionObj:
        """
        # Generate a dict
        idict = dict(spat=[], spec=[], detname=[], fwhm=[], neg=[], boxcar_rad=[])
        m_es = inp.split(';')
        for m_e in m_es:
            loc = parse.parse_image_location(m_e, spectrograph)
            if len(loc) not in [5,6]:
                msgs.error('Definition of manual extraction aperture does not have the correct '
                           f'number of parameters: {m_e}.')

            # TODO: Why is this spat:spec and not spec:spat like everything else??
            idict['neg'] += [loc[0]]
            idict['detname'] += [loc[1]]
            idict['spat'] += [loc[2]]
            idict['spec'] += [loc[3]]
            idict['fwhm'] += [loc[4]]
            idict['boxcar_rad'] += [loc[5] if len(loc) == 6 else -1.]
                
        # Build me
        return cls(frame=frame, spat=np.array(idict['spat']), 
                   spec=np.array(idict['spec']),
                   fwhm=np.array(idict['fwhm']),
                   detname=np.array(idict['detname']),
                   neg=np.array(idict['neg']),
                   boxcar_rad=np.array(idict['boxcar_rad']))

    def __init__(self, frame=None, spat=None, spec=None, detname=None, 
                 fwhm=None, neg=None, boxcar_rad=None):
        # Parse
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        d = dict([(k,values[k]) for k in args[1:]])
        # Setup the DataContainer
        datamodel.DataContainer.__init__(self, d=d)

    def _validate(self):
        """Validate

        A couple of quick checks..

        Raises:
            ValueError: Raised if one of the arrays is not set or if they don't have the same length
        """
        if len(self.spec) != len(self.spat):
            raise ValueError("spec and spat not of the same length")
        if len(self.fwhm) != len(self.detname):
            raise ValueError("FWHM and det not of the same length")

    def dict_for_objfind(self, detname, neg=False):
        """
        Repackage into a dict for the extraction code

        Args:
            det (str):
                Detector name under consideration
            neg (bool, optional):
                If True, return the negative image requests

        Returns:
            dict or None: To be passed into reduce.find_objects()

        """
        # Find the ones we want
        if neg:
            gd_det = (self.neg == True) & (self.detname == detname)
        else:
            gd_det = (self.neg == False) & (self.detname == detname)
        # None?
        if not np.any(gd_det):
            return None
        # Fill 
        manual_extract_dict = {}
        for key in ['spec', 'spat', 'detname', 'fwhm', 'boxcar_rad']:
            manual_extract_dict[key] = self[key][gd_det]
        # Return
        return manual_extract_dict



class ManualCubeExtractionObj(datamodel.DataContainer):
    """
    A data container holding the arguments for how to perform the
    manual extraction of a spectrum from a datacube with 
    pypeit_extract_datacube.
    
    To perform a set of user-defined extractions on a datacube, a list of these 
    objects is created in the .extract file which can be 
    input to pypeit_extract_datacube with the -e option. 

    The datamodel attributes are:

    .. include:: ../include/class_datamodel_manual_cube_extractionobj.rst

    Args:
        spatx (`numpy.ndarray`_): Array of spatial positions to hand extract
        spaty (`numpy.ndarray`_): Array of spectral positions to hand extract
        fwhm (`numpy.ndarray`_): Array of FWHM for hand extraction in arcserconds.  
            This must be aligned with spatx and spaty.
        boxcar_rad (`numpy.ndarray`_, optional): Array of boxcar_radii for hand extraction. 
            This must be aligned with spatx and spaty.
            It is to be arcsec and is optional. 

    """
    version = '1.1.0'

    datamodel = {
        'spatx': dict(otype=np.ndarray, atype=np.floating, 
                    descr='Cube spatial x positions to hand extract'),
        'spaty': dict(otype=np.ndarray, atype=np.floating, 
                    descr='Cube spatial y positions to hand extract'),
        'fwhm': dict(otype=np.ndarray, atype=np.floating, 
                    descr='FWHMs for hand extractions in arcseconds (optional, not yet used)'),
        'boxcar_rad': dict(otype=np.ndarray, atype=np.floating, 
                    descr='Boxcar radius for hand extractions in arcseconds (optional, not yet used)'),
    }

    @classmethod
    def parse(cls, inp: str):
        """Generate the object from a string input from the manual parameter in the parset

        Args:
            inp (str):
                String specifying the manual aperture: ``spatx:spaty:fwhm``;
                e.g., ``25.6:10.2:1.1``

        Returns:
            ManualExtractionObj:
        """
        # Generate a dict
        idict = dict(spatx=[], spaty=[], fwhm=[], boxcar_rad=[])
        m_es = inp.split(';')
        for m_e in m_es:
            parse = m_e.split(':')
            idict['spatx'] += [float(parse[0])]
            idict['spaty'] += [float(parse[1])]
            
            # FWHM?
            if len(parse) >= 3:
                idict['fwhm'] += [float(parse[2])]
            else:
                idict['fwhm'] += [-1.]

            # Boxcar?
            if len(parse) >= 4:
                idict['boxcar_rad'] += [float(parse[3])]
            else:
                idict['boxcar_rad'] += [-1.]

        # Build me
        return cls(spatx=np.array(idict['spatx']), 
                   spaty=np.array(idict['spaty']),
                   fwhm=np.array(idict['fwhm']),
                   boxcar_rad=np.array(idict['boxcar_rad']))

    def __init__(self, spatx=None, spaty=None, fwhm=None, boxcar_rad=None):
        # Parse
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        d = dict([(k,values[k]) for k in args[1:]])
        # Setup the DataContainer
        datamodel.DataContainer.__init__(self, d=d)

    def _validate(self):
        """Validate

        A couple of quick checks..

        Raises:
            ValueError: Raised if one of the arrays is not set or if they don't have the same length
        """
        if len(self.spatx) != len(self.spaty):
            raise ValueError("spatx and spaty not of the same length")
        #if len(self.fwhm) != len(self.spatx):
        #    raise ValueError("FWHM and spatx and spaty not of the same length")


    def to_dict(self):
        """
        Repackage into a dict for the extraction code

        Returns:
            dict or None: To be passed into datacube.extract_point_source

        """
        # Fill 
        manual_extract_dict = {}
        for key in ['spatx', 'spaty', 'fwhm', 'boxcar_rad']:
            manual_extract_dict[key] = self[key]
        # Return
        return manual_extract_dict
