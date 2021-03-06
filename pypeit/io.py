# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
Provides a set of I/O routines.
"""
import os
import gzip
import shutil
import numpy

from astropy.io import fits


def init_record_array(shape, dtype):
    r"""
    Utility function that initializes a record array using a provided
    input data type.  For example::

        dtype = [ ('INDX', numpy.int, (2,) ),
                  ('VALUE', numpy.float) ]

    Defines two columns, one named `INDEX` with two integers per row and
    the one named `VALUE` with a single float element per row.  See
    `numpy.recarray`_.
    
    Args:
        shape (:obj:`int`, :obj:`tuple`):
            Shape of the output array.
        dtype (:obj:`list`):
            List of the tuples that define each element in the record
            array.

    Returns:
        `numpy.recarray`: Zeroed record array
    """
    return numpy.zeros(shape, dtype=dtype).view(numpy.recarray)


def rec_to_fits_type(rec_element):
    """
    Return the string representation of a fits binary table data type
    based on the provided record array element.
    """
    n = 1 if len(rec_element[0].shape) == 0 else rec_element[0].size
    if rec_element.dtype == numpy.bool:
        return '{0}L'.format(n)
    if rec_element.dtype == numpy.uint8:
        return '{0}B'.format(n)
    if rec_element.dtype == numpy.int16 or rec_element.dtype == numpy.uint16:
        return '{0}I'.format(n)
    if rec_element.dtype == numpy.int32 or rec_element.dtype == numpy.uint32:
        return '{0}J'.format(n)
    if rec_element.dtype == numpy.int64 or rec_element.dtype == numpy.uint64:
        return '{0}K'.format(n)
    if rec_element.dtype == numpy.float32:
        return '{0}E'.format(n)
    if rec_element.dtype == numpy.float64:
        return '{0}D'.format(n)
    
    # If it makes it here, assume its a string
    l = int(rec_element.dtype.str[rec_element.dtype.str.find('U')+1:])
#    return '{0}A'.format(l) if n==1 else '{0}A{1}'.format(l*n,l)
    return '{0}A'.format(l*n)


def rec_to_fits_col_dim(rec_element):
    """
    Return the string representation of the dimensions for the fits
    table column based on the provided record array element.

    The shape is inverted because the first element is supposed to be
    the most rapidly varying; i.e. the shape is supposed to be written
    as row-major, as opposed to the native column-major order in python.
    """
    return None if len(rec_element[0].shape) == 1 else str(rec_element[0].shape[::-1])


def rec_to_bintable(arr, name=None):
    """
    Construct an `astropy.io.fits.BinTableHDU` from a record array.

    Args:
        arr (`numpy.recarray`):
            The data array to write to a binary table.
        name (:obj:`str`, optional):
            The name for the binary table extension.
    
    Returns:
        `astropy.io.fits.BinTableHDU`: The binary fits table that can be
        included in an `astropy.io.fits.HDUList` and written to disk.
    """
    return fits.BinTableHDU.from_columns([fits.Column(name=n,
                                                      format=rec_to_fits_type(arr[n]),
                                                      dim=rec_to_fits_col_dim(arr[n]),
                                                      array=arr[n])
                                            for n in arr.dtype.names], name=name)


def compress_file(ifile, overwrite=False, rm_original=True):
    """
    Compress a file using gzip package.
    
    Args:
        ifile (:obj:`str`):
            Name of file to compress.  Output file with have the same
            name with '.gz' appended.
        overwrite (:obj:`bool`, optional):
            Overwrite any existing file.
        rm_original (:obj:`bool`, optional):
            The method writes the compressed file such that both the
            uncompressed and compressed file will exist when the
            compression is finished.  If this is True, the original
            (uncompressed) file is removed.

    Raises:
        ValueError:
            Raised if the file name already has a '.gz' extension or if
            the file exists and `overwrite` is False.
    """
    # Nominally check if the file is already compressed
    if ifile.split('.')[-1] == 'gz':
        raise ValueError('File appears to already have been compressed! {0}'.format(ifile))

    # Construct the output file name and check if it exists
    ofile = '{0}.gz'.format(ifile)
    if os.path.isfile(ofile) and not overwrite:
        raise FileExistsError('{0} exists! To overwrite, set overwrite=True.'.format(ofile))

    # Compress the file
    with open(ifile, 'rb') as f_in:
        with gzip.open(ofile, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    if rm_original:
        # Remove the uncompressed file
        os.remove(ifile)


def parse_hdr_key_group(hdr, prefix='F'):
    """
    Parse a group of fits header values grouped by a keyword prefix.

    If the prefix is 'F', the header keywords are expected to be, e.g.,
    'F1', 'F2', 'F3', etc.  The list of values returned are then, e.g.,
    [ hdr['F1'], hdr['F2'], hdr['F3'], ... ].  The function performs not
    retyping, so the values in the returned list have whatever type they
    had in the fits header.

    Args:
        hdr (`fits.Header`):
            Astropy Header object
        prefix (:obj:`str`, optional):
            The prefix used for the header keywords.
        
    Returns:
        list: The list of header values ordered by their keyword index.
    """
    values = {}
    for k, v in hdr.items():
        # Check if this header keyword starts with the required prefix
        if k[:len(prefix)] == prefix:
            try:
                # Try to convert the keyword without the prefix into an
                # integer; offseting to an array index.
                i = int(k[len(prefix):])-1
            except ValueError:
                # Assume the value is some other random keyword that
                # starts with the prefix but isn't part of the keyword
                # group
                continue
            # Assume we've found a valid value; assign it and keep
            # trolling the header
            values[i] = v

    # Convert from dictionary with integer keys to an appropriately
    # sorted list
    return [values[i] for i in range(max(values.keys())+1)]


