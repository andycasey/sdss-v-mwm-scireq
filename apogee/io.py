
import os
import numpy as np
from astropy.io import fits

from .config import config


def read_spectrum(telescope, location_id, filename, combined=True,
    combined_weighting_method="individual", full_output=False):
    """
    Read an APOGEE spectrum given the telescope used to observe it, the location
    identifier of the spectrum, and the spectrum filename.

    :param telescope:
        The telescope identifier (e.g., apo25m).

    :param location_id:
        The location identifier given to the spectrum (e.g., 2227).

    :param filename:
        The basename of the file, (e.g., apStar-r8-2M14593130+4725291.fits)

    :param combined: [optional]
        Return the combined spectrum. Otherwise, return individual visits.

    :param combined_weighting_method: [optional]
        The method to use when returning a combined spectrum. Options are either
        "individual" or "global". This parameter is ignored if `combined` is
        `False`.

    :returns:
        A four-length tuple containing: 
            an array of vacuum wavelengths for each pixel in units of Angstrom, 
            the flux in units of 10^-17 ergs/s/cm^2/Angstrom,
            the inverse variance of the flux in 10^-17 ergs/s/cm^2/Angstrom,
            a dictionary metadata.
    """

    location_id = str(location_id)
    if "/" in "".join([telescope, location_id, filename]):
        raise ValueError("telescope, location_id, and filename cannot contain"
                         "slashes (/)")

    available_methods = ["individual", "global"]
    combined_weighting_method = str(combined_weighting_method).lower()
    if combined and combined_weighting_method not in available_methods:
        raise ValueError("combined_weighting_method must be either {}"\
                         .format(" or ".join(available_methods)))

    path = os.path.join(
        config["APOGEE_DR14_DIR"], telescope, location_id, filename)

    image = fits.open(path)

    #with fits.open(path) as image:

    # Build wavelength array first.
    P = image[0].header["NWAVE"]
    crval, cdelt = (image[0].header["CRVAL1"], image[0].header["CDELT1"])
    vacuum_wavelength = 10**(crval + cdelt * np.arange(P))

    if combined:
        flux_start = available_methods.index(combined_weighting_method)
        flux_end = flux_start + 1
    else:
        flux_start, flux_end = (2, None)
        raise NotImplementedError("only access combined spectra so far")

    flux = np.atleast_2d(image[1].data)[flux_start:flux_end]
    flux_error = np.atleast_2d(image[2].data)[flux_start:flux_end]
    
    ivar = flux_error**-2

    assert flux.size >= P
    assert flux.shape == ivar.shape

    assert np.all(np.isfinite(flux))
    assert np.all(np.isfinite(ivar))
    assert np.all(ivar >= 0)

    image.close()
        
    # TODO: Add units?


    metadata = dict(
        telescope=telescope, location_id=location_id, filename=filename,
        path=path, combined_weighting_method=combined_weighting_method,)

    if full_output:
        raise NotImplementedError

    return (vacuum_wavelength, flux, ivar, metadata)