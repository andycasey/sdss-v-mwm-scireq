
import os
import numpy as np
from astropy.io import fits
from glob import glob

from .config import config

def get_vacuum_wavelength_array(hdu):

    P = hdu.header.get("NWAVE", hdu.header.get("NAXIS1", 8575))
    crval, cdelt = (hdu.header["CRVAL1"], hdu.header["CDELT1"])
    vacuum_wavelength = 10**(crval + cdelt * np.arange(P))
    return vacuum_wavelength

def get_spectrum_path(telescope, field, location_id, filename):
    """
    Return the full path of a specified spectrum.

    :param telescope:
        The telescope identifier (e.g., apo25m).

    :param field:
        The field identifier given to the spectrum (e.g., 120+12).
    
    :param location:
        The location identifier given to the spectrum (e.g., 120+12).

    :param filename:
        The basename of the file, (e.g., apStar-r8-2M14593130+4725291.fits)
    """

    field = str(field).strip()
    telescope, filename = (telescope.strip(), filename.strip())
    location_id = str(location_id).strip()
    if "/" in "".join([telescope, field, location_id, filename]):
        raise ValueError("telescope, field, and filename cannot contain"
                         "slashes (/)")

    possible_paths = [
        os.path.join(config["APOGEE_DR14_DIR"], telescope, field, filename),
        os.path.join(config["APOGEE_DR14_DIR"], telescope, location_id, filename)
    ]

    for path in possible_paths:
        if os.path.exists(path):
            break

    return path
    
def get_aspcapstar_spectrum_path(location_id, apogee_id):

    location_id = str(location_id).strip()
    apogee_id = apogee_id.strip()
    if "/" in "".join([location_id, apogee_id]):
        raise ValueError("location_id and apogee_id cannot contain slashes (/)")

    filename = "aspcapStar-r8-l31c.2-{}.fits".format(apogee_id)
    path = os.path.join(config["ASPCAP_DR14_DIR"], location_id, filename)

    return path



def read_aspcapstar_spectrum(location_id, apogee_id, return_model_spectrum=False,
    full_output=False, **kwargs):

    path = get_aspcapstar_spectrum_path(location_id, apogee_id)

    image = fits.open(path)
    vacuum_wavelength = get_vacuum_wavelength_array(image[1])

    if return_model_spectrum:
        flux = image[3].data
        ivar = 10000 * np.ones_like(flux) # sigma = 0.01
    
    else:
        flux = image[1].data
        ivar = (image[2].data)**-2

    image.close()
    
    metadata = dict(
        location_id=location_id, apogee_id=apogee_id, path=path,
        is_model_spectrum=return_model_spectrum)

    if full_output:
        raise NotImplementedError

    return (vacuum_wavelength, flux, ivar, metadata)
    

def read_spectrum(telescope, field_id, location_id, filename, combined=True,
    combined_weighting_method="individual", full_output=False, **kwargs):
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

    available_methods = ["individual", "global"]
    combined_weighting_method = str(combined_weighting_method).lower()
    if combined and combined_weighting_method not in available_methods:
        raise ValueError("combined_weighting_method must be either {}"\
                         .format(" or ".join(available_methods)))

    path = get_spectrum_path(telescope, field_id, location_id, filename)

    image = fits.open(path)

    #with fits.open(path) as image:

    # Build wavelength array first.
    vacuum_wavelength = get_vacuum_wavelength_array(image[0])

    if combined:
        flux_start = available_methods.index(combined_weighting_method)
        flux_end = flux_start + 1
    else:
        flux_start, flux_end = (2, None)
        
    flux = np.atleast_2d(image[1].data)[flux_start:flux_end]
    flux_error = np.atleast_2d(image[2].data)[flux_start:flux_end]
    
    ivar = flux_error**-2

    small_value = kwargs.pop("small_value", 1e-20)
    ivar[ivar <= small_value] = 0

    if combined: assert flux.size >= vacuum_wavelength.size
    assert flux.shape == ivar.shape

    assert np.all(np.isfinite(flux))
    assert np.all(np.isfinite(ivar))
    assert np.all(ivar >= 0)

        
    # TODO: Add units?
    snr_combined = image[0].header["SNR"]
    snr_visits = [image[0].header["SNRVIS{:.0f}".format(i)] \
                  for i in range(1, 1 + image[0].header["NVISITS"])]

    image.close()
    
    metadata = dict(
        telescope=telescope, location_id=location_id, field_id=field_id,
        filename=filename,
        path=path, combined_weighting_method=combined_weighting_method,
        snr_combined=snr_combined, snr_visits=snr_visits)


    if full_output:
        raise NotImplementedError

    return (vacuum_wavelength, flux, ivar, metadata)