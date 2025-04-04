import scopesim
from scopesim_templates.stellar.clusters import cluster
import matplotlib.pyplot as plt
from scopesim.optics import FieldOfView
from astropy.io.fits import Header

scopesim.rc.__config__["!SIM.file.local_packages_path"] = "./irdb"
scopesim.rc.__packages__ = {}

source = cluster(
    mass=1000,
    distance=50000,
    core_radius=0.3,
    seed=9002,
)

# Create empty image data
naxis1, naxis2 = 512, 512

# Define your WCS parameters
pixel_scale_deg = 0.00027778  # arcsec/pixel ≈ 1 arcsec → deg
crpix1 = (naxis1 + 1) / 2
crpix2 = (naxis2 + 1) / 2
crval1 = 150.0  # RA in deg
crval2 = 2.0    # DEC in deg

# Construct the header
hdr = Header()
hdr["NAXIS"]    = 2
hdr["NAXIS1"]   = 512
hdr["NAXIS2"]   = 512

hdr["CRPIX1"]   = 256.5
hdr["CRPIX2"]   = 256.5
hdr["CRVAL1"]   = 150.0
hdr["CRVAL2"]   = 2.0

hdr["CDELT1"]   = -0.00027778  # degrees/pixel
hdr["CDELT2"]   =  0.00027778

hdr["CTYPE1"]   = "RA---TAN"
hdr["CTYPE2"]   = "DEC--TAN"
hdr["CUNIT1"]   = "deg"
hdr["CUNIT2"]   = "deg"

hdr["CD1_1"]    = -0.00027778
hdr["CD1_2"]    =  0.0
hdr["CD2_1"]    =  0.0
hdr["CD2_2"]    =  0.00027778

hdr["WCSAXES"]  = 2
hdr["RADESYS"]  = "ICRS"
hdr["EQUINOX"]  = 2000.0
hdr["LATPOLE"]  = 90.0
hdr["LONPOLE"]  = 180.0

hdr["CDELT1D"]  = -0.00027778
hdr["CRVAL1D"]  = 150.0
hdr["CRPIX1D"]  = 256.5


# Confirm it works with Astropy before giving to ScopeSim
from astropy.wcs import WCS
w = WCS(hdr)
print("WCS has celestial?", w.has_celestial)

try:
    simulation = scopesim.Simulation("ZShooter", ["SPECTRO_MODE"])

    print("Image-plane keys?", all(k in hdr for k in ["CDELTD1", "CRVALD1", "CRPIXD1", "NAXIS1"]))

    from scopesim.utils import has_needed_keywords

    # Show what keys are actually in the header
    print("Header keys:", list(hdr.keys()))

    # Emulate the internal check
    print("has_needed_keywords D:", has_needed_keywords(hdr, "D"))
    print("has_needed_keywords S:", has_needed_keywords(hdr, "S"))
    print("has_needed_keywords :", has_needed_keywords(hdr, ""))

    fov = FieldOfView(header=hdr, waverange=(0.6, 1.0))
    fov.meta["image_plane_id"] = 0
    fov.image_plane_id = 0
    simulation.optical_train.fov_manager.fovs.append(fov)
    simulation.optical_train.fov_manager._fovs_list = [fov]
    

    print("FOVs:", simulation.optical_train.fov_manager.fovs)

    print("FOV wave_min:", fov.meta.get("wave_min"))
    print("Image plane ID:", fov.meta.get("image_plane_id"))
except Exception as e:
    print(f'Error creating simulation: {e}')
    raise

hdul = simulation(source)

# # hdul.writeto("output.fits")  # optional

# simulation.plot(norm="log", vmin=3e3, vmax=3e4, cmap="hot", fig_kwargs={"figsize": (8, 8), "layout": "tight"})
# plt.show()
