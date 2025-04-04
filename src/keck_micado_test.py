import matplotlib.pyplot as plt
import scopesim
scopesim.rc.__config__["!SIM.file.local_packages_path"] = "./irdb"
from scopesim_templates.stellar.clusters import cluster

source = cluster(
    mass=1000,        # Msun
    distance=50000,   # parsec
    core_radius=0.3,  # parsec
    seed=9002,        # random number seed
)

simulation = scopesim.Simulation("keck_MICADO", ["SCAO", "IMG_4mas"])
hdul = simulation(source)

# hdul.writeto("TEST.fits")

simulation.plot(norm="log", vmin=3e3, vmax=3e4, cmap="hot",
                fig_kwargs={"figsize": (8, 8), "layout": "tight"})
plt.show()
