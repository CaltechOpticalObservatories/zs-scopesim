import scopesim
from scopesim_templates.stellar.clusters import cluster
import matplotlib.pyplot as plt

scopesim.rc.__config__["!SIM.file.local_packages_path"] = "./irdb"
scopesim.rc.__packages__ = {}

source = cluster(
    mass=1000,
    distance=50000,
    core_radius=0.3,
    seed=9002,
)

try:
    simulation = scopesim.Simulation("basic_instrument", ["imaging"])
    hdul = simulation(source)
except Exception as e:
    print(f'Error creating simulation: {e}')
    raise


# hdul.writeto("output.fits")  # optional
simulation.plot(norm="log", vmin=3e3, vmax=3e4, cmap="hot", fig_kwargs={"figsize": (8, 8), "layout": "tight"})
plt.show()
