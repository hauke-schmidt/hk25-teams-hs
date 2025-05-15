# %% load packages
import os 
from glob import glob
from calendar import c
import numpy as np
import intake
import pandas as pd
from easygems import healpix as egh
import healpix
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
from cmocean import cm
import cfgrib

warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # don't warn us about future package conflicts

# %% define mask for relevant region

extent = [280, 350, 58, 85]

#%%
# #-------------------
# # LOAD ERA5 SST DATA
# #-------------------

# # Directory containing the GRIB files
datadir = '/pool/data/ERA5/E5/sf/an/1H/034'

filelist = np.sort(glob(datadir + '/*.grb').filter(lambda x: '2020' in x))


# # List to store datasets
datasets = []

# Iterate over all GRIB files in the directory
for filename in sorted(os.listdir(directory)):
    if filename.endswith('.grb'):
        filepath = os.path.join(directory, filename)
        ds = cfgrib.open_dataset(filepath)
        datasets.append(ds)

# Concatenate all datasets along the 'time' dimension
combined_ds = xr.concat(datasets, dim='time')

# %% load icon data

current_location = "online"
cat = intake.open_catalog(
    "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
)[current_location]

# pd.DataFrame(cat["icon_d3hp003"].describe()["user_parameters"])
icon = cat["icon_d3hp003"](zoom=7, time="P1D").to_dask()
icon = egh.attach_coords(icon)

# %% load era5 data
# list(intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"))

cat = intake.open_catalog("https://tcodata.mpimet.mpg.de/internal.yaml")
pd.DataFrame(cat["HERA5"].describe()["user_parameters"])

# %% load data
era5 = cat["HERA5"](zoom=7, time="P1D").to_dask()
era = egh.attach_coords(era5)

# egh.healpix_show(ds["ts"].sel(time="2020-05-10T00:00:00"), cmap="inferno", dpi=72);

# %%

extent = [280, 350, 58, 85]
mask = (
    (ds.lon > extent[0])
    & (ds.lon < extent[1])
    & (ds.lat > extent[2])
    & (ds.lat < extent[3])
)

gris_cells = ds.sftgif.where(mask & (ds.sftgif > 0.5), drop=True).cell

masked_ds = ds[["hus", "ua", "va", "pr"]].where(mask, drop=True)


# %% define the ivt function


def calc_ivt(ds):
    g = 9.81  # m/s^2

    """
    Calculate the integrated vapor transport (IVT) from the ICON data.
    """
    # get the specific humidity
    q = ds.hus

    # get the wind speed
    u = ds.ua
    v = ds.va

    zontrans = u * q
    mertrans = v * q

    # compute the IVT
    izontrans = zontrans.integrate("pressure") / g
    imertrans = mertrans.integrate("pressure") / g

    ivt = np.sqrt(izontrans**2 + imertrans**2)

    return ivt


# %%

ivt = calc_ivt(masked_ds)

# %% find days with IVT > 100 kg m-1 s-1

gris_max_pr = masked_ds.sel(cell=gris_cells).pr.max(dim="cell").compute() * 24 * 360
gris_mean_pr = masked_ds.sel(cell=gris_cells).pr.mean(dim="cell").compute() * 24 * 360

hp_idx = np.argwhere(gris_max_pr.values > 3)

ardays = (ivt.sel(cell=gris_cells) > 100).any(dim="cell").compute()

max_ivt = ivt.sel(cell=gris_cells).max(dim="cell").compute()


# %% plot as stereographic

for i in range(249, 255):
    fig, ax = stereogr_ax(extent, dpi=72)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    p = egh.healpix_show(ivt.isel(time=i), ax=ax, vmin=0, vmax=250)
    cbar = plt.colorbar(p, orientation="horizontal", pad=0.05)
    cbar.set_label("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=12)
    ax.annotate(
        "day " + str(i), xy=(0.5, 0.95), xycoords="axes fraction", annotation_clip=False
    )


# %%  load higher resolution


ds10 = cat["icon_d3hp003"](zoom=10, time="PT6H", time_method="inst").to_dask()
ds10 = egh.attach_coords(ds10)

mask10 = (
    (ds10.lon > extent[0])
    & (ds10.lon < extent[1])
    & (ds10.lat > extent[2])
    & (ds10.lat < extent[3])
)

ds_high = (
    ds10[["hus", "ua", "va", "pr"]]
    .sel(time=slice("2020-09-10", "2020-09-15"))
    .where(mask10, drop=True)
)

ivt10 = calc_ivt(ds_high)

# %% plot as stereographic

fig, ax = stereogr_ax(extent, shape=(4, 4), dpi=300, grid=False)

for i in range(16):
    # fig, ax = stereogr_ax(extent, dpi=72)
    # ax[i].set_extent(extent, crs=ccrs.PlateCarree())
    p = egh.healpix_show(ivt10.isel(time=i), ax=ax.flatten()[i], vmin=0, vmax=1000)
    ax.flatten()[i].set_title(ivt10.time[i].values.astype(str)[:16], fontsize=4)
    # cbar = plt.colorbar(p, orientation="horizontal", pad=0.05)

    # cbar.ax.tick_params(labelsize=4)
    # #ax.annotate('day ' + str(i), xy=(0.5, 0.95), xycoords='axes fraction', annotation_clip=False)
cbar = fig.colorbar(p, ax=ax, orientation="vertical", pad=0.05, aspect=40)
cbar.set_label("IVT (kg m$^{-1}$ s$^{-1}$)")


# %%

fig, ax = stereogr_ax(extent, shape=(4, 4), dpi=300, grid=False)

spd = 24 * 360
for i in range(16):

    p = egh.healpix_show(
        ds_high.pr.isel(time=i) * spd, ax=ax.flatten()[i], vmin=0, vmax=10, cmap=cm.rain
    )
    # cbar = plt.colorbar(p, orientation="horizontal", pad=0.05)
    # cbar.set_label("precip (kg m$^{-2}$ s$^{-1}$)", fontsize=4)
    ax.flatten()[i].set_title(ds_high.time[i].values.astype(str)[:16], fontsize=4)
cbar = fig.colorbar(p, ax=ax, orientation="vertical", pad=0.05, aspect=40)
cbar.set_label("inst. precip. (mm / day)")


# %%
