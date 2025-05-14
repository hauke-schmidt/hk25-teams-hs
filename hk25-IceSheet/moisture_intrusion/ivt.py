# %% load packages
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


warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # don't warn us about future package conflicts


# %% define stereographic plot 


def make_boundary_path(minlon, maxlon, minlat, maxlat, n=50):
    """
    return a matplotlib Path whose points are a lon-lat box given by
    the input parameters
    """

    boundary_path = []
    # North (E->W)
    edge = [np.linspace(minlon, maxlon, n), np.full(n, maxlat)]
    boundary_path = np.stack([*edge], axis=0).T

    # West (N->S)
    edge = [np.full(n, maxlon), np.linspace(maxlat, minlat, n)]
    boundary_path = np.concatenate([boundary_path, np.stack([*edge], axis=0).T], axis=0)

    # South (W->E)
    edge = [np.linspace(maxlon, minlon, n), np.full(n, minlat)]
    boundary_path = np.concatenate([boundary_path, np.stack([*edge], axis=0).T], axis=0)

    # East (S->N)
    edge = [np.full(n, minlon), np.linspace(minlat, maxlat, n)]
    boundary_path = np.concatenate([boundary_path, np.stack([*edge], axis=0).T], axis=0)

    boundary_path = mpath.Path(boundary_path)

    return boundary_path


def stereogr_ax(extent, set_boundary=True, **kwargs):

    midlon = (extent[0] + extent[1]) / 2
    midlat = (extent[2] + extent[3]) / 2

    projection = ccrs.Stereographic(central_longitude=midlon, central_latitude=midlat)
    fig, ax = plt.subplots(
        subplot_kw={"projection": projection}, dpi=kwargs.get("dpi", 300)
    )

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)

    boundary = make_boundary_path(*extent)

    if set_boundary:
        ax.set_boundary(boundary, transform=ccrs.PlateCarree())

    g = ax.gridlines(
        ylocs=np.arange(-90, 91, 10),
        xlocs=np.arange(-180, 181, 10),
    )

    ax.set_title(str(extent))
    return fig, ax

# %%
# list(intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"))

current_location = "online"
cat = intake.open_catalog(
    "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
)[current_location]
pd.DataFrame(cat["icon_d3hp003"].describe()["user_parameters"])

# %% load data
ds = cat["icon_d3hp003"](zoom=5, time="P1D").to_dask()
ds = egh.attach_coords(ds)

# egh.healpix_show(ds["ts"].sel(time="2020-05-10T00:00:00"), cmap="inferno", dpi=72);

# %%

extent = [280, 350, 58, 85]
mask = (
    (ds.lon > extent[0])
    & (ds.lon < extent[1])
    & (ds.lat > extent[2])
    & (ds.lat < extent[3])
)

gris_cells = ds.sftgif.where(mask & (ds.sftgif > 0.5), drop = True).cell

masked_ds = ds[['hus', 'ua', 'va', 'pr']].where(mask, drop=True)


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
    izontrans = zontrans.integrate('pressure')/g
    imertrans = mertrans.integrate('pressure')/g

    ivt = np.sqrt(izontrans**2 + imertrans**2)

    return ivt


# %%

ivt = calc_ivt(masked_ds)

#%% find days with IVT > 100 kg m-1 s-1

gris_max_pr = masked_ds.sel(cell = gris_cells).pr.max(dim="cell").compute() * 24 * 360
hp_idx = np.argwhere(gris_max_pr.values >3)

ardays = (ivt.sel(cell = gris_cells) > 100).any(dim="cell").compute()

max_ivt = ivt.sel(cell = gris_cells).max(dim="cell").compute()



# %% plot as stereographic

for i in range(249,255):
    fig, ax = stereogr_ax(extent, dpi=72)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    p = egh.healpix_show(ivt.isel(time = i), ax = ax, vmin = 0, vmax = 250)
    cbar = plt.colorbar(p, orientation="horizontal", pad=0.05)
    cbar.set_label("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=12)
    ax.annotate('day ' + str(i), xy=(0.5, 0.95), xycoords='axes fraction', annotation_clip=False)


# %%  load higher resolution

ds10 = cat["icon_d3hp003"](zoom=10, time="PT6H", time_method = 'inst').to_dask()
ds10 = egh.attach_coords(ds10)
ds_high = ds10[['hus', 'ua', 'va', 'pr']].sel(time = slice('2020-09-10', '2020-09-15')).where(mask, drop=True)

ivt10 = calc_ivt(ds_high)

# %% plot as stereographic

for i in range(24):
    fig, ax = stereogr_ax(extent, dpi=72)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    p = egh.healpix_show(ivt10.isel(time = i), ax = ax, vmin = 0, vmax = 250)
    cbar = plt.colorbar(p, orientation="horizontal", pad=0.05)
    cbar.set_label("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=12)
    ax.annotate('day ' + str(i), xy=(0.5, 0.95), xycoords='axes fraction', annotation_clip=False)


# %%
