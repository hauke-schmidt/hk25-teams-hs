# %% load packages
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
pd.DataFrame(cat["icon_ngc4008"].describe()["user_parameters"])

# %% load data
ds = cat["icon_d3hp003"](zoom=7, time="P1D").to_dask()
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

masked_ds = ds.isel(time =10).where(mask, drop=True)

# %%

def calc_ivt(ds):
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
    izontrans = zontrans.integrate('pressure')
    imertrans = mertrans.integrate('pressure')

    ivt = np.sqrt(izontrans**2 + imertrans**2)

    return ivt


# %%

ivt = calc_ivt(masked_ds)
# %%

fig, ax = stereogr_ax(extent, dpi=600)
ax.set_extent(extent, crs=ccrs.PlateCarree())
# egh.healpix_show(ivt, ax = ax)

_, _, nx, ny = np.array(ax.bbox.bounds, dtype=int) * 2
xlims = ax.get_xlim()
ylims = ax.get_ylim()

data = egh.healpix_resample(
    ivt, xlims, ylims, nx, ny, ax.projection
)

ax.imshow(data, extent = xlims + ylims, origin = 'lower')

# %%
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, dpi=300)
ax.add_feature(cfeature.COASTLINE)
ax.set_extent(extent, crs=ccrs.PlateCarree())

_, _, nx, ny = np.array(ax.bbox.bounds, dtype=int) * 2
xlims = ax.get_xlim()
ylims = ax.get_ylim()

test = egh.healpix_resample(
    ivt, xlims, ylims, nx, ny, ax.projection
)

egh.healpix_show(ivt, dpi=300, ax = ax)
# ax.contourf(data.x, data.y, data)




# %%
fig, ax = stereogr_ax(extent, dpi=300)
ax.set_extent(extent, crs=ccrs.PlateCarree())
# ax.add_feature(cfeature.LAND)
# ax.add_feature(cfeature.OCEAN)

_, _, nx, ny = np.array(ax.bbox.bounds, dtype=int) * 2
xlims = ax.get_xlim()
ylims = ax.get_ylim()

data = egh.healpix_resample(
    pr.sel(time="2025-01-01"), xlims, ylims, nx, ny, ax.projection
)

ax.contourf(data.x, data.y, data)
