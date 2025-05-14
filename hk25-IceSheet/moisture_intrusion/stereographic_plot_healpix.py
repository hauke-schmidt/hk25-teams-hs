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


# %% make boundary path

# https://github.com/SciTools/cartopy/issues/1831


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


# #%% check the healpix resample function

# # https://github.com/mpimet/easygems/blob/main/easygems/healpix/__init__.py

# def create_geoaxis(add_coastlines=True, **subplot_kw):
#     """Convenience function to create a figure with a default map projection."""
#     if "projection" not in subplot_kw:
#         subplot_kw["projection"] = ccrs.Robinson(central_longitude=-135.58)

#     _, ax = plt.subplots(subplot_kw=subplot_kw)
#     ax.set_global()

#     if add_coastlines:
#         ax.coastlines(color="#333333", linewidth=plt.rcParams["grid.linewidth"])

#     return ax

# ax = create_geoaxis(add_coastlines=False)

# # subplot_kw = {}
# # subplot_kw["projection"] = ccrs.Robinson(central_longitude=-135.58)
# # _, ax = plt.subplots(subplot_kw=subplot_kw)

# xlims = ax.get_xlim()
# ylims = ax.get_ylim()

# _, _, nx, ny = np.array(ax.bbox.bounds, dtype=int)

# dx = (xlims[1] - xlims[0]) / nx
# dy = (ylims[1] - ylims[0]) / ny
# xvals = np.linspace(xlims[0] + dx / 2, xlims[1] - dx / 2, nx)
# yvals = np.linspace(ylims[0] + dy / 2, ylims[1] - dy / 2, ny)
# xvals2, yvals2 = np.meshgrid(xvals, yvals)

# src_crs = ax.projection

# latlon = ccrs.PlateCarree().transform_points(
#     src_crs, xvals2, yvals2, np.zeros_like(xvals2)
# )

# valid = np.all(np.isfinite(latlon), axis=-1)
# points = latlon[valid].T

# res = np.full(latlon.shape[:-1], np.nan)

# var = ds.pr.sel(time="2025-01-01")
# pix = healpix.ang2pix(
#             egh.get_nside(var),
#             theta=points[0],
#             phi=points[1],
#             nest=True,
#             lonlat=True,
#         )

# res[valid] = var[pix]

# %%
# list(intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"))

current_location = "online"
cat = intake.open_catalog(
    "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
)[current_location]
pd.DataFrame(cat["icon_ngc4008"].describe()["user_parameters"])

# %% load data
ds = cat["icon_ngc4008"](zoom=6, time="P1D").to_dask()
ds = egh.attach_coords(ds)


# %%

extent = [280, 350, 58, 85]
mask = (
    (ds.lon > extent[0])
    & (ds.lon < extent[1])
    & (ds.lat > extent[2])
    & (ds.lat < extent[3])
)
pr = ds.pr.where(mask, drop = True)

# %% create a plot

fig, ax = stereogr_ax(extent, dpi=300)
ax.add_feature(cfeature.COASTLINE)
ax.set_extent(extent, crs=ccrs.PlateCarree())
egh.healpix_show(pr.sel(time="2025-01-01"), ax=ax)
# ax.add_feature(cfeature.LAND)
# ax.add_feature(cfeature.OCEAN)

# _, _, nx, ny = np.array(ax.bbox.bounds, dtype=int) * 2
# xlims = ax.get_xlim()
# ylims = ax.get_ylim()

# data = egh.healpix_resample(
#     pr.sel(time="2025-01-01"), xlims, ylims, nx, ny, ax.projection
# )

# this is not correct, not sure why.
# ax.contourf(data.x, data.y, data)
