# %%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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


def stereogr_ax(extent, set_boundary=True, shape=None, grid=True, **kwargs):
    midlon = (extent[0] + extent[1]) / 2
    midlat = (extent[2] + extent[3]) / 2

    projection = ccrs.Stereographic(central_longitude=midlon, central_latitude=midlat)

    if shape is None:
        shape = (1, 1)  # Default to a single subplot if shape is not provided

    fig, axes = plt.subplots(
        *shape, subplot_kw={"projection": projection}, dpi=kwargs.get("dpi", 300)
    )

    # Ensure axes is always a 2D array for consistency
    if shape == (1, 1):
        axes = np.array([[axes]])
    elif shape[0] == 1 or shape[1] == 1:
        axes = np.atleast_2d(axes)

    boundary = make_boundary_path(*extent)

    for ax in axes.flat:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, lw=0.5)

        if set_boundary:
            ax.set_boundary(boundary, transform=ccrs.PlateCarree())

        if grid:
            ax.gridlines(
                ylocs=np.arange(-90, 91, 10),
                xlocs=np.arange(-180, 181, 10),
            )

    return fig, axes
