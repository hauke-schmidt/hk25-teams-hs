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
from cmocean import cm
from plot_fun import stereogr_ax

warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # don't warn us about future package conflicts

# %% define relevant extent for analysis

extent = [280, 350, 58, 85]

spd = 24 * 360  # seconds per day to convert kg m-2 s-1 to mm / day

# %% define some functions

# %%


def filter_region(ds):
    mask = (
        (ds.lon > extent[0])
        & (ds.lon < extent[1])
        & (ds.lat > extent[2])
        & (ds.lat < extent[3])
    )
    return ds.where(mask, drop=True)


def filter_gris(ds):
    gris_mask = ds.sftgif > 0.5
    region_mask = (
        (ds.lon > extent[0])
        & (ds.lon < extent[1])
        & (ds.lat > extent[2])
        & (ds.lat < extent[3])
    )
    mask = gris_mask & region_mask
    return ds.where(mask, drop=True)


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


# %% load data
# list(intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"))

current_location = "online"
cat = intake.open_catalog(
    "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
)[current_location]
pd.DataFrame(cat["icon_d3hp003"].describe()["user_parameters"])
pd.DataFrame(cat["IR_IMERG"].describe()["user_parameters"])


# %% load data
vars = list(cat["icon_d3hp003"](zoom=0, time="P1D").to_dask().data_vars)
varlist = ["hus", "ua", "va", "pr", "sftgif"]
droplist = [var for var in varlist if var not in varlist]

ds5P1D = cat["icon_d3hp003"](zoom=5, time="P1D", drop_variables=droplist).to_dask()
ds5P1D = egh.attach_coords(ds5P1D)
ds5P1D_gris = filter_gris(ds5P1D)

# ds10PT6H = cat["icon_d3hp003"](zoom=10, time="PT6H", time_method="inst", drop_variables = droplist).to_dask()
# ds10PT6H = egh.attach_coords(ds10PT6H)

ds10PT3H = cat["icon_d3hp003"](
    zoom=10, time="PT3H", time_method="mean", drop_variables=droplist
).to_dask()
ds10PT3H = egh.attach_coords(ds10PT3H)

# %% change catalog to load era5 data
cat = intake.open_catalog("https://tcodata.mpimet.mpg.de/internal.yaml")
pd.DataFrame(cat["HERA5"].describe()["user_parameters"])

# %% load data
era5 = cat["HERA5"](zoom=7, time="P1D").to_dask()
era = egh.attach_coords(era5)

# %% get precipitation maxima over GRIS

fig, ax = plt.subplots()

ax.plot(
    ds5P1D_gris.pr.max(dim="cell").compute() * 24 * 360,
    label="max",
    color="blue",
)
ax.set_ylabel("max gris precip (mm / day)", color="blue")
ax.set_xlabel("time (days)")
rax = ax.twinx()
rax.plot(
    ds5P1D_gris.pr.mean(dim="cell").compute() * 24 * 360,
    label="mean",
    color="red",
)
rax.set_ylabel("mean gris precip (mm / day)", color="red")
ax.set_xlim(35, 45)
xticks = np.arange(35, 45, 1)
ax.set_xticks(
    xticks, labels=[str(t)[:10] for t in ds5P1D_gris.time[xticks].values], rotation=90
)


# egh.healpix_show(ds5["ts"].sel(time="2020-05-10T00:00:00"), cmap="inferno", dpi=72);

# %%

ivt5P1D = calc_ivt(ds5P1D)

# %% find days with IVT > 100 kg m-1 s-1

# hp_idx = np.argwhere(gris_max_pr.values > 3)

# ardays = (ivt.sel(cell=gris_cells) > 100).any(dim="cell").compute()

# max_ivt = ivt.sel(cell=gris_cells).max(dim="cell").compute()


# %% plot low res februar event as stereographic

for i in range(35, 36):
    fig, ax = stereogr_ax(extent, shape=(1, 2), dpi=300)
    p0 = egh.healpix_show(ivt5P1D.isel(time=i), ax=ax[0, 0], vmin=0, vmax=250)
    p1 = egh.healpix_show(ds5P1D.prw.isel(time=i), ax=ax[0, 1], vmin=0, vmax=20)
    cbar0 = plt.colorbar(p0, orientation="horizontal", pad=0.05, ax=ax[0, 0])
    cbar0.set_label("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=12)
    cbar1 = plt.colorbar(p1, orientation="horizontal", pad=0.05, ax=ax[0, 1])
    cbar1.set_label("prw (kg m$^{-2}$)", fontsize=12)
    fig.suptitle(ds5P1D.time[i].values.astype(str)[:10], fontsize=12, y=0.75)
    # fig.subplots_adjust(top = 0.99)
    fig.savefig(
        "figures/icon_5P1D_ivt_prw" + str(ds5P1D.time[i])[:10] + ".png", dpi=300
    )


# %% plot high res februar event as stereographic
pr_high = filter_region(ds10PT3H.pr.sel(time=slice("2020-02-09", "2020-02-14"))) * spd
prw_high = filter_region(ds10PT3H.prw.sel(time=slice("2020-02-09", "2020-02-14")))
figtitle = "icon_d3hp003 z10 PT3H mean "
for i in range(pr_high.time.size):
    time = pr_high.time[i].values.astype(str)[:16]
    fig, ax = stereogr_ax(extent, shape=(1, 2), dpi=300)
    p0 = egh.healpix_show(
        pr_high.isel(time=i), ax=ax[0, 0], vmin=0, vmax=10, cmap=cm.rain
    )
    p1 = egh.healpix_show(prw_high.isel(time=i), ax=ax[0, 1], vmin=0, vmax=20)
    cbar0 = plt.colorbar(p0, orientation="horizontal", pad=0.05, ax=ax[0, 0])
    cbar0.set_label("precip (mm/day)", fontsize=12)
    cbar1 = plt.colorbar(p1, orientation="horizontal", pad=0.05, ax=ax[0, 1])
    cbar1.set_label("prw (kg m$^{-2}$)", fontsize=12)
    fig.suptitle(figtitle + "\n" + time, fontsize=12, y=0.75)
    fig.savefig(
        "figures/icon_10PT3H_pr_prw_" + time + ".png", dpi=300, bbox_inches="tight"
    )


# %%

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
plt.savefig("ivt.png", dpi=300, bbox_inches="tight")


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
plt.savefig("precip.png", dpi=300, bbox_inches="tight")


# %%

dsP1D10 = cat["icon_d3hp003"](zoom=10, time="P1D").to_dask()
dsP1D10 = egh.attach_coords(dsP1D10)
mask10 = (
    (dsP1D10.lon > extent[0])
    & (dsP1D10.lon < extent[1])
    & (dsP1D10.lat > extent[2])
    & (dsP1D10.lat < extent[3])
)

dsP1D10 = dsP1D10.pr.where(mask10, drop=True).sel(
    time=slice("2020-09-10", "2020-09-15")
)

# %%
fig, ax = stereogr_ax(extent, shape=(1, 3), dpi=300, grid=False)
spd = 24 * 360

for i in range(1, 4):

    p = egh.healpix_show(
        dsP1D10.isel(time=i) * spd,
        ax=ax.flatten()[i - 1],
        vmin=0,
        vmax=10,
        cmap=cm.rain,
    )
    # cbar = plt.colorbar(p, orientation="horizontal", pad=0.05)
    # cbar.set_label("precip (kg m$^{-2}$ s$^{-1}$)", fontsize=4)
    ax.flatten()[i - 1].set_title(dsP1D10.time[i].values.astype(str)[:12], fontsize=4)
cbar = fig.colorbar(p, ax=ax, orientation="horizontal", pad=0.05, aspect=40)
cbar.set_label("daily mean precip. (mm / day)")
plt.savefig("precip.png", dpi=300, bbox_inches="tight")

# %%
