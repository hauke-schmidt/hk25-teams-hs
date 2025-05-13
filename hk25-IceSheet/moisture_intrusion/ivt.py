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

