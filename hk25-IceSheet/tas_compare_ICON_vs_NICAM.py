#%%
import cartopy.crs as ccrs
import cartopy.feature as cf
import intake
import matplotlib.pyplot as plt
import numpy as np
from easygems import healpix as egh

# 打开数据目录和变量配置
cat = intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml")["EU"]

# 参数设置
extent = [285, 350, 58, 85]
extent_data = [275, 360, 50, 90]
var = "tas"

# 读取ICON数据，时间均值（日均）
ds_icon = cat["icon_d3hp003"](chunks={}, time='P1D', zoom=9).to_dask().sel(time=slice("2020-03", "2021-02")).mean("time")
ds_icon = egh.attach_coords(ds_icon)

# 读取NICAM数据，时间均值（3小时步长）
ds_nicam = cat["nicam_gl11"](chunks={}, time='PT3H', zoom=9).to_dask().sel(time=slice("2020-03", "2021-02")).mean("time")

# 使用ICON的陆冰掩码
sftgif_mask = ds_icon["sftgif"] > 0
if "time" in ds_icon[var].dims and "time" not in ds_icon["sftgif"].dims:
    sftgif_mask = sftgif_mask.broadcast_like(ds_icon[var])

# 地理范围掩码
geo_mask = (
    (ds_icon.lon > extent_data[0]) & (ds_icon.lon < extent_data[1]) &
    (ds_icon.lat > extent_data[2]) & (ds_icon.lat < extent_data[3])
)

# 联合掩码（陆冰 + 地理范围）
combined_mask = sftgif_mask & geo_mask
combined_mask_np = combined_mask.compute()

# 取温度并转摄氏度，同时应用掩码
tas_icon = ds_icon[var].where(combined_mask_np, drop=True) - 273.15
tas_nicam = ds_nicam[var].where(combined_mask_np, drop=True) - 273.15

# 计算温差（ICON - NICAM）
temp_diff = tas_icon - tas_nicam

# 提取海拔高度（单位：米），并应用同样的掩码
orog_icon = ds_icon["orog"].where(combined_mask_np, drop=True)
orog_nicam = ds_nicam["orog"].where(combined_mask_np, drop=True)

# 海拔差
orog_diff = orog_icon - orog_nicam

#%% 2×2 Pattern 图像绘制并保存

fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=100,
                         subplot_kw={"projection": ccrs.Stereographic(70, -42)})
fig.patch.set_facecolor("#151515")

titles = [
    "ICON Temperature (°C)",
    "NICAM Temperature (°C)",
    "Temperature Difference (ICON - NICAM)",
    "Elevation Difference (ICON - NICAM)"
]
datasets = [tas_icon, tas_nicam, temp_diff, orog_diff]
cmaps = ["coolwarm", "coolwarm", "bwr", "terrain"]
vmins = [-20, -20, -10, -1500]
vmaxs = [20, 20, 10, 1500]

for ax, data, title, cmap, vmin, vmax in zip(axes.flat, datasets, titles, cmaps, vmins, vmaxs):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(lw=0.6, color="gray")
    ax.add_feature(cf.BORDERS, linewidth=0.3, edgecolor="gray")

    im = egh.healpix_show(data, vmin=vmin, vmax=vmax, cmap=cmap, ax=ax)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True

    ax.set_aspect('auto')

    # 添加 colorbar（不再设置标签为 title）
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, aspect=40)
    cbar.set_label("Value", color="white")
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_xticklabels(), color='white')

    # 设置每个子图标题
    ax.set_title(title, fontsize=14, color="white", pad=12)

fig.suptitle("ICON vs NICAM Temperature and Elevation Patterns", fontsize=20, color="white", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("icon_nicam_patterns.png", facecolor=fig.get_facecolor())
plt.show()

#%% 散点图：温度差 vs 海拔差，并标红高海拔点

# 展平数组，去除 NaN
valid_mask = (~np.isnan(temp_diff)) & (~np.isnan(orog_diff)) & (~np.isnan(orog_icon))
temp_diff_flat = temp_diff.values[valid_mask]
orog_diff_flat = orog_diff.values[valid_mask]
orog_icon_flat = orog_icon.values[valid_mask]

# 设置颜色条件：海拔 > 1500 为红，其它为灰
colors = np.where(orog_icon_flat > 1500, "red", "gray")

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(orog_diff_flat, temp_diff_flat, c=colors, s=5, alpha=0.7, edgecolors="k", linewidths=0.3)
plt.axhline(0, color='black', lw=0.8, linestyle="--")
plt.axvline(0, color='black', lw=0.8, linestyle="--")
plt.xlabel("Elevation Difference (ICON - NICAM) [m]")
plt.ylabel("Temperature Difference (ICON - NICAM) [°C]")
plt.title("Temperature vs Elevation Difference\n(Red = ICON Elevation > 1500)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("temp_vs_elevation_scatter.png", dpi=300)
plt.show()

# %%
