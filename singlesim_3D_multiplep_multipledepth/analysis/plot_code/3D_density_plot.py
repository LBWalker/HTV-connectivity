#2D density plot

import numpy as np
import xarray as xr

data_path_MERC = '/bucket/MitaraiU/Users/heng/GLORYS12/'

mask_bathy_path = data_path_MERC + 'GLO-MFC_001_024_mask_bathy.nc' 
mesh_mask_path = data_path_MERC + 'GLO-MFC_001_024_coordinates.nc'

data_plot = xr.open_mfdataset(mask_bathy_path)
lon = data_plot.longitude.values
lat = data_plot.latitude.values

#=======================================================
# define area of interest
lat_min = -70
lat_max   = 40
lon_min = 25
lon_max   = 125
#=======================================================


data_SA = xr.Dataset(data_plot.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max)))
lat_SA = data_SA.latitude.values
lon_SA = data_SA.longitude.values


# year = 2021
# ptcl_name = 'GLORYS12_3D_passive_Sep' + str(year) + '.zarr'

#ptcl_path = '2d_multip1_multidepth_100minterval_test_60d_30r'
ptcl_path = 'reordered_num_depth_3d_multipl2_multidepth_maxdepth_100minterval_60d_30r'
ptcl_name = ptcl_path + '.zarr' #**
#ptcls = xr.open_zarr(ptcl_name + '.zarr')

plat = xr.open_zarr(ptcl_name).lat.fillna(0)
plon = xr.open_zarr(ptcl_name).lon.fillna(0)
pz = xr.open_zarr(ptcl_name).z.fillna(0)
#ptime = xr.open_zarr(ptcl_name).time.fillna(0)

#=======================================================
# # soph
# lon_flat_30 = plon[:,0:360].values.flatten()
# lat_flat_30 = plat[:,0:360].values.flatten()

# H_30, xedges_30, yedges_30 = np.histogram2d(lon_flat_30,lat_flat_30, bins = (lon_SA[::1], lat_SA[::1])) 
# X_30, Y_30 = np.meshgrid(xedges_30, yedges_30)

# lon_flat_60 = plon[:,0:720].values.flatten()
# lat_flat_60 = plat[:,0:720].values.flatten()

# H_60, xedges_60, yedges_60 = np.histogram2d(lon_flat_60,lat_flat_60, bins = (lon_SA[::1], lat_SA[::1]))
# X_60, Y_60 = np.meshgrid(xedges_60, yedges_60)

# lon_flat_90 = plon[:,0:1080].values.flatten()
# lat_flat_90 = plat[:,0:1080].values.flatten()

# H_90, xedges_90, yedges_90 = np.histogram2d(lon_flat_90,lat_flat_90, bins = (lon_SA[::1], lat_SA[::1]))
# X_90, Y_90 = np.meshgrid(xedges_90, yedges_90)

#=======================================================
# lily

lon_flat = plon[:,:].values.flatten()
lat_flat = plat[:,:].values.flatten()

H, xedges, yedges = np.histogram2d(lon_flat,lat_flat, bins = (lon_SA[::1], lat_SA[::1])) 
X, Y = np.meshgrid(xedges, yedges)

#=======================================================
#import netCDF4 as nc

# Load landmask and bathymetry
#mask_bathy      = nc.Dataset(mask_bathy_path)
#landmask = mask_bathy['mask']  # dim=(depth=50, lat=2041, lon=4320); Land: 0, Ocean: 1

# Make negative of land mask (i.e. Land: 1, Ocean: 0)
#landmask_negative = 1 - landmask[:]

# Load coordinates file
#mesh_mask = nc.Dataset(mesh_mask_path)

#lon_vals, lat_vals = np.meshgrid(mask_bathy['longitude'], mask_bathy['latitude'])

# Centers of the gridcells formed by 4 nodes = velocity nodes + 0.5 dx
#x = mask_bathy['longitude'][:-1] + np.diff(mask_bathy['longitude']) / 2
#y = mask_bathy['latitude'][:-1] + np.diff(mask_bathy['latitude']) / 2
#lon_centers, lat_centers = np.meshgrid(x, y)

# Choose land mask at depth closest to release depth
#release_depth = 0  # [m]
#difference_array = np.abs(mesh_mask['depth'][:] - release_depth)
#index = difference_array.argmin()

# Interpolate landmask to cell centers
# -> Only cells with 4 neighboring land points will be land
#from scipy import interpolate
#fl = interpolate.RectBivariateSpline(
#    mesh_mask['latitude'], mesh_mask['longitude'], landmask_negative[index], kx=1, ky=1
#)

#l_centers = fl(lat_centers[:, 0], lon_centers[0, :])

# land when interpolated value == 1
#lmask = np.ma.masked_values(l_centers, 1)


# Plot
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 30 days

fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.Mercator()), figsize=(20,15))

ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())  # extent of map

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.BORDERS, zorder=10)
ax.add_feature(cfeature.COASTLINE, zorder=10)                  # coastlines

gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.1, linestyle='-', draw_labels=True)

#plt.scatter(plon[0:19000:1000,0].values, plat[0:19000:1000,0].values, marker = '^', c='darkorange', s=5)

#=======================================================
# soph
# # Load particle density data
# density_data = np.log10(H_30.T)

# # Plot the density data as a raster overlaying the baseline map
# im = ax.imshow(density_data, origin='lower', extent=[lon_SA[0], lon_SA[-1], lat_SA[0], lat_SA[-1]],
#          cmap='viridis', alpha= 1.0, transform=ccrs.PlateCarree())

# # Add colorbar
# plt.colorbar(im)
# ax.scatter(plon[0:19000:1000,0].values, plat[0:19000:1000,0].values, marker = '^', edgecolor='black', s=5, transform=ccrs.PlateCarree())

# #plt.savefig('plots/'+ '3D_passive_30days_Sep' + str(year)  + '.png', bbox_inches='tight', dpi=300)
# plt.savefig('plots/'+ '3d_multiplep_10day_test.png', bbox_inches='tight', dpi=300)

#=======================================================
# lily
print(f'The H is:{H}')

# Load particle density data
density_data = np.log10(H.T)

# Plot the density data as a raster overlaying the baseline map
im = ax.imshow(density_data, origin='lower', extent=[lon_SA[0], lon_SA[-1], lat_SA[0], lat_SA[-1]],
         cmap='viridis', alpha= 1.0, transform=ccrs.PlateCarree())

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Add colorbar
#plt.colorbar(im)
cbar = plt.colorbar(im, ax=ax)          # plot the colorbar
cbar.ax.tick_params(labelsize=14)       # tick font size
cbar.set_label(r'Particle density (10$^{x}$)', fontsize=16)  # caption

#ax.scatter(plon[0:19000:1000,0].values, plat[0:19000:1000,0].values, marker = '^', edgecolor='black', s=5, transform=ccrs.PlateCarree())
ax.scatter(plon[:,0].values, plat[:,0].values, marker = '^', edgecolor='black', s=5, transform=ccrs.PlateCarree()) #plot initial of all trajectories

#plt.savefig('plots/'+ '3D_passive_30days_Sep' + str(year)  + '.png', bbox_inches='tight', dpi=300)
plt.savefig('plots/' + 'densityplot_'+ ptcl_path + '.png', bbox_inches='tight', dpi=300)

#=======================================================
# # 60 days
# fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.Mercator()), figsize=(20,15))

# ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())  # extent of map

# ax.add_feature(cfeature.LAND, facecolor='lightgray')
# ax.add_feature(cfeature.BORDERS, zorder=10)
# ax.add_feature(cfeature.COASTLINE, zorder=10)                  # coastlines

# gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.1, linestyle='-', draw_labels=True)

# #plt.scatter(plon[0:19000:1000,0].values, plat[0:19000:1000,0].values, marker = '^', c='darkorange', s=5)

# # #=======================================================
# # # Load particle density data
# # density_data = np.log10(H_60.T)

# # # Plot the density data as a raster overlaying the baseline map
# # im = ax.imshow(density_data, origin='lower', extent=[lon_SA[0], lon_SA[-1], lat_SA[0], lat_SA[-1]],
# #          cmap='viridis', alpha= 1.0, transform=ccrs.PlateCarree())

# # # Add colorbar
# # plt.colorbar(im)
# # ax.scatter(plon[0:19000:1000,0].values, plat[0:19000:1000,0].values, marker = '^', edgecolor='black', s=5, transform=ccrs.PlateCarree())

# # plt.savefig('soph_plots/'+ '3D_passive_60days_Sep' + str(year)  + '.png', bbox_inches='tight', dpi=300)

# # #=======================================================
# # 90 days
# #fig = plt.figure(figsize=(20, 15))
# #ax = fig.add_subplot(111, aspect='equal')
# #cax = plt.pcolormesh(X_90, Y_90, np.log10(H_90.T), cmap='binary', vmin=3, vmax=8)

# #plt.colorbar()

# #ax = plt.pcolormesh(
# #    lon_vals,
# #    lat_vals,
# #    lmask.mask,
# #    cmap='Blues_r',
# #    alpha=.1
# #)

# #plt.scatter(plon[0:19000:1000,0].values, plat[0:19000:1000,0].values, marker = '^', c='darkorange', s=5)

# #plt.xlim(lon_min, lon_max)
# #plt.ylim(lat_min, lat_max)

# #plt.savefig('soph_plots/'+ '3D_passive_90days_Sep' + str(year)  + '.png', bbox_inches='tight', dpi=300)

# fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.Mercator()), figsize=(20,15))

# ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())  # extent of map

# ax.add_feature(cfeature.LAND, facecolor='lightgray')
# ax.add_feature(cfeature.BORDERS, zorder=10)
# ax.add_feature(cfeature.COASTLINE, zorder=10)                  # coastlines

# gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.1, linestyle='-', draw_labels=True)

# # Load particle density data
# density_data = np.log10(H_90.T)

# # Plot the density data as a raster overlaying the baseline map
# im = ax.imshow(density_data, origin='lower', extent=[lon_SA[0], lon_SA[-1], lat_SA[0], lat_SA[-1]],
#          cmap='viridis', alpha= 1.0, transform=ccrs.PlateCarree())

# # Add colorbar
# plt.colorbar(im)
# ax.scatter(plon[0:19000:1000,0].values, plat[0:19000:1000,0].values, marker = '^', edgecolor='black', s=5, transform=ccrs.PlateCarree())

# plt.savefig('soph_plots/'+ '3D_passive_90days_Sep' + str(year)  + '.png', bbox_inches='tight', dpi=300)

