#!/usr/bin/env python
# coding: utf-8

# # Code for splitting parcels simulations output into individual locations, importing dataframe data of simulation metrics (final particle location, age, etc.) from another python file output, and plotting Probability Distribution Functions for the vent outputs. 

# This will give outputs:
# plots of kde probability densities around each vent site 
# 2 matricies: one with the kde probability densities around the vents, and one giving the prob mass (actual probability) at a vent site
# Both calculated directly into matricies, instead of dataframes 

# Record time taken for simulations
import time
start_time = time.time()
import sys
import json
import numpy as np
import netCDF4 as nc
import pandas as pd
import networkx as nx
import matplotlib as mpl 
import xarray as xr
#%matplotlib notebook
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.cm as cm 
from random import randrange
import seaborn as sns
import math as math
#import adjustText
#import adjustText
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import seaborn as sns


# In[2]:


# Metadata - some values redundant, all stated for ease of copy and paste from the simulation input

# ===============================================================================
# Set up model runs:

# model dimensions: 2D vs 3D
ndim = 3     # 2 for 2D simulations; 3 for 3D simulations

# time
year_start = 2022
year_end = 2023
run_period = 150  # days; Total runtime of simulation
time_step = 2   # hours; Time increment of particle tracking
release_period = 75 # days; consecutive days of release

# particle release
release_depth = 'max_depth'  # Release depth; for 2D: 100-800m (100m-increments); for 3D: max_depth
bouncing = True  # Use displacement functions to prevent particles getting stuck
multi_particles = True # If 'True' uniform release of 'num_pts' particles within 1km radius of HTV coordinates
num_pts = 100
continuous_release = True # If 'True' release particles continuously, else release particles all at once
particle = 'passive'  # if 'passive': no vertical migration; if 'active': vertical migration

incremental_depth_release = True # If 'True' release particles at multiple depths over set increments
depth_interval = 100 #meters

# # output
# #output_name = 'reordered_num_depth_3d_multipl2_multidepth_maxdepth_100minterval_10d_5r'
# output_name = (f"reordered_num_depth_{ndim}d_multipl{num_pts}_multidepth_maxdepth_{depth_interval}minterval_{run_period}d_{release_period}r")

#output_name = 'reordered_num_depth_3d_multipl2_multidepth_maxdepth_100minterval_10d_5r'

if incremental_depth_release==True:
    output_name =(f"reordered_num_depth_{ndim}d_multipl{num_pts}_multidepth_maxdepth_{depth_interval}minterval_{run_period}d_{release_period}r_accuratedepths_repeat_printdepths_modifyreleasedepthscheck")
else:
    output_name =(f"reordered_num_depth_{ndim}d_multipl{num_pts}_singledepth_maxdepth_{run_period}d_{release_period}r")
#output_name = 'soph_TEST_14d_dt2h_cont_release_4d_100part_Sep2022'
# ===============================================================================

#if in current folder just leave blank ''
file_location = '../../simulations/output/'

n_locations = 13 # number of release locations

############################
print('Imported parameters')
############################

# In[3]:


# Open simulation dataset

ds = xr.open_zarr(file_location + output_name + '.zarr')

#extract vent data from .json
with open(file_location + "sim_metadata_tmp" + output_name + ".json", "r") as f:
    metadata = json.load(f)

n_depths_per_location = metadata["n_depths_per_site"]
n_depths_per_location = np.array(n_depths_per_location) #convert to np.array
vent_lon = metadata["vent_lon"]
vent_lat = metadata["vent_lat"]

############################
print(f"n_depths_per_site: {n_depths_per_location}") 
print('extracted vent metadata (number of depths per site)')
############################

#===================================================================#
#========================= BEGIN SPLIT =============================#
#===================================================================#

## Split with separated releases ## 
print("""Begin splitting particles into their respective releases\n""", flush = True)

############################
print('computing number of particles per location')
############################
# Compute number of particles per location
particles_per_location = n_depths_per_location * num_pts

# Sanity check
total_particles_per_release = np.sum(particles_per_location)
print(f"Total particles per release for all locations: {total_particles_per_release}")
print(f"Number of trajectories tracked/total no. of particles in simulation: {ds.dims['trajectory']}")
n_releases = ds.dims['trajectory'] // total_particles_per_release

print(f"Detected {n_releases} releases")
#sys.exit()

# Allocate list to hold output
split_by_location = []
loc_end_offset = []

offset = 0  # trajectory index offset
##location_release_offset = 0

# Set chunk size
ds = ds.chunk({"trajectory": total_particles_per_release})

for loc_idx in range(n_locations):
    loc_ds_list = []
    #location_release_offset = first loc_end of previous release 
    if loc_idx == 0:
        release_start = 0
    else:
        release_start = loc_end_offset[(loc_idx-1)*n_releases]  #release_idx * total_particles_per_release, +1 so indexes don't overlap
    ###print(f"release start of loc_idx loop {loc_idx} is {release_start}")
    for release_idx in range(n_releases):
        if release_idx == 0:
           loc_start = release_start
        else:
           loc_start = release_start + offset
           
        loc_end = loc_start + particles_per_location[loc_idx]
        
        ###print(f"{loc_idx}th location for {release_idx}th release: start index {loc_start} end index {loc_end}")
        loc_end_offset.append(loc_end)
        # select the trajectories for this location in this release
        ds_loc_release = ds.isel(trajectory=slice(loc_start, loc_end))


        #================
        # Stop trajectories from blowing up
        #ds_loc_release = ds_loc_release.transpose("trajectory", "obs") # Fix dimension ordering after slicing

        # Reset trajectory coordinate to match local indices (0 to N-1)
        n_traj = loc_end - loc_start
        ds_loc_release = ds_loc_release.assign_coords(trajectory=np.arange(n_traj))

        #================
        loc_ds_list.append(ds_loc_release)
        
        #offset += particles_per_location[loc_idx]  # prepare for next location in same release
        offset = (release_idx+1) * total_particles_per_release #11 so indexes don't overlap
        #release_start = release_idx * total_particles_per_release
        #release_start = release_idx * total_particles_per_release
        ###print(f"offset now {offset}")
        ###print(loc_end_offset)

    # concatenate all releases for this location along a new dimension
    ds_loc_all_releases = xr.concat(loc_ds_list, dim='release')
    # Add lat/lon metadata here:
    ds_loc_all_releases.attrs["site_index"] = loc_idx
    ds_loc_all_releases.attrs["vent_lon"] = vent_lon[loc_idx]
    ds_loc_all_releases.attrs["vent_lat"] = vent_lat[loc_idx]
    ds_loc_all_releases.attrs["n_depths"] = n_depths_per_location[loc_idx]

    # Remove unwanted attributes that don't carry over entire dataset
    for key in ["site_index", "vent_lon", "vent_lat", "n_depths"]:
        ds_loc_all_releases.attrs.pop(key, None)

    split_by_location.append(ds_loc_all_releases)
    

print("Split complete.", flush = True)
print("\n\n")

#===================================================================#
#========================= COMBINE LOCATIONS =======================#
#===================================================================#
print('combine all locations into a single dataset')

# Optionally combine all locations into a single dataset
ds_all_locations = xr.concat(split_by_location, dim='location')

# Assign useful coordinates
ds_all_locations = ds_all_locations.assign_coords(
    vent_lon=("location", vent_lon),
    vent_lat=("location", vent_lat),
    site_index=("location", np.arange(n_locations)),
    n_depths=("location", n_depths_per_location)
)

print("Combined dataset shape:")
print(ds_all_locations)

#===================================================================#
#==================== FLATTEN DIMENSIONS ===========================#
#===================================================================#

# === Flatten 'release' and 'trajectory' into a single 'trajectory' dimension ===
print("flatten 'release' and 'trajectory' into a single 'trajectory' dimension")

# Stack 'release' and 'trajectory' into a new combined dimension
ds_flat = ds_all_locations.stack(trajectory_flat=("release", "trajectory"))

# Drop the old 'trajectory' coordinate if it exists
if "trajectory" in ds_flat.coords:
    ds_flat = ds_flat.drop_vars("trajectory")

# Rename the stacked dimension to just 'trajectory'
ds_flat = ds_flat.rename_dims({"trajectory_flat": "trajectory"})

# Drop the 'trajectory_flat' coordinate if it's still there
if "trajectory_flat" in ds_flat.coords:
    ds_flat = ds_flat.drop_vars("trajectory_flat")

# Assign new integer-based trajectory coordinate (optional but keeps clean)
n_traj = ds_flat.dims["trajectory"]
ds_flat = ds_flat.assign_coords(trajectory=np.arange(n_traj))

# === Done: Now have (location, trajectory, obs) ===

print('flattened output into (location, trajectory, obs)')
print('Flattened dataset:')
print(ds_flat, flush = True)

#===================================================================#
#=================== IMPORT SITES FOR LABELS =======================#
#===================================================================#

print('import names of vent sites for plotting labels')
excel_file_path = '/flash/MitaraiU/oceanparcels/htv/lily/indianocean_accurate_perez2021_fields.xlsx' #indianocean_perez2021_fields.xlsx' 
# vent_sites = pd.read_excel(excel_file_path)
# vent_sites = htv_data.set_index('Vent Name')
# htv_data

vent_sites = pd.read_excel(excel_file_path,index_col = 'Vent Name')
vent_sites['name']=vent_sites.index.str.replace(" ", "_")
v_name=vent_sites.name[0]
vent_sites


# In[7]:

####### whisker plots for particle age that relies on a .csv dataframe (maybe this is from Otis ########## 

# df_allparticles = pd.read_csv("trash/particles_dataframe" + output_name + ".csv")

# # Drops rows of NaNs that allowed all vent sites to have the same number of columns, making shallower sites have extra rows.
# df_allparticles = df_allparticles.dropna()
# # Round 'age_days' to nearest integer
# df_allparticles['age_days'] = df_allparticles['age_days'].round().astype(int)
# df_allparticles


# # In[8]:


# # plot whisker plots 
# #tips = sns.load_dataset('df_yr')
# fig, ax = plt.subplots(
#     1, 1, figsize=(20, 14))
# ax = sns.boxplot(x="release_site", y="age_days", data=df_allparticles,showfliers = False)
# ax.set_ylabel('LDD (days)',fontsize=24)
# ax.set_xlabel('Source Site',fontsize=24)

# ax.tick_params(axis='x',labelsize=22)
# ax.tick_params(axis='y',labelsize=22)
# #plt.savefig('figures/box_LDD.pdf')

# plt.show()


#===================================================================#
#================= CALCULATE AND INTEGRATE PDFS ====================#
#===================================================================#
############################
print('Integrate PDFs')
############################



#### Largely adapted from Otis Brunner's great work https://github.com/otistwo/hydrothermal_thesis/blob/main/PARCELS/Copepod_posthoc.ipynb ####



# ====================== files ======================= #
data_path_glorys = '/bucket/MitaraiU/Users/heng/parcels_workshop/ogcm_data/'
hgr = xr.open_dataset(data_path_glorys + 'glorys12v1_nwp_trmd_hgr.nc') #coords
zgr = xr.open_dataset(data_path_glorys + 'glorys12v1_nwp_trmd_zgr.nc') #depth data

# =============== Get bathymetry files =============== #
data_path_MERC = '/bucket/MitaraiU/Users/heng/GLORYS12/'
# Load landmask and bathymetry
mask_bathy_path = data_path_MERC + 'GLO-MFC_001_024_mask_bathy.nc' 
mask_bathy      = nc.Dataset(mask_bathy_path)
landmask = mask_bathy['mask']  # dim=(depth=50, lat=2041, lon=4320); Land: 0, Ocean: 1
# Make negative of land mask (i.e. Land: 1, Ocean: 0)
landmask_negative = 1 - landmask[:]
# Load coordinates file
mesh_mask_path = data_path_MERC + 'GLO-MFC_001_024_coordinates.nc' 
mesh_mask      = nc.Dataset(mesh_mask_path)

seafloor = xr.open_mfdataset(mask_bathy_path)
# ====================================================== #

# Define function to calculate Haversine distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0  # radius of the Earth in meters
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d

debug = False   
def get_prob_mass_in_box(kde, point):#, debug=False):
    lon, lat = point
    box_size = 100.0  # meters

    # degrees corresponding to 100 m at this latitude
    lat_diff = box_size / haversine(lat, lon, lat + 1, lon) # m per 1 degree of latitude 
    lon_diff = box_size / haversine(lat, lon, lat, lon + 1) # m per 1 degree of longitude

    # bounds must match KDE order: [lon, lat]
    low  = [lon - lon_diff/2, lat - lat_diff/2]
    high = [lon + lon_diff/2, lat + lat_diff/2]

    # integrated probability mass
    m = kde.integrate_box(low, high)

    if debug==True:
        # pointwise density at vent
        f0 = kde.evaluate(np.array([[lon], [lat]]))[0]

        # area in degree^2
        area_deg2 = lon_diff * lat_diff

        # small-area approximation
        m_approx = f0 * area_deg2

        ratio = m / (m_approx + 1e-300)

        print(
            f"lon={lon:.3f}, lat={lat:.3f} | "
            f"f0={f0:.3e}, "
            f"m={m:.3e}, "
            f"f0×A={m_approx:.3e}, "
            f"ratio={ratio:.3e}"
        )
        print(f'Lily Integral')
        print(f'lon, lat = {point}')
        print(f'box area: (lon_min, lat_min),(lon_max, lat_max) = {low}, {high}')

    return m

# Diagnose why some non-zero prob density functions integrate to 0 to test Numerical CDF issues
box_diagnostics = False
def box_diagnostics(kde, lon, lat, box_size=100.0, ngrid=31):
    # degrees for box_size meters
    lat_diff = box_size / haversine(lat, lon, lat + 1, lon)
    lon_diff = box_size / haversine(lat, lon, lat, lon + 1)

    low  = [lon - lon_diff/2, lat - lat_diff/2]
    high = [lon + lon_diff/2, lat + lat_diff/2]

    # point density
    f0 = kde.evaluate(np.array([[lon], [lat]]))[0]

    # integrate_box
    m = kde.integrate_box(low, high)

    # sample inside box
    lons = np.linspace(low[0], high[0], ngrid)
    lats = np.linspace(low[1], high[1], ngrid)
    Lon, Lat = np.meshgrid(lons, lats)
    pts = np.vstack([Lon.ravel(), Lat.ravel()])
    f_grid = kde.evaluate(pts).reshape(ngrid, ngrid)

    fmax = float(f_grid.max())

    dlon = (high[0] - low[0]) / (ngrid - 1)
    dlat = (high[1] - low[1]) / (ngrid - 1)
    m_brute = float(f_grid.sum() * dlon * dlat)

    return {
        "f0": float(f0),
        "m_integrate_box": float(m),
        "m_brute": m_brute,
        "fmax_in_box": fmax,
        "low": low,
        "high": high,
    }

def check_kde_box(kde, point, vent_index=None, v_index=None, 
                  source_name=None, target_name=None,
                  threshold=0.0):
    """
    Runs diagnostics on KDE box integration and prints only suspicious cases.
    """
    lon, lat = point
    diag = box_diagnostics(kde, lon, lat)

    f0      = diag["f0"]
    m_box   = diag["m_integrate_box"]
    m_brute = diag["m_brute"]
    fmax    = diag["fmax_in_box"]

    # Suspicious logic
    suspicious = (
        (f0 <= threshold and m_box > threshold) or
        (f0 > threshold and m_box <= threshold)
    )
    if suspicious:
        print("\n" + "="*60)
        print("SUSPICIOUS KDE CASE")
        print(f"Source index: {vent_index} ({source_name})")
        print(f"Target index: {v_index} ({target_name})")
        print(f"lon={lon:.6f}, lat={lat:.6f}")
        print(f"f0={f0:.3e}")
        print(f"integrate_box m={m_box:.3e}")
        print(f"brute_force m={m_brute:.3e}")
        print(f"max density in box={fmax:.3e}")
        print("="*60)
    return diag


#============================================================================================================#
#======================================= PDF MATRIX AND PLOTS  ==============================================#
#============================================================================================================#

print('Begin plotting PDF plots and probability values', flush = True)

## Create matrix
# Preallocate the matrix and labels
names = vent_sites['name'].astype(str).str.strip().tolist()
n = len(names)
#conn_matrix = np.zeros((n, n))   # rows = source, cols = target
kde_conn_matrix = np.zeros((n, n))   # rows = source, cols = target

print(f'empty matrix has been created', flush = True)
# As a loop
#=======================================================
# define area of interest
plot_lat_start = -57
plot_lat_end   = 30
plot_lon_start = 17
plot_lon_end   = 113
#=======================================================

# manually move labels of text position to prevent overlap 
txtpos_x = np.array([60.1800, 60.3030, 63.7500 + 9, 63.9227, 66.4230, 65.2990, 65.8478 + 10, 69.9387, 70.0405 + 4, 71.4404 + 11, 79.1030, 50.2400 + 11, 49.3900])
txtpos_y = np.array([6.8000 + 6, 6.3630, 3.7500, -27.8508, -11.4147, -18.3365 +1, -19.5569 +1, -23.9183, -25.3195 -2, -26.1567, -41.2520, -37.3900, -37.4700])
txtpos_y = txtpos_y -1 
txtpos_x = txtpos_x -3
s=1000 #subsample of all particles to help with plotting


print('Calculate and print pdfs (weights) for each source vent', flush = True)

for vent_index in range(n_locations):
    print(f'Start loop at vent {vent_index}', flush = True)
    print(f'Isolate a vent location', flush = True)
    ptcls = ds_flat.isel(location=vent_index)#ds_all_locations.isel(location=1).isel(release=1)
    print(f'transpose the particles dataset', flush = True)
    ptcls = ptcls.transpose("trajectory", "obs")
    print(f'redefine', flush = True)
    data_a=ptcls
    print(f'assign lat and lot', flush = True)
    lat_a = np.ma.filled(data_a.variables['lat'], np.nan)
    lon_a = np.ma.filled(data_a.variables['lon'], np.nan)
    # --- set all longitudes to a consistent 0–360 convention
    print(f'Convert lon', flush = True)
    vent_lon = (vent_sites.Longitude.values % 360)
    print(f'subsample lat and lon', flush = True)
    x_a = (lon_a[~np.isnan(lon_a)][0::s] % 360)
    y_a =  lat_a[~np.isnan(lat_a)][0::s]

    # KDE grid in the same space
    print(f'define grid limits', flush = True)
    xmin, xmax = x_a.min(), x_a.max()
    ymin, ymax = y_a.min(), y_a.max()
    print(f'define grid', flush = True)
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    print(f'stack the grid positions', flush = True)
    positions = np.vstack([X.ravel(), Y.ravel()])
    print(f'stack the vent positions', flush = True)
    xy = np.vstack([x_a, y_a])
    print(f'define kde', flush = True)
    kde = gaussian_kde(xy)
    print(f'calculate kde', flush = True)
    Z = np.reshape(kde(positions).T, X.shape)

    # PDF at vent points (same lon convention)
    print(f'evaluate pdf at vent index {vent_index}.', flush = True)
    v_name = vent_sites.name.iloc[vent_index]  
    points = np.vstack([vent_lon, vent_sites.Latitude.values])
    pdf = kde.evaluate(points)

    # store row directly into the connectivity matrix
    #conn_matrix[vent_index, :] = pdf
    kde_conn_matrix[vent_index, :] = pdf

    eg_pdf = pd.DataFrame({
        'Source': np.repeat(v_name, len(vent_lon)),
        'Target': vent_sites.name,
        'lon': vent_lon,
        'lat': vent_sites.Latitude.values,
        'weight': pdf
    })
    print(eg_pdf)
    # annotation offsets (also wrap to 0–360)
    #txtpos_x = (vent_lon - 3) % 360
    #txtpos_y = vent_sites.Latitude.values - 1

    fig, ax = plt.subplots(figsize=(10, 6))

    #============================ depth map ================================
   
    # Subset rectilinear seafloor dataset to the plotting box
    lon_c = seafloor['longitude'].sel(longitude=slice(plot_lon_start, plot_lon_end)).values  # 1D centers
    lat_c = seafloor['latitude'].sel(latitude=slice(plot_lat_start, plot_lat_end)).values    # 1D centers
    Z_sea     = seafloor['deptho'].sel(longitude=slice(plot_lon_start, plot_lon_end),
                                   latitude=slice(plot_lat_start, plot_lat_end)).compute().values  # 2D (lat, lon)
    
    # Build cell EDGES from 1D centers (needed for pcolormesh with shading='flat')
    def edges_from_centers(x):
        dx = np.diff(x)
        left  = x[0]  - dx[0]/2
        right = x[-1] + dx[-1]/2
        return np.concatenate(([left], (x[:-1] + x[1:]) / 2, [right]))
    
    lon_e = edges_from_centers(lon_c)   # length = lon_c.size + 1
    lat_e = edges_from_centers(lat_c)   # length = lat_c.size + 1
    
    Xe, Ye = np.meshgrid(lon_e, lat_e)
    
    ax.pcolormesh(
        Xe, Ye, Z_sea,
        cmap=cmocean.cm.deep, vmin=0, alpha=0.5, zorder=0, shading='flat'
    )
    
    #========================== PDFs ===============================================
    
    # 3) scatter of particle subsample and vent PDFs on TOP
    z_vals = gaussian_kde(np.vstack([x_a, y_a]))(np.vstack([x_a, y_a]))
    ax.scatter(x_a, y_a, c=z_vals, cmap=plt.cm.spring_r, s=3, zorder=3)
    
    scat = ax.scatter(
        eg_pdf.lon, eg_pdf.lat, c=eg_pdf.weight, cmap=plt.cm.spring_r,
        s=40, linewidths=0, alpha=1, zorder=4
    )
    #========================== Axis and labels =======================================
    # 4) axis limits: x=longitude, y=latitude  (previously had them swapped)
    ax.set_xlim(plot_lon_start, plot_lon_end)  # was set_xlim(lat…)
    ax.set_ylim(plot_lat_start, plot_lat_end)  # was set_ylim(lon…)

    print('Make the plot.', flush = True)
    # 5) vents and labels (now aligned with the same lon wrap)
    ax.plot(vent_lon, vent_sites.Latitude.values, 'x', color='black', markersize=8, zorder=5)
    
    for i, w in enumerate(eg_pdf.weight):
        ax.text(txtpos_x[i] - 2, txtpos_y[i] - 2.5, f'{w:.2g}', ha='center', va='center',
                color='black', fontsize=10, zorder=6)
    for i, name in enumerate(eg_pdf.Target):
        ax.text((txtpos_x[i] - 2) % 360, txtpos_y[i], f'{name}',
                ha='center', va='center', color='black', fontsize=10, zorder=6)
    
    # 6) labels, titles, colorbars
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title('pointPDF_1to')
    
    # If you want the KDE colorbar:
    #fig.colorbar(im, ax=ax, label='Gaussian density (KDE)')
    
    # If you also want the vent PDF point colorbar, add another:
    fig.colorbar(scat, ax=ax, label='Gaussian point density')
    
    plt.tight_layout()
#    plt.show()
    plt.savefig('../output/plots/pointPDF_s_'+ str(s) + '_' + output_name + '__' +v_name+'.png')
    print(f'Saved graph pointPDF_ {v_name} to ..output//plots/', flush = True)
    
#connectivity = pd.DataFrame(conn_matrix, index=names, columns=names)
connectivity = pd.DataFrame(kde_conn_matrix, index=names, columns=names)
connectivity.to_csv(f'../output/matrices/kde_connectivity_matrix_{output_name}_s_{s}.csv')
print(f'connectivity matrix has been saved as kde_connectivity_matrix_{output_name}_s_{s}.csv')
print('Probability density matrix has been saved', flush = True)
print('This is a matrix whose entries represent Gaussian KDE density evaluated at target vent locations, used as relative connectivity weights between source and target vents.')

# #====================== Create PDF network (not working)- I think it's now redundant =================================
# # ## Create PDF network
# # Using the subset of particles (full set will be used in python code), extract pairwise probability between each vent site

# # In[ ]:

# ##TRY PRINT HERE to pinpoint 'lon' problem
# # print('Print ptcls:')
# # print(ptcls)
# # Not correct section anyway
# # edges_pdf = pd.DataFrame(columns=['Source','Target','weight'])
# # s=100
# # for v_name in vent_sites.name:
# #     ptcls = ptcls#xr.open_dataset('output/particles/'+run+'_'+depth+'_alt'+v_name+'.nc')
# #     data_a=ptcls 
# #     lat_a = np.ma.filled(data_a.variables['lat'], np.nan)
# #     lon_a = np.ma.filled(data_a.variables['lon'], np.nan)
# #     x_a = lon_a[~np.isnan(lon_a)][0::s]
# #     y_a = lat_a[~np.isnan(lat_a)][0::s]
# #     xy_a = np.vstack([x_a,y_a])
# #     z_a = gaussian_kde(xy_a)(xy_a)
# #     kde = gaussian_kde(xy_a)
    
# #     for t_name in vent_sites.index:
        
# #         point = vent_sites.at[t_name,'lon'],vent_sites.at[t_name,'lat']
# #         int_pdf = get_prob_density_in_box(kde, point)

        
# #         eg_pdf = pd.DataFrame({'Source': [v_name],
# #                        'Target': [str(vent_sites.at[t_name,'name'])],
# #                        'weight': [float(int_pdf)]},
# #                        index=[0])
                
# #         edges_pdf= pd.concat([edges_pdf,eg_pdf])
# #         print(depth,'_@_',v_name,'_',t_name,'finished')

# #edges_pdf.to_csv('output/'+run+'_'+str(s)+'_int_edges_'+depth+'.csv',)


# # In[ ]:


# # edges_pdf = pd.DataFrame(columns=['Source','Target','weight'])
# # s=100
# # for vent_index in range(n_locations):
# #     ptcls = ds_flat.isel(location=vent_index)#ds_all_locations.isel(location=1).isel(release=1)
# #     ptcls = ptcls.transpose("trajectory", "obs")
# #     data_a=ptcls
# #     lat_a = np.ma.filled(data_a.variables['lat'], np.nan)
# #     lon_a = np.ma.filled(data_a.variables['lon'], np.nan)
# #     # --- set all longitudes to a consistent 0–360 convention
# #     vent_lon = (vent_sites.Longitude.values % 360)
# #     x_a = (lon_a[~np.isnan(lon_a)][0::s] % 360)
# #     y_a =  lat_a[~np.isnan(lat_a)][0::s]

# #     # KDE grid in the same space
# #     xmin, xmax = x_a.min(), x_a.max()
# #     ymin, ymax = y_a.min(), y_a.max()
# #     X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
# #     positions = np.vstack([X.ravel(), Y.ravel()])
# #     xy = np.vstack([x_a, y_a])
# #     kde = gaussian_kde(xy)
# #     Z = np.reshape(kde(positions).T, X.shape)

# #     # PDF at vent points (same lon convention)
    
# #     # points = np.vstack([vent_lon, vent_sites.Latitude.values])
# #     # pdf = kde.evaluate(points)
# #     # eg_pdf = pd.DataFrame({
# #     #     'Source': np.repeat(v_name, len(vent_lon)),
# #     #     'Target': vent_sites.name,
# #     #     'lon': vent_lon,
# #     #     'lat': vent_sites.Latitude.values,
# #     #     'weight': pdf
# #     # })
# #     for vent_index in range(n_locations):
# #         v_name = vent_sites.name.iloc[vent_index]  
# #         point = vent_sites.at[vent_index, 'Longitude'], vent_sites.at[vent_index, 'Latitude']
# #         int_pdf = get_prob_density_in_box(kde, point)

# #         eg_pdf = pd.DataFrame({'Source': vent_sites.name,
                                       
# #                        'Target': [str(vent_sites.at[vent_index, 'name'])],
# #                        'weight': [float(int_pdf)]},
# #                         index=[0])
                       
# #         edges_pdf = pd.concat([edges_pdf,ed_pdf])
# #         print(depth, '_@_', v_name,'_',vent_index,'/',n_locations,'_','finished')
# #     #      np.repeat(v_name, len(vent_lon)),
        
# #     #     'lon': vent_lon,
# #     #     'lat': vent_sites.Latitude.values,
# #     #     'weight': pdf
# #     # })
    
# # print(edges_pdf)
# # edges_pdf.to_csv('output/edges_pdf_'+ output_name + '__' +v_name+'.csv')

#============================================================================================================#
#======================================== PROBABILITY MATRIX ================================================#
#============================================================================================================#

print('reached point before edges pdf. But skipped plotting before')
print("start calculating 'edges'", flush = True)

##remove dataframe since now appending directly to matrix
#edges_pdf = pd.DataFrame(columns=['Source','Target','weight'])

## Create matrix
# Preallocate the matrix and labels
names = vent_sites['name'].astype(str).str.strip().tolist()
n = len(names)
conn_matrix = np.zeros((n, n))   # rows = source, cols = target



#s=100 #subsample
s=1   #no subsample
for vent_index in range(n_locations):
    print(f'Start loop at vent {vent_index}', flush = True)
    print(f'Isolate a vent location', flush = True)
    ptcls = ds_flat.isel(location=vent_index)#ds_all_locations.isel(location=1).isel(release=1)
    print(f'Transpose', flush = True)
    ptcls = ptcls.transpose("trajectory", "obs")
    data_a=ptcls
    print(f'Asign lat lon with NaNs', flush = True)
    lat_a = np.ma.filled(data_a.variables['lat'], np.nan)
    lon_a = np.ma.filled(data_a.variables['lon'], np.nan)
    # --- set all longitudes to a consistent 0–360 convention
    print(f'reassign lon values', flush = True)
    vent_lon = (vent_sites.Longitude.values % 360)
    print(f'subsample lon and lat values', flush = True)
    x_a = (lon_a[~np.isnan(lon_a)][0::s] % 360)
    y_a =  lat_a[~np.isnan(lat_a)][0::s]

    # KDE grid in the same space
    print(f'Define grid limits', flush = True)
    xmin, xmax = x_a.min(), x_a.max()
    ymin, ymax = y_a.min(), y_a.max()
    print(f'Define grid', flush = True)
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    xy = np.vstack([x_a, y_a])
    print(f'Define kde gaussian', flush = True)
    kde = gaussian_kde(xy)
    print(f'Calculate pdf', flush = True)
    Z = np.reshape(kde(positions).T, X.shape)

    ###################################################################### 
    #for v_index in vent_sites.index:
    for v_index in range(n_locations):
        print('_@_', vent_index, '_', v_index, '_started')
        point = (vent_sites.iloc[v_index]["Longitude"] % 360,
                 vent_sites.iloc[v_index]["Latitude"])
        if debug==True:
            check_kde_box(kde, point, vent_index=vent_index, v_index=v_index,
                          source_name=vent_sites.name.iloc[vent_index],
                          target_name=vent_sites.name.iloc[v_index]
                         )
        print(f'Calculate prob mass in a box for vent pair {vent_index} and {v_index}', flush = True)   
        int_pdf = get_prob_mass_in_box(kde, point)#, debug=False)
        
        print(f'Write into matrix', flush = True)
        # write directly into matrix
        conn_matrix[vent_index, v_index] = float(int_pdf)
        print('_@_', vent_index, '_', v_index, '_finished')
        print(f"{vent_index} __ {v_index} calculated", flush = True)

    ######################################################################

    
#print(edges_pdf)
#edges_pdf.to_csv('pdf_plots/subset'+str(s)+'_int_edges_'+output_name+'.csv',)
connectivity = pd.DataFrame(conn_matrix, index=names, columns=names)
connectivity.to_csv(f'../output/matrices/s_{s}_connectivity_matrix_{output_name}.csv')

print('Probability mass matrix has been saved', flush = True)

print('Code complete.', flush = True)
# #=================== Network =======================
# print('Create a network')
# g = nx.DiGraph()
# g.add_weighted_edges_from([tuple(x) for x in framex.values])
# a_m = nx.to_numpy_array(g, weight = 'weight')
# node_attr = vent_sites.set_index('name').to_dict('index')
# nx.set_node_attributes(g, node_attr)
# pos = nx.spring_layout(g, k=0.1)
# edges,weights = zip(*nx.get_edge_attributes(g,'weight').items())

# nx.draw(g,pos, node_size=10, with_labels =True)


# pos = {n : (g.nodes[n].get('lon')%360, g.nodes[n].get('lat')) for n in g.nodes()}

# fig, ax = plt.subplots(
#     1, 1, figsize=(20, 14),constrained_layout=True)

# fig.suptitle("Dispersal Network:"+depth+"_altitude", fontsize=30)

# ax.pcolormesh(hgr['glamf'][0]%360,hgr['gphif'][0],zgr['hdepw'][0,:,:].values, cmap=cmocean.cm.deep, alpha=1)

# nx.draw(g,pos, 
#         node_size=200, node_color = 'gray',
#         edge_color=weights,edge_cmap=plt.cm.hot_r, width=3,
#         connectionstyle="arc3,rad=0.2",
#         with_labels =True)



# ax.set_xlim(plot_lon_start, plot_lon_end)  # was set_xlim(lat…)
# ax.set_ylim(plot_lat_start, plot_lat_end)


# plt.savefig('pdfplots/DispersalNetwork_int_'+output_name)

# print('Plotting complete')
# #plt.show()

# Print time statement
end_time = time.time()
elapsed_seconds = end_time - start_time
elapsed_minutes = elapsed_seconds / 60
elapsed_hours = elapsed_minutes / 60

print(f"Simulation complete in {elapsed_seconds:.1f} seconds ({elapsed_minutes:.1f} minutes / {elapsed_hours:.2f} hours).")
