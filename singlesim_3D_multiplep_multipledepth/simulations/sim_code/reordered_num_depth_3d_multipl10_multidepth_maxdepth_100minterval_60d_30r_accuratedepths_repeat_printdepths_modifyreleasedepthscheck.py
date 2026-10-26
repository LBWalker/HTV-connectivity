#!/usr/bin/env python
# coding: utf-8

# In[1]:
# Record time taken for simulations
import time
start_time = time.time()

#import cmocean
import sys
import numpy as np
import netCDF4 as nc
import pandas as pd
import parcels
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
import xarray as xr

from glob import glob
from copy import copy
from scipy import interpolate
from datetime import timedelta
from matplotlib.lines import Line2D

from parcels import (
    AdvectionRK4,
    AdvectionRK4_3D,
    Field,
    FieldSet,
    Geographic,
    GeographicPolar,
    JITParticle,
    ScipyParticle,
    ParticleSet,
    Variable,
)


# In[2]:


# ===============================================================================
# Set up model runs:

# model dimensions: 2D vs 3D
ndim = 3     # 2 for 2D simulations; 3 for 3D simulations

# time
year_start = 2022
year_end = 2023
run_period = 60  # days; Total runtime of simulation
time_step = 2   # hours; Time increment of particle tracking
release_period = 30 # days; consecutive days of release

# particle release
release_depth = 'max_depth'  # Release depth; for 2D: 100-800m (100m-increments); for 3D: max_depth
bouncing = True  # Use displacement functions to prevent particles getting stuck
multi_particles = True # If 'True' uniform release of 'num_pts' particles within 1km radius of HTV coordinates
num_pts = 10
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

print('check point')

# In[3]:


release_depth_cpy = release_depth
data_path_MERC = '/bucket/MitaraiU/Users/heng/GLORYS12/'


# In[4]:


def load_mercator_fields(year_start, months_start, year_end):
    # declare variables
    uv_temp = []
    w_temp = []
    t_temp = []
    uv_files_start = []
    w_files_start = []
    t_files_start = []
    # set data paths
    data_path_start = '/bucket/MitaraiU/Users/heng/GLORYS12/' + str(year_start) + '/'
    data_path_end = '/bucket/MitaraiU/Users/heng/GLORYS12/' + str(year_end) + '/'
    # load velocity dataset names
    for month in range(len(months_start)):
        uv_temp = sorted(glob(data_path_start + 'glo12_rg_1d-m_' + str(year_start) + str(months_start[month]) +'*_3D-uovo_hcst*.nc'))
        w_temp = sorted(glob(data_path_start + 'glo12_rg_1d-m_' + str(year_start) + str(months_start[month]) +'*_3D-wo_hcst*.nc'))
        t_temp = sorted(glob(data_path_start + 'glo12_rg_1d-m_' + str(year_start) + str(months_start[month]) +'*_3D-thetao_hcst*.nc'))
        uv_files_start = uv_files_start + uv_temp
        w_files_start = w_files_start + w_temp
        t_files_start = t_files_start + t_temp
    uv_files_end = sorted(glob(data_path_end + 'glo12_rg_1d-m_*_3D-uovo_hcst*.nc'))
    w_files_end = sorted(glob(data_path_end + 'glo12_rg_1d-m_*_3D-wo_hcst*.nc'))
    t_files_end = sorted(glob(data_path_end + 'glo12_rg_1d-m_*_3D-thetao_hcst*.nc'))
    uv_files_comb = uv_files_start + uv_files_end
    w_files_comb = w_files_start + w_files_end
    t_files_comb = t_files_start + t_files_end
    return uv_files_comb, w_files_comb, t_files_comb


# In[5]:


months_start = ["09", 10, 11, 12] # 09 not accepted as integer, so string
uv_files, w_files, t_files =load_mercator_fields(year_start, months_start, year_end)


# In[6]:


len(uv_files), len(w_files), len(t_files) # should equal to 487 


# In[7]:


import dask
with dask.config.set(**{'array.slicing.split_large_chunks': False}): # this line needed
    t=xr.open_mfdataset(t_files,combine='nested', concat_dim = 'time')


# In[8]:


# Load landmask and bathymetry
mask_bathy_path = data_path_MERC + 'GLO-MFC_001_024_mask_bathy.nc' 
mask_bathy      = nc.Dataset(mask_bathy_path)

landmask = mask_bathy['mask']  # dim=(depth=50, lat=2041, lon=4320); Land: 0, Ocean: 1

# Make negative of land mask (i.e. Land: 1, Ocean: 0)
landmask_negative = 1 - landmask[:]


# In[9]:


landmask_negative.shape


# In[10]:


# Load coordinates file
mesh_mask_path = data_path_MERC + 'GLO-MFC_001_024_coordinates.nc' 
mesh_mask      = nc.Dataset(mesh_mask_path)


# In[11]:


# Specify filenames, variables, and dimensions
filenames = {
    'U': {'lon': mesh_mask_path, 'lat': mesh_mask_path, 'depth': mesh_mask_path, 'data': uv_files},
    'V': {'lon': mesh_mask_path, 'lat': mesh_mask_path, 'depth': mesh_mask_path, 'data': uv_files},
    'W': {'lon': mesh_mask_path, 'lat': mesh_mask_path, 'depth': mesh_mask_path, 'data': w_files},
    'T': {'lon': mesh_mask_path, 'lat': mesh_mask_path, 'depth': mesh_mask_path, 'data': t_files}
}
variables = {
    'U': 'uo',
    'V': 'vo',
    'W': 'wo',
    'T': 'thetao'
}
dimensions = {
    'U': {'lon': 'longitude', 'lat': 'latitude', 'depth': 'depth', 'time': 'time'},
    'V': {'lon': 'longitude', 'lat': 'latitude', 'depth': 'depth', 'time': 'time'},
    'W': {'lon': 'longitude', 'lat': 'latitude', 'depth': 'depth', 'time': 'time'},
    'T': {'lon': 'longitude', 'lat': 'latitude', 'depth': 'depth', 'time': 'time'}
}

# filenames = {
#     'U': {'lon': mesh_mask_path, 'lat': mesh_mask_path, 'depth': mesh_mask_path, 'data': uv_files},
#     'V': {'lon': mesh_mask_path, 'lat': mesh_mask_path, 'depth': mesh_mask_path, 'data': uv_files},
#     'W': {'lon': mesh_mask_path, 'lat': mesh_mask_path, 'depth': mesh_mask_path, 'data': w_files}
# }
# variables = {
#     'U': 'uo',
#     'V': 'vo',
#     'W': 'wo'
# }
# dimensions = {
#     'U': {'lon': 'longitude', 'lat': 'latitude', 'depth': 'depth', 'time': 'time'},
#     'V': {'lon': 'longitude', 'lat': 'latitude', 'depth': 'depth', 'time': 'time'},
#     'W': {'lon': 'longitude', 'lat': 'latitude', 'depth': 'depth', 'time': 'time'}
# }

# Create field set
fieldset = FieldSet.from_netcdf(filenames, variables, dimensions) #.from_netcdf has different types (in docs)

# interp_method='linear_invdist_land_tracer'
# filenames_temp = {
#     'T': {'lon': mesh_mask_path, 'lat': mesh_mask_path, 'depth': mesh_mask_path, 'data': t_files}
# }
# variables_temp = {
#     'T': 'thetao'
# }
# dimensions_temp = {
#     'T': {'lon': 'longitude', 'lat': 'latitude', 'depth': 'depth', 'time': 'time'}
#}
# temp_field = Field.from_netcdf(filenames_temp, variables_temp, dimensions_temp, interp_method='linear_invdist_land_tracer')
# fieldset.add_field(temp_field)

# fieldset.add_field(
#         Field(
#             "temp_interp",
#             data=v_displacement,
#             lon=fieldset.U.grid.lon,
#             lat=fieldset.U.grid.lat,
#             mesh="spherical",
#         )
#     )


# In[12]:


# ===============================================================================
# Indian HTV (HTV) release locations:
excel_file_path = '/flash/MitaraiU/oceanparcels/htv/lily/indianocean_accurate_perez2021_fields_singlemeandepth.csv' #indianocean_perez2021_fields.xlsx' 
htv_data = pd.read_csv(excel_file_path)

htv_lon = htv_data['Longitude']
htv_lat = htv_data['Latitude']
htv_literature_depths = htv_data['Depth (m)']
#htv_dep = htv_data['WaterDepth_m']


# In[13]:


# Run uniform point release and coordinate cleaning function
depth_data = xr.open_mfdataset(mask_bathy_path)


#=================================================================================

#get_ipython().run_line_magic('run', '../../soph_master_thesis/soph_uniform_point_release.ipynb')
# can't import so write directly 

from numpy import pi, cos, sin, sqrt, arange
import matplotlib.pyplot as pp

#=================================================================================
# Define uniform point distribution (multi particles per site)
# ==============================================================================
# Define multi-particle point distribution using sunflower spiral
def uniform_point_release(lon_vector, lat_vector):
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    r = np.sqrt(indices / num_pts) * (1 / 200)
    theta = np.pi * (1 + 5**0.5) * indices  # golden ratio
    new_x, new_y = [], []

    for i in range(len(lon_vector)):
        lon = lon_vector[i]
        lat = lat_vector[i]
        x = r * np.cos(theta) + lon
        y = r * np.sin(theta) + lat
        new_x.extend(x)
        new_y.extend(y)

    return new_x, new_y

# ==============================================================================
# Filter coordinates that are on land (depth = 0 or NaN)
def sea_coord(lon_list, lat_list, depth_dataset):
    valid_lon, valid_lat = [], []
    for i in range(len(lon_list)):
        depth = depth_dataset.interp(
            longitude=lon_list[i],
            latitude=lat_list[i],
            method='nearest'
        )
        if np.all(np.nan_to_num(depth)) != 0:
            valid_lon.append(lon_list[i])
            valid_lat.append(lat_list[i])
        else:
            print(f"Datapoint lon={lon_list[i]} lat={lat_list[i]} is on land.", flush=True)
    return valid_lon, valid_lat

# ==============================================================================
# Interpolate bathymetric depth at each coordinate
def int_depth_model(dataset, lon, lat):
    depths = []
    for i in range(len(lon)):
        depth = dataset['deptho'].interp(
            longitude=lon[i],
            latitude=lat[i],
            method='nearest'
        )
        depths.append(depth.values)
    return np.array(depths)

# FIND RELEASE DEPTHS PER SITE#
# ==============================================================================
# BLOCK 1: Set up coordinates and compute max release depths (filtered first)

# Save original coordinates before any processing, may differ slightly after release pattern decided
original_vent_lon = htv_lon.copy()
original_vent_lat = htv_lat.copy()


######## From block 1 (which was moved below block 2)
# Interpolate seafloor depth and compute max release depth (minus 10m safety buffer)
seafloor = xr.open_mfdataset(mask_bathy_path)
max_release_depths_bathymetry = int_depth_model(seafloor, htv_lon, htv_lat) - 10 ## changed this
max_release_depths_literature = htv_literature_depths

print(f"The maximum depths at the site points from the interpolated bathymetry are: {max_release_depths_bathymetry}")

max_release_depths = []
# Define the maximum depth that the simulation will use (this depends on both the bathmetry and literature data)
# Fallback for flat depth if not using incremental
if release_depth == 'max_depth':
    for i in range(len(original_vent_lon)):
        if max_release_depths_bathymetry[i]-max_release_depths_literature[i]<0:
            max_release_depths.append(max_release_depths_bathymetry[i])
        else:
            max_release_depths.append(max_release_depths_literature[i])
    new_release_depth = max_release_depths
else:
    new_release_depth = [release_depth] * len(max_release_depths)

print(f"number of original_vent_lon: {len(original_vent_lon)}")
print(f"number of htv_lon after processing: {len(htv_lon)}")
print(f"number of max_release_depths: {len(max_release_depths)}")
print(f"original_vent_lon: {original_vent_lon}")
print(f"htv_lon after processing: {htv_lon}")
print(f"max depths interpolated from the bathymetry: {max_release_depths_bathymetry}")
print(f"max_release_depths including literature depth, depending on which is more shallow: {max_release_depths}")


# ==============================================================================
# BLOCK 2: Count number of depth intervals for each site
min_depth = 10
n_depths_per_site = []

for i in range(len(original_vent_lon)):
    if release_depth == 'max_depth':
        gen_max_depth = max_release_depths[i]
    else:
        gen_max_depth = min(release_depth, max_release_depths[i])

    if gen_max_depth < depth_interval:
        n_depths_per_site.append(0)
    else:
        release_depths = np.arange(min_depth, gen_max_depth + 1, depth_interval)
        n_depths_per_site.append(len(release_depths))

#Save depths as a json file for later to avoid contamination
import json

# Save to JSON
metadata_tmp = {
    "n_depths_per_site": n_depths_per_site,
    "vent_lon": list(original_vent_lon),
    "vent_lat": list(original_vent_lat),
}
with open("../output/" + "sim_metadata_tmp" + output_name + ".json", "w") as f:
    json.dump(metadata_tmp, f)

print(f"n_depths_per_site before multiple particles: {n_depths_per_site}", flush =True)
# ==============================================================================
# BLOCK 1 continued

# Apply particle spreading and land filtering
if multi_particles:
    htv_lon_uni, htv_lat_uni = uniform_point_release(htv_lon, htv_lat)
    htv_lon, htv_lat = sea_coord(htv_lon_uni, htv_lat_uni, depth_data.deptho)
else:
    htv_lon, htv_lat = sea_coord(htv_lon, htv_lat, depth_data.deptho)

# # Interpolate seafloor depth and compute max release depth (minus 10m safety buffer)
# seafloor = xr.open_mfdataset(mask_bathy_path)
# max_release_depths = int_depth_model(seafloor, htv_lon, htv_lat) - 10

# # Fallback for flat depth if not using incremental
# if release_depth == 'max_depth':
#     new_release_depth = max_release_depths
# else:
#     new_release_depth = [release_depth] * len(max_release_depths)

# ==============================================================================

print(f"original_vent_lon before release: {len(original_vent_lon)}")
print(f"htv_lon after processing before release: {len(htv_lon)}")
print(f"max_release_depths before release: {len(max_release_depths)}")

if multi_particles:
    release_lon = htv_lon_uni
else:
    release_lon = htv_lon

print(f"max_release_depths before release: {len(max_release_depths)}")
print('reinitialising release depths')
# recompute max depths for multiple particles ****

# #max_release_depths = int_depth_model(seafloor, htv_lon, htv_lat) - 10 ## changed this
# max_release_depths_bathymetry = int_depth_model(seafloor, htv_lon, htv_lat) - 10 ## changed this
# max_release_depths_literature = htv_literature_depths

# max_release_depths = []
# # Define the maximum depth that the simulation will use (this depends on both the bathmetry and literature data)
# #Fallback for flat depth if not using incremental
# if release_depth == 'max_depth':
#     for i in range(len(original_vent_lon)):
#         if max_release_depths_bathymetry[i]-max_release_depths_literature[i]<0:
#             max_release_depths.append(max_release_depths_bathymetry[i])
#         else:
#             max_release_depths.append(max_release_depths_literature[i])
#     new_release_depth = max_release_depths
# else:
#     new_release_depth = [release_depth] * len(max_release_depths)

# # # Fallback for flat depth if not using incremental (ORIGINAL)
# # if release_depth == 'max_depth':
# #     new_release_depth = max_release_depths
# # else:
# #     new_release_depth = [release_depth] * len(max_release_depths)

# #added
# max_release_depths = np.repeat(max_release_depths, num_pts)

print('reinitialising release depths')

# Bathymetry depth for each spread point
max_release_depths_bathymetry = (
    int_depth_model(seafloor, htv_lon, htv_lat) - 10
)

# Match literature depths to the same point ordering

max_release_depths_literature = np.asarray(
    htv_literature_depths, dtype=float
).reshape(-1)

if multi_particles:
    max_release_depths_literature = np.repeat(
        htv_literature_depths, num_pts
    )
else:
    max_release_depths_literature = np.asarray(htv_literature_depths)

assert len(max_release_depths_bathymetry) == len(max_release_depths_literature)

# Shallower of local bathymetry and the corresponding site's literature depth
max_release_depths = np.minimum(
    max_release_depths_bathymetry,
    max_release_depths_literature
)

if release_depth == 'max_depth':
    new_release_depth = max_release_depths.copy()
else:
    new_release_depth = np.minimum(release_depth, max_release_depths)

print(f"reinitialised max_release_depths before release: {len(max_release_depths)}")

# BLOCK 3: Generate per-depth particle release coordinates

max_release_depths = np.asarray(max_release_depths, dtype=float)

print("gen_max_depth:", type(gen_max_depth), repr(gen_max_depth), flush=True)
print("depth_interval:", type(depth_interval), repr(depth_interval), flush=True)
print(
    "max_release_depths:",
    type(max_release_depths),
    np.shape(max_release_depths),
    flush=True
)

if incremental_depth_release:
    plon, plat, pdep = [], [], []
    for i in range(len(release_lon)):
        if release_depth == 'max_depth':
            gen_max_depth = max_release_depths[i]
        else:
            gen_max_depth = min(release_depth, max_release_depths[i])

        if gen_max_depth < depth_interval:
            continue

        release_depths = np.arange(min_depth, gen_max_depth + 1, depth_interval)
        for d in release_depths:
            plon.append(htv_lon[i])
            plat.append(htv_lat[i])
            pdep.append(d)
else:
    npart = 1
    plon = htv_lon * np.ones(npart)
    plat = htv_lat * np.ones(npart)
    pdep = new_release_depth

# Time interval between releases
repeatdt = timedelta(hours=24)

print(f"Release depths array: {max_release_depths}")
#================================================================================
# In[16]:


# Calculate lon/lat on grid edges: dx = 0.5
lon_vals, lat_vals = np.meshgrid(mesh_mask['longitude'], mesh_mask['latitude'])

# Centers of the gridcells formed by 4 nodes = velocity nodes + 0.5 dx
x = mesh_mask['longitude'][:-1] + np.diff(mesh_mask['longitude']) / 2 # add edge values, divide by 2, get centre 
y = mesh_mask['latitude'][:-1] + np.diff(mesh_mask['latitude']) / 2   # add edge values, divide by 2, get centre 
lon_centers, lat_centers = np.meshgrid(x, y)


# In[17]:


# makes interpolation work :))
if ndim == 2:
    # Choose land mask at depth closest to release depth
    difference_array = np.abs(mesh_mask['depth'][:] - release_depth_cpy)
    index = difference_array.argmin()
    
    # Interpolate landmask to cell centers
    # -> Only cells with 4 neighboring land points will be land
    fl = interpolate.RectBivariateSpline(
        mesh_mask['latitude'], mesh_mask['longitude'], landmask_negative[index], kx=1, ky=1
    )
    l_centers = fl(lat_centers[:, 0], lon_centers[0, :])
    
    # Land when interpolated value == 1
    lmask = np.ma.masked_values(l_centers, 1)
elif ndim == 3:
    landmask_3D = np.zeros((50, 2040, 4319))
    for d in range(len(mesh_mask['depth'])):
        fl = interpolate.RectBivariateSpline(
            mesh_mask['latitude'], mesh_mask['longitude'], landmask_negative[d], kx=1, ky=1
        )
        l_centers = fl(lat_centers[:, 0], lon_centers[0, :])
        
        # Land when interpolated value == 1
        lmask = np.ma.masked_values(l_centers, 1)
        landmask_3D[d] = lmask
else:
    print("Invalid input for ndim. Must be 2 or 3.")


# In[18]:


# NEEDED FOR A GRID, NOT NEEDED FOR C GRID


#=============================================================================

# Import displacement functions
#get_ipython().run_line_magic('run', '../../soph_master_thesis/soph_particle_displacement.ipynb')

# Import displacement functions
#%run soph_particle_displacement.ipynb

def get_coastal_nodes(lmask):
    
    mask_lap = np.roll(lmask, -1, axis=0) + np.roll(lmask, 1, axis=0)
    mask_lap += np.roll(lmask, -1, axis=1) + np.roll(lmask, 1, axis=1)
    mask_lap -= 4 * lmask
    coastal = np.ma.masked_array(lmask, mask_lap > 0)
    coastal = coastal.mask.astype("int")
    
    return coastal

def get_shore_nodes(lmask):
    
    mask_lap = np.roll(lmask, -1, axis=0) + np.roll(lmask, 1, axis=0)
    mask_lap += np.roll(lmask, -1, axis=1) + np.roll(lmask, 1, axis=1)
    mask_lap -= 4 * lmask
    shore = np.ma.masked_array(lmask, mask_lap < 0)
    shore = shore.mask.astype("int")

    return shore

def get_coastal_nodes_diagonal(lmask):
    
    mask_lap = np.roll(lmask, -1, axis=0) + np.roll(lmask, 1, axis=0)
    mask_lap += np.roll(lmask, -1, axis=1) + np.roll(lmask, 1, axis=1)
    mask_lap += np.roll(lmask, (-1, 1), axis=(0, 1)) + np.roll(
        lmask, (1, 1), axis=(0, 1)
    )
    mask_lap += np.roll(lmask, (-1, -1), axis=(0, 1)) + np.roll(
        lmask, (1, -1), axis=(0, 1)
    )
    mask_lap -= 8 * lmask
    coastal = np.ma.masked_array(lmask, mask_lap > 0)
    coastal = coastal.mask.astype("int")

    return coastal

def get_shore_nodes_diagonal(lmask):

    mask_lap = np.roll(lmask, -1, axis=0) + np.roll(lmask, 1, axis=0)
    mask_lap += np.roll(lmask, -1, axis=1) + np.roll(lmask, 1, axis=1)
    mask_lap += np.roll(lmask, (-1, 1), axis=(0, 1)) + np.roll(
        lmask, (1, 1), axis=(0, 1)
    )
    mask_lap += np.roll(lmask, (-1, -1), axis=(0, 1)) + np.roll(
        lmask, (1, -1), axis=(0, 1)
    )
    mask_lap -= 8 * lmask
    shore = np.ma.masked_array(lmask, mask_lap < 0)
    shore = shore.mask.astype("int")

    return shore
    
  
def create_displacement_field(lmask, double_cell=False):

    shore = get_shore_nodes(lmask)

    # nodes bordering ocean directly and diagonally
    shore_d = get_shore_nodes_diagonal(lmask)
    # corner nodes that only border ocean diagonally
    shore_c = shore_d - shore

    # Simple derivative
    Ly = np.roll(lmask, -1, axis=0) - np.roll(lmask, 1, axis=0)   # difference in the vertical (latitude)
    Lx = np.roll(lmask, -1, axis=1) - np.roll(lmask, 1, axis=1)    # difference in the horizontal (longitude)
    Ly_c = np.roll(lmask, -1, axis=0) - np.roll(lmask, 1, axis=0)  # difference in the vertical (corners only)
    Lx_c = np.roll(lmask, -1, axis=1) - np.roll(lmask, 1, axis=1)    # difference in the horizontal (corners only)
    
    # Include y-component of diagonal neighbors
    Ly_c += np.roll(lmask, (-1, -1), axis=(0, 1)) + np.roll(     #Check for corners above and below, left side.
        lmask, (-1, 1), axis=(0, 1)
    )
    Ly_c += -np.roll(lmask, (1, -1), axis=(0, 1)) - np.roll(    #Check for corners above and below, right side.
        lmask, (1, 1), axis=(0, 1)
    )
    
    # Include x-component of diagonal neighbors    
    Lx_c += np.roll(lmask, (-1, -1), axis=(1, 0)) + np.roll(
        lmask, (-1, 1), axis=(1, 0)
    )
    Lx_c += -np.roll(lmask, (1, -1), axis=(1, 0)) - np.roll(
        lmask, (1, 1), axis=(1, 0)
    )

    v_x = -Lx * (shore) # negative, to be displaced away from shore!
    v_y = -Ly * (shore)

    v_x_c = -Lx_c * (shore_c)
    v_y_c = -Ly_c * (shore_c)

    v_x = v_x + v_x_c
    v_y = v_y + v_y_c

    magnitude = np.sqrt(v_y**2 + v_x**2)
    # the coastal nodes between land create a problem. Magnitude there is zero
    # I force it to be 1 to avoid problems when normalizing.
    ny, nx = np.where(magnitude == 0)
    magnitude[ny, nx] = 1

    v_x = v_x / magnitude / 10
    v_y = v_y / magnitude / 10

    return v_x, v_y
    
def distance_to_shore(lmask, dx=1):
    
    ci = get_coastal_nodes(lmask)  # direct neighbors
    dist = ci * dx  # 1 dx away

    ci_d = get_coastal_nodes_diagonal(lmask)  # diagonal neighbors
    dist_d = (ci_d - ci) * np.sqrt(2 * dx**2)  # sqrt(2) dx away

    return dist + dist_d
    
def distance_2_seabed(landmask, dx=1):
    d2seabed_layer=[]
    for d in range(len(landmask[:,0,0])):
        distance = landmask[d,:,:] * dx
        d2seabed_layer.append(distance) 
    d2seabed_array = np.array(d2seabed_layer)
    
    return d2seabed_array

def set_displacement(particle, fieldset, time):
    particle.d2s = fieldset.distance2shore[
        time, particle.depth, particle.lat, particle.lon
    ]
    if particle.d2s < 0.5:
        dispUab = fieldset.dispU[time, particle.depth, particle.lat, particle.lon]
        dispVab = fieldset.dispV[time, particle.depth, particle.lat, particle.lon]
        particle.dU = dispUab
        particle.dV = dispVab
    else:
        particle.dU = 0.0
        particle.dV = 0.0

def set_displacement_depth(particle, fieldset, time):
    particle.d2seabed = fieldset.distance2seabed[
        time, particle.depth, particle.lat, particle.lon
    ]

def displace(particle, fieldset, time):
    if particle.d2s < 0.5:
        particle_dlon += particle.dU * particle.dt
        particle_dlat += particle.dV * particle.dt

def displace_z(particle, fieldset, time):
    if particle.d2seabed == 1:
        particle_ddepth -= 10
        
#Create displacement field for all depth layers using landmask field provided by GLORYS model
#Note: landmask defined by GLORYS is the inverse of the PARCELS defined landmask!!

def displacement_field_3D(landmask):
    v_x_all=[]
    v_y_all=[]

    for d in range(len(landmask[:,0,0])):
        v_x_per_layer=[]
        v_y_per_layer=[]
        v_x_per_layer, v_y_per_layer = create_displacement_field(landmask[d,:,:])
        v_x_all.append(v_x_per_layer)
        v_y_all.append(v_y_per_layer) 

    v_x_array = np.array(v_x_all)
    v_y_array = np.array(v_y_all)
    
    return v_x_array, v_y_array

def distance_2_shore_3D(landmask):
    d2s_per_layer=[]
    for d in range(len(landmask[:,0,0])):
        distance= distance_to_shore(landmask[d,:,:])
        d2s_per_layer.append(distance) 
    d2s_array = np.array(d2s_per_layer)
    
    return d2s_array
#=============================================================================

if ndim == 2:
    coastal  = get_coastal_nodes_diagonal(landmask_negative[index])
    shore    = get_shore_nodes_diagonal(landmask_negative[index])
    v_x, v_y = create_displacement_field(landmask_negative[index])
    d_2_s    = distance_to_shore(landmask_negative[index])
elif ndim == 3:
    vx_3D, vy_3D = displacement_field_3D(landmask_negative)
    d2s_3D = distance_2_shore_3D(landmask_negative)
    d_2_seabed = distance_2_seabed(landmask_negative, dx=1)
else:
    print("Invalid input for ndim. Must be 2 or 3.")


# In[19]:


# Add displacement fields
if ndim == 2:
    u_displacement = v_x
    v_displacement = v_y
    fieldset.add_field(
        Field(
            "dispU",
            data=u_displacement,
            lon=fieldset.U.grid.lon,
            lat=fieldset.U.grid.lat,
            mesh="spherical",
        )
    )
    fieldset.add_field(
        Field(
            "dispV",
            data=v_displacement,
            lon=fieldset.U.grid.lon,
            lat=fieldset.U.grid.lat,
            mesh="spherical",
        )
    )
    fieldset.dispU.units = GeographicPolar()
    fieldset.dispV.units = Geographic()
    
    fieldset.add_field(
        Field(
            "landmask",
            landmask_negative[index],
            lon=fieldset.U.grid.lon,
            lat=fieldset.U.grid.lat,
            mesh="spherical",
        )
    )
    fieldset.add_field(
        Field(
            "distance2shore",
            d_2_s,
            lon=fieldset.U.grid.lon,
            lat=fieldset.U.grid.lat,
            mesh="spherical",
        )
    )
elif ndim == 3:
    fieldset.add_field(
        Field(
            "dispU",
            data=vx_3D,
            lon=fieldset.U.grid.lon,
            lat=fieldset.U.grid.lat,
            depth=fieldset.U.grid.depth,
            mesh="spherical",
        )
    )
    fieldset.add_field(
        Field(
            "dispV",
            data=vy_3D,
            lon=fieldset.V.grid.lon,
            lat=fieldset.V.grid.lat,
            depth = fieldset.V.grid.depth,
            mesh="spherical",
        )
    )
    fieldset.dispU.units = GeographicPolar()
    fieldset.dispV.units = Geographic()
    
    fieldset.add_field(
        Field(
            "lmask",
            landmask_negative,
            lon=fieldset.U.grid.lon,
            lat=fieldset.U.grid.lat,
            depth = fieldset.U.grid.depth,
            mesh="spherical",
        )
    )
    fieldset.add_field(
        Field(
            "distance2shore",
            d2s_3D,
            lon=fieldset.U.grid.lon,
            lat=fieldset.U.grid.lat,
            depth = fieldset.U.grid.depth,
            mesh="spherical",
        )
    )
    fieldset.add_field(
        Field(
            "distance2seabed",
            d_2_seabed,
            lon=fieldset.U.grid.lon,
            lat=fieldset.U.grid.lat,
            depth = fieldset.U.grid.depth,
            mesh="spherical",
        )
    )


# In[27]:


# # ==============================================================  
# # Define Particles
# # Displacement Particle:    
# if particle == 'passive':
#     class DisplacementParticle(JITParticle): #JIT or Scipy (better debugging for writing own kernels)
#         dU  = Variable('dU')
#         dV  = Variable('dV')
#         d2s = Variable('d2s', initial=1e3) #distance to seabed
#         temp = Variable('temp', dtype=np.float32, initial=np.nan)
#         # added more variable 06/03/25
#         d2seabed = Variable('d2seabed', initial=1e3)
#         dZ = Variable('dZ')
        

# elif particle == 'active': 
#     #fieldset.maxdepth = fieldset.U.depth[49] -1 # lowermost depth layer
#     get_ipython().run_line_magic('run', 'soph_VerticalMigration.ipynb')
#     class SwimParticle(JITParticle):
#         migration_phase = Variable('migration_phase', dtype=np.int32, initial=0.0)
#         larvae_age = Variable('larvae_age', dtype=np.float32, initial=0.0)
#         temp = Variable('temp', dtype=np.float32, initial=np.nan)
#         dU  = Variable('dU')
#         dV  = Variable('dV')
#         dZ = Variable('dZ')
#         d2s = Variable('d2s', initial=1e3)
#         d2seabed = Variable('d2seabed', initial=1e3)
        
# def SampleTemp (particle, fieldset, time): # this is kernel definition, tracks temps of particle
#     particle.temp = fieldset.T[time, particle.depth , particle.lat, particle.lon]
# # ===============================================================

# ==============================================================  
# Define Particles
# Displacement Particle

class DisplacementParticle(JITParticle):
    dU  = Variable('dU')
    dV  = Variable('dV')
    d2s = Variable('d2s', initial=1e3)
    temp = Variable('temp', dtype=np.float32, initial=np.nan)
    # added more variable 26/03/25
    d2seabed = Variable('d2seabed', initial=1e3)
    dZ = Variable('dZ')
        

class SwimParticle(JITParticle):
    migration_phase = Variable('migration_phase', dtype=np.int32, initial=0.0)
    cycle_age = Variable('cycle_age', dtype=np.float32, initial=0.0)
    larvae_age = Variable('larvae_age', dtype=np.float32, initial=0.0)
    temp = Variable('temp', dtype=np.float32, initial=np.nan)
    u = Variable('u', dtype=np.float32, initial=np.nan)
    v = Variable('v', dtype=np.float32, initial=np.nan)
    dU  = Variable('dU')
    dV  = Variable('dV')
    dZ  = Variable('dZ')
    d2s = Variable('d2s', initial=1e3)
    d2seabed = Variable('d2seabed', initial=1e3)
    growth_dt = Variable('growth_dt', dtype=np.float32, initial=0.0)
    lifetime = Variable('lifetime', dtype=np.float32, initial=0.0)
    hit_bottom_time = Variable('hit_bottom_time', dtype=np.float32, initial=0.0)
    hit_bottom_depth = Variable('hit_bottom_depth', dtype=np.float32, initial=0.0)
 

def SampleTemp (particle, fieldset, time):
    particle.temp = fieldset.T[time, particle.depth , particle.lat, particle.lon]

def SampleVel(particle, fieldset, time):
    particle.u, particle.v = fieldset.UV[time, particle.depth , particle.lat, particle.lon] 

# ===============================================================

# Vertical Migration
def VerticalMigration(particle, fieldset, time):
    top_depth = 20                                               # maximum upper depth particle travels to, m
    swimspeed_initial =  0                                       # m/s, juvenile
    swim_fast = True 						 # if '1': fast swimspeed (after Fox et al. 2016); if '0': reduced 
							         # swimspeed  incl. turubulence (after Fagerström et al. 2022).
    if swim_fast:
        swimspeed_max = 0.5 * 10**-3
        swim_down = 0.5 * 10**-3
    else:
        swimspeed_max = 0.36 * 10**-3                                # m/s, mature ind
        swim_down = 0.36 * 10**-3                                    # m/s     
    beta_lophelia = 3.0095
    beta_multi = 3.1671863
    double_PLD = False 							 # '1': PLD values are doubled; '0': original PLD values
    settle_max = True
    sum_bottom_depth = 0						 # '1': Particle will travel downward until it hits the bottom, 
								 # even if lifetime exceeds 1.0;
								 # '0': Particle will travel downward until lifetime = 1.0
    # ====================================================================

    beta = beta_lophelia

    # Phase 0: Upward swim until end_depth and/or t_competency
    if particle.migration_phase == 0:
       # particle.temp = SampleTemp(particle, fieldset, time)
        # daily larval growth rate after Connor et al. (2007): 1/PLD
        if double_PLD:
            growth_daily = 1 / (2*(math.exp(beta) * (particle.temp / 15) ** (-(1.4) - 0.27 * math.log(particle.temp / 15))))
        else: 
            growth_daily = 1 / (math.exp(beta) * (particle.temp / 15) ** (-(1.4) - 0.27 * math.log(particle.temp / 15)))
        # accumulated growth per timestep (particle.dt)
        particle.growth_dt = growth_daily * (particle.dt/86400)
        # total lifetime of particle is sum of timestep life increments, and extends from 0 (hatching) to 1.0 (settlement) 
        particle.lifetime += particle.growth_dt 
        particle.larvae_age += particle.dt
        # swimspeed increases with respect to growth
        if particle.lifetime < 0.25:
            if particle.depth > top_depth:
                swimspeed = (swimspeed_max - swimspeed_initial)*(particle.lifetime/0.25)
                particle_ddepth -= swimspeed * particle.dt
            else: 
                particle.migration_phase = 1
        elif particle.lifetime >= 0.25 and particle.lifetime < 0.5:
            if particle.depth > top_depth:
                swimspeed = swimspeed_max
                particle_ddepth -= swimspeed * particle.dt
            else: 
                particle.migration_phase = 1
        elif particle.lifetime >= 0.5:
            particle.migration_phase = 2
            
    # Phase 1: Passive drift at max depth until reaching t_competency    
    elif particle.migration_phase == 1:
        particle.larvae_age += particle.dt
        if double_PLD:
            growth_daily = 1 / (2*(math.exp(beta) * (particle.temp / 15) ** (-(1.4) - 0.27 * math.log(particle.temp / 15))))
        else:
            growth_daily = 1 / (math.exp(beta) * (particle.temp / 15) ** (-(1.4) - 0.27 * math.log(particle.temp / 15)))
        particle.growth_dt = growth_daily * (particle.dt/86400)
        particle.lifetime += particle.growth_dt  
        if particle.lifetime < 0.5:
            particle_ddepth += 0   
        else:
            particle.migration_phase = 2
            
    # Phase 2: Downward migration
    elif particle.migration_phase == 2:
        particle.larvae_age += particle.dt
        if double_PLD:
            growth_daily = 1 / (2*(math.exp(beta) * (particle.temp / 15) ** (-(1.4) - 0.27 * math.log(particle.temp / 15))))
        else:
            growth_daily = 1 / (math.exp(beta) * (particle.temp / 15) ** (-(1.4) - 0.27 * math.log(particle.temp / 15)))
        particle.growth_dt = growth_daily * (particle.dt/86400)
        particle.lifetime += particle.growth_dt
        sum_bottom_depth += particle.hit_bottom_depth
        if settle_max:
            if particle.d2seabed != 1:
                if sum_bottom_depth == 0:
                    swimspeed = swim_down
                    particle_ddepth += swimspeed * particle.dt
                else:
                    particle_ddepth += 0
            else:
                if particle.hit_bottom_time == 0.0:                                # we only want to record time and depth of first bottom hit
                    particle.hit_bottom_time = particle.larvae_age / particle.dt
                    particle.hit_bottom_depth = particle.depth
                particle_ddepth += 0
        else:
            if particle.lifetime < 1.0:
                if sum_bottom_depth == 0:
                    swimspeed = swim_down
                    particle_ddepth += swimspeed * particle.dt
                else:
                    particle_ddepth += 0
            else:
                particle_ddepth += 0

#´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´
#3D - avoid "ThroughSurface" error by defining max particle.depth
def AdvectionRK4_3D_lmtd(particle, fieldset, time):
    (u1, v1, w1) = fieldset.UVW[particle]

    lon1 = particle.lon + u1 * 0.5 * particle.dt
    lat1 = particle.lat + v1 * 0.5 * particle.dt
    dep1 = particle.depth + w1 * 0.5 * particle.dt
    dep1 = max(0, dep1)
    (u2, v2, w2) = fieldset.UVW[time + 0.5 * particle.dt, dep1, lat1, lon1, particle]

    lon2 = particle.lon + u2 * 0.5 * particle.dt
    lat2 = particle.lat + v2 * 0.5 * particle.dt
    dep2 = particle.depth + w2 * 0.5 * particle.dt
    dep2 = max(0, dep2)
    (u3, v3, w3) = fieldset.UVW[time + 0.5 * particle.dt, dep2, lat2, lon2, particle]

    lon3 = particle.lon + u3 * particle.dt
    lat3 = particle.lat + v3 * particle.dt
    dep3 = particle.depth + w3 * particle.dt
    dep3 = max(0, dep3)
    (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3, particle]

    particle_dlon += (u1 + 2 * u2 + 2 * u3 + u4) / 6.0 * particle.dt
    particle_dlat += (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * particle.dt
    particle_ddepth += (w1 + 2 * w2 + 2 * w3 + w4) / 6.0 * particle.dt
    particle.depth = max(20, particle.depth) #0 and 0.5 gave out-of-bound error
#´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´

#beware of initialising with NaN values


# In[28]:


if bouncing:
    if continuous_release:
        if particle == 'active':
            pset = ParticleSet(
                fieldset=fieldset, 
                pclass=SwimParticle, 
                lon=plon, 
                lat=plat, 
                depth=pdep,
                repeatdt=repeatdt
            )
        elif particle == 'passive':
            pset = ParticleSet(
                fieldset=fieldset,
                pclass=DisplacementParticle,
                lon=plon,
                lat=plat,
                depth=pdep,
                repeatdt=repeatdt
            )
    else: 
        if particle == 'active':
            pset = ParticleSet(
                fieldset=fieldset, 
                pclass=SwimParticle, 
                lon=plon, 
                lat=plat, 
                depth=pdep
            )
        elif particle == 'passive':
            pset = ParticleSet(
                fieldset=fieldset,
                pclass=DisplacementParticle,
                lon=plon,
                lat=plat,
                depth=pdep
            )
else:
    pset = ParticleSet(
        fieldset=fieldset,
        pclass=JITParticle,
        lon=plon,
        lat=plat,
        depth=pdep
    )


# # Specify kernels

# if ndim == 2:
#     kernels = pset.Kernel(displace) + pset.Kernel(AdvectionRK4) + pset.Kernel(set_displacement) + pset.Kernel(SampleTemp)
# elif ndim == 3:
#     if particle == 'passive':
#         kernels = pset.Kernel(displace) + pset.Kernel(displace_z) + pset.Kernel(AdvectionRK4_3D_lmtd) + pset.Kernel(set_displacement) + pset.Kernel(set_displacement_depth) + pset.Kernel(SampleTemp)
#     elif particle == 'active':
#         kernels = pset.Kernel(SampleTemp) + pset.Kernel(SampleVel) +  pset.Kernel(VerticalMigration) + pset.Kernel(displace) + pset.Kernel(displace_z) + pset.Kernel(AdvectionRK4_3D_lmtd) + pset.Kernel(set_displacement) + pset.Kernel(set_displacement_depth)

# ====================================================================
# Specify kernels

if ndim == 2:
    kernels = pset.Kernel(displace) + pset.Kernel(AdvectionRK4) + pset.Kernel(set_displacement) + pset.Kernel(SampleTemp)
elif ndim == 3:
    if particle == 'passive':
        kernels = pset.Kernel(displace) + pset.Kernel(displace_z) + pset.Kernel(AdvectionRK4_3D_lmtd) + pset.Kernel(set_displacement) + pset.Kernel(set_displacement_depth) + pset.Kernel(SampleTemp)
    elif particle == 'active':
        kernels = pset.Kernel(SampleTemp) + pset.Kernel(SampleVel) +  pset.Kernel(VerticalMigration) + pset.Kernel(displace) + pset.Kernel(displace_z) + pset.Kernel(AdvectionRK4_3D_lmtd) + pset.Kernel(set_displacement) + pset.Kernel(set_displacement_depth)

# ====================================================================

dt = timedelta(hours=time_step)
output_file = pset.ParticleFile(name='../output/' + output_name,   
                                outputdt=timedelta(hours=time_step))

## BLOCK 4: Save n_depths_per_site as metadata to output file
# output_file.metadata["n_depths_per_site"] = n_depths_per_site
# output_file.metadata["vent_lon"] = list(original_vent_lon)  # save original vent coords if desired
# output_file.metadata["vent_lat"] = list(original_vent_lat)
# Load metadata and assign to output

# #Below was not hashed before
# with open("sim_metadata_tmp_" + output_name + ".json", "r") as f:
#     metadata_tmp = json.load(f)

# output_file.metadata["n_depths_per_site"] = metadata_tmp["n_depths_per_site"]
# output_file.metadata["vent_lon"] = metadata_tmp["vent_lon"]
# output_file.metadata["vent_lat"] = metadata_tmp["vent_lat"]

# ====================================================================

# Run simulation
if continuous_release:
    pset.execute(kernels, runtime=timedelta(days=release_period), dt=dt, output_file=output_file)
    
    pset.repeatdt = None     # stop the repeated release

    #continue running for the remaining simulation time
    remaining_period = run_period - release_period
    pset.execute(kernels, runtime=timedelta(days=remaining_period), dt=dt, output_file=output_file)
else:
    pset.execute(kernels, runtime=timedelta(days=run_period), dt=dt, output_file=output_file)

print('Simulation complete')

# ====================================================================

# print('Attach metadata')

# # Load the output zarr file
# ptcls = xr.open_zarr(output_name + ".zarr")

# # Load metadata from the pre-saved JSON
# with open("sim_metadata_tmp_" + output_name + ".json", "r") as f:
#     metadata_tmp = json.load(f)

# # Add as attributes to xarray dataset
# ptcls.attrs["n_depths_per_site"] = metadata_tmp["n_depths_per_site"]
# ptcls.attrs["vent_lon"] = metadata_tmp["vent_lon"]
# ptcls.attrs["vent_lat"] = metadata_tmp["vent_lat"]

# # Save the updated file (you can overwrite or create a new file)
# ptcls.to_zarr("Output_with_metadata.zarr", mode="w")

# print('Metadata attach complete')

# ====================================================================

# Print time statement
end_time = time.time()
elapsed_seconds = end_time - start_time
elapsed_minutes = elapsed_seconds / 60
elapsed_hours = elapsed_minutes / 60

print(f"Simulation complete in {elapsed_seconds:.1f} seconds ({elapsed_minutes:.1f} minutes / {elapsed_hours:.2f} hours).")
