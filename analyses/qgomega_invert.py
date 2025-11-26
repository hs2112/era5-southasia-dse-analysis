# +
import os, xarray as xr, numpy as np, pandas as pd, dask, gc
import cartopy.crs as ccrs, matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy.feature as cfeature

from datetime import datetime as dt
import matplotlib.patches as mpatches
dask.config.set({"array.slicing.split_large_chunks": True})

import metpy.calc as mpcalc
# import metpy.constants as mpconstants
# from metpy.units import units

import xinvert
from xinvert import invert_omega
from xinvert import FiniteDiff
# -

nibox_lon = [68, 78]
nibox_lat = [24, 31]

# +
import sys

n = len(sys.argv)
print("Total arguments passed:", n)
print("Name of Python script:", sys.argv[0])
print("Arguments passed:", [s for s in sys.argv])

year = int(sys.argv[1])
print(year)


# +
# python Xinv_fullZXtnt_autom 2000
# -

year_min = year
year_max = year


# +
def day_sort(s):
    return int(s.split('_')[-1].split('.')[0])

def mon_sort(s):
    return int(s.split('_')[-2])

def year_sort(s):
    return int(s.split('_')[-3])

def filt_files_period(files=list(), year_rng = np.arange(year_min, year_max+1)):
    files2 = [s for s in files if (('.grib' in s) | ('.nc' in s)) & ('idx' not in s)]
    files2 = [s for s in files2 if (int(s.split('_')[-3]) in year_rng) and (int(s.split('_')[-2]) in [3,4])]
    return files2
    
dir_v = '/home/data/lab_hardik/data/ERA5/daily_means/v_component_of_wind/data/'
v_files = os.listdir(dir_v)
v_files = filt_files_period(v_files)
# v_files = [s for s in v_files if (('.grib' in s) | ('.nc' in s)) & ('idx' not in s)]
# v_files = [s for s in v_files if (s.split('_')[-3] == '2016') and int(s.split('_')[-2]) in [3,4,5,6]]
v_files.sort(key = day_sort)
v_files.sort(key = mon_sort)
v_files.sort(key = year_sort)

print(len(v_files))
print(v_files[:2])

dir_u = '/home/data/lab_hardik/data/ERA5/daily_means/u_component_of_wind/data/'
u_files = os.listdir(dir_u)
u_files = filt_files_period(u_files)
# u_files = [s for s in u_files if (('.grib' in s) | ('.nc' in s)) & ('idx' not in s)]
# u_files = [s for s in u_files if (s.split('_')[-3] == '2016') and int(s.split('_')[-2]) in [3,4,5,6]]
u_files.sort(key = day_sort)
u_files.sort(key = mon_sort)
u_files.sort(key = year_sort)

print(len(u_files))
print(u_files[:2])

dir_w = '/home/lab_hardik/daily_means/vertical_velocity/'
w_files = os.listdir(dir_w)
w_files = filt_files_period(w_files)
# w_files = [s for s in w_files if (('.grib' in s) | ('.nc' in s)) & ('idx' not in s)]
# w_files = [s for s in w_files if (s.split('_')[-3] == '2016') and int(s.split('_')[-2]) in [3,4,5,6]]
w_files.sort(key = day_sort)
w_files.sort(key = mon_sort)
w_files.sort(key = year_sort)

print(len(w_files))
print(w_files[:2])

dir_z = '/home/data/lab_hardik/data/ERA5/daily_means/geopotential/data/'
z_files = os.listdir(dir_z)
z_files = filt_files_period(z_files)
# z_files = [s for s in z_files if (('.grib' in s) | ('.nc' in s)) & ('idx' not in s)]
# z_files = [s for s in z_files if (s.split('_')[-3] == '2016') and int(s.split('_')[-2]) in [3,4,5,6]]
z_files.sort(key = day_sort)
z_files.sort(key = mon_sort)
z_files.sort(key = year_sort)

print(len(z_files))
print(z_files[:2])

dir_t = '/home/data/lab_hardik/data/ERA5/daily_means/temperature/data/'
t_files = os.listdir(dir_t)
t_files = filt_files_period(t_files)

# t_files = [s for s in t_files if (('.grib' in s) | ('.nc' in s)) & ('idx' not in s)]
# t_files = [s for s in t_files if (s.split('_')[-3] == '2016') and int(s.split('_')[-2]) in [3,4,5,6]]
t_files.sort(key = day_sort)
t_files.sort(key = mon_sort)
t_files.sort(key = year_sort)

print(len(t_files))
print(t_files[:2])

dir_sp = '/home/scratch/HS_Surface/'
sp_files = os.listdir(dir_sp)
sp_files = [s for s in sp_files if 'surface_pressure' in s if (('.grib' in s) | ('.nc' in s)) & ('idx' not in s)]
sp_files = [s for s in sp_files if (int(s.split('_')[-3]) in np.arange(year_min, year_max + 1)) and 
            int(s.split('_')[-2]) in [3,4]]
sp_files.sort(key = day_sort)
sp_files.sort(key = mon_sort)
sp_files.sort(key = year_sort)

print(len(sp_files))
print(sp_files[:2])

def prprcs(ds):
    ds = ds['sp'].groupby(ds.time.dt.date).mean().to_dataset()
    ds['date'] = ds.date.astype('datetime64[ns]')
    return ds
# -



# +
# os.listdir(dir_out)

# +
dir_out ='/home/data/lab_hardik/heatwaves/ERA5/analyses/RWP/qg_omega/data/'

for i in range(0,len(v_files)): 
    
    ds_v = xr.open_mfdataset(dir_v + v_files[i]).compute()
    
    file_str = 'xinv_forcing_omega_{}.nc'\
               .format(str(ds_v.date.dt.date.item()).replace('-','_'))
    print(file_str)
    
    if file_str in os.listdir(dir_out) and '2016' not in file_str:
        print('skipping', file_str)
        continue
    
    print("proceeding with", file_str)
    ds_u = xr.open_mfdataset(dir_u + u_files[i]).compute()
    ds_w = xr.open_mfdataset(dir_w + w_files[i]).compute()
    ds_z = xr.open_mfdataset(dir_z + z_files[i]).compute()
    ds_t = xr.open_mfdataset(dir_t + t_files[i]).compute()
    ds_sp = xr.open_mfdataset(
        dir_sp + sp_files[i],
        preprocess=prprcs,
        backend_kwargs = {'indexpath':''},
    ).compute()
    
    ds = ds_v.merge(ds_u).merge(ds_w).merge(ds_z).merge(ds_t).merge(ds_sp).metpy.parse_cf().squeeze('date')

    print('dataset ready')

    ds = ds.assign(hgt=ds['z']/9.80665).drop('z')\
    .assign(vertical = ds['isobaricInhPa'] * 100)\
    .swap_dims({'isobaricInhPa':'vertical'})\
    .drop('isobaricInhPa')

    print(list(ds.data_vars))
    print(ds.dims)

    Rd = 287.04
    Cp = 1004.88
    omega = 7.292e-5

    ################## calculate basic variables ##################
    fd = FiniteDiff({'X':'longitude', 'Y':'latitude', 'Z':'vertical'},
                    BCs={'X':('periodic','periodic'),
                         'Y':('reflect','reflect'),
                         'Z':('extend','extend')}, 
                    fill=0, 
                    coords='lat-lon')

    ############# zonal running smooth of polar grids ###############
    def smooth(v, gridpoint=13, latitude=80):
        rolled = v.pad({'longitude':(gridpoint,gridpoint)},mode='wrap')\
                  .rolling(longitude=gridpoint, center=True, min_periods=1).mean()\
                  .isel(longitude=slice(gridpoint, -gridpoint))
        return xr.where(np.abs(v-v+v.latitude)>latitude, rolled, v)

    # smooth out zonal grid-scale noise
    T   = smooth(ds.t, latitude=85)
    U   = smooth(ds.u, latitude=85)
    V   = smooth(ds.v, latitude=85)
    W   = smooth(ds.w, latitude=85)
    H   = smooth(ds.hgt, latitude=85)

    print('variables smoothened')
    
    p   = ds.vertical
    Psfc= ds.sp

    f    = 2*omega*np.sin(np.deg2rad(ds.latitude)) # Coriolis parameter
    th   = T * (100000 / p)**(Rd/Cp)   # potential temperature
    TH   = th.mean(['latitude','longitude']) # domain-mean
    vor  = fd.curl(U,V).load() # vorticity to approx. geostrophic vorticity

    _, tmp = xr.broadcast(vor, vor.mean('longitude'))

    vor[:,0,:] = tmp[:,0,:]
    vor[:,-1,:] = tmp[:,-1,:]

    dTHdp= TH.differentiate('vertical')
    RPiP = (Rd * T.mean(['latitude','longitude']) / p / TH)
    S    = - RPiP * dTHdp              # static stability parameter

    ################ traditional form of forcings Eq. (15) ################
    grdthx, grdthy = fd.grad(T , ['X', 'Y'])
    grdvrx, grdvry = fd.grad(vor + f, ['X', 'Y'])

    print('gradients computed')

    F1 = f*((U * grdvrx + V * grdvry).differentiate('vertical'))
    F2 = Rd/p*fd.Laplacian((U * grdthx + V * grdthy))

    print('F1, F2 ready')
    
    #%% prepare lower boundary for inversion
    p3D = T-T+p # broadcast

    FAll = F1.transpose('vertical','latitude','longitude') + F2.transpose('vertical','latitude','longitude')
    FAll = smooth(xr.where(np.isinf(FAll), np.nan, FAll), latitude=85)
    FAll2  =  FAll.where(p<=Psfc)
    print(FAll.sel(vertical=30000, latitude=25, longitude=75).values)
    
    print('FAll, FAll2 ready')

    ## Prepare to invert
    WBC = xr.where(p3D<=Psfc, 0, W).load() # BC: use ERA5 omega as below surface omega, up to 1000 hPa
    iParams = {
        'BCs'      : ['fixed', 'fixed', 'periodic'],
        'mxLoop'   : 5000,
        'tolerance': 1e-16,
    }

    mParams = {'N2': S}

    print('invert_omega() to be executed')

    WQG    = invert_omega(FAll  , dims=['vertical', 'latitude', 'longitude'], iParams=iParams, mParams=mParams)
    # WQvec  = invert_omega(FQvec , dims=['vertical', 'latitude', 'longitude'], iParams=iParams, mParams=mParams)
    
    print('WQG ready')

    WQG2   = invert_omega(FAll2 , dims=['vertical', 'latitude', 'longitude'], iParams=iParams, mParams=mParams, icbc=WBC)
    # WQvec2 = invert_omega(FQvec2, dims=['vertical', 'latitude', 'longitude'], iParams=iParams, mParams=mParams, icbc=WBC)
    print('WQG2 ready')
    
    xr.Dataset(dict(F1=F1,F2=F2,FAll=FAll,FAll2=FAll2,Omega=WQG,Omega_topo=WQG2)).drop('metpy_crs')\
    .transpose('vertical','latitude','longitude')\
    .to_netcdf(dir_out + file_str, mode='w')
    
    ds_u.close(); ds_v.close(); ds_w.close(); ds_t.close(); ds_z.close(); ds_sp.close();
#     del ds_u, ds_v, ds_t, ds_w, ds_z
    gc.collect()
    gc.collect()
# -
# # output string contains dates like 2000_03_01 and not 2000_3_1

