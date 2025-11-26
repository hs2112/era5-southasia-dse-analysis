import os, xarray as xr, numpy as np, pandas as pd, datetime, dask, random, time, netCDF4
from datetime import datetime, timedelta
import gc
# dask.config.set({"array.slicing.split_large_chunks": True})

map_invar_ds = \
{
    'u': 'ds_u', 
    'v': 'ds_v', 
    'w': 'ds_w', 
    't': 'ds_t', 
    'z': 'ds_z', 
}


# +
# map_invar_ds['u']
# -

def merge_files(incl_geopotential, incl_hor_vel, incl_temp, filter_ni, press_levs, ):
    global ds_bflx
    
    if (incl_geopotential==True) and (incl_hor_vel==False):
        ## MATCH time format in both datasets; warnings generated upon merge
        if incl_temp == True:
            ds_bflx = ds_z.merge(ds_t)\
            .sortby(['latitude','longitude','isobaricInhPa','date']).load()
        else: 
            ds_bflx = ds_z\
            .sortby(['latitude','longitude','isobaricInhPa','date']).load()

    if (incl_geopotential==True) and (incl_hor_vel==True):
        ## MATCH time format in both datasets; warnings generated upon merge
        if incl_temp == True:
            ds_bflx = ds_z.merge(ds_t).merge(ds_u).merge(ds_v).merge(ds_w)\
            .sortby(['latitude','longitude','isobaricInhPa','date']).load()
        else: 
            ds_bflx = ds_z.merge(ds_u).merge(ds_v).merge(ds_w)\
            .sortby(['latitude','longitude','isobaricInhPa','date']).load()
        
    elif (incl_geopotential==False) and (incl_hor_vel==False) and ():
        ## MATCH time format in both datasets; warnings generated upon merge
        if incl_temp == True:
            ds_bflx = ds_t.merge(ds_w)\
            .sortby(['latitude','longitude','isobaricInhPa','date']).load()
        if incl_temp == False:
            ds_bflx = ds_w\
            .sortby(['latitude','longitude','isobaricInhPa','date']).load()
            
    elif (incl_geopotential==False) and (incl_hor_vel==True):
        if incl_temp==True:
            ds_bflx = ds_t.merge(ds_u).merge(ds_v).merge(ds_w)\
            .sortby(['latitude','longitude','isobaricInhPa','date']).load()
        else: 
            ds_bflx = ds_u.merge(ds_v).merge(ds_w)\
            .sortby(['latitude','longitude','isobaricInhPa','date']).load()
        
    print(ds_bflx.data_vars)
               
    if filter_ni == True:
        return ds_bflx.sel(
            latitude=slice(nibox_lat[0], nibox_lat[1]), 
            longitude=slice(nibox_lon[0], nibox_lon[1]), 
            isobaricInhPa = press_levs            
                )
    else: 
        print(ds_bflx.sel(isobaricInhPa = press_levs).dims)
        return ds_bflx.sel(isobaricInhPa = press_levs)

    print('isobaricInhPa in ds_bflx is now sorted in ascending, as are lat, lon and date')
def compute_zdiff_fields(ds = xr.Dataset(), create_vars = list()):
    
    if ('isobaricInhPa' in ds.dims):
        if len(ds.isobaricInhPa) > 0:
            if ds.isobaricInhPa[0] < ds.isobaricInhPa[1]:
                print('pressure dim in ascending order')
            else: 
                ds = ds.sortby('isobaricInhPa')
                print('pressure dim NOW in ascending order')

    ds['z_diff_down'] = \
    ds.z - ds.z.shift(isobaricInhPa=-1)
    
    if create_vars == ['theta','theta_grad_z']:

        ds['theta'] = ds['t']*(10**5/(ds['isobaricInhPa']*100))**0.287
        print(ds['theta'].isobaricInhPa.data)

        # diff = upper level - lower level (sorted ascending p levs <=> shift(1) - value)
        ds['theta_diff'] = (ds['theta'] - ds['theta'].shift(isobaricInhPa = -1)).compute()

        ds['theta_grad_z'] = ds['theta_diff']/ds['z_diff_down']
        # theta_diff and z_diff both are (upper level - lower level) differences
    
    # w needs to be positive upwards instead of downwards for the advection term, since z is +ve upwards for equations to hold as it is
    # but w is in Pa s-1 so this does not make sense
    # ds['adv_theta_w'] = ds['w']*ds['theta_grad_z']
    
    return ds


# +
nibox_lon = [68, 78]
nibox_lat = [24, 31]

sa_lon = [60, 98]
sa_lat = [6, 40]

dir_z = '/home/data/lab_hardik/data/ERA5/daily_means/geopotential/data/'
dir_u = '/home/data/lab_hardik/data/ERA5/daily_means/u_component_of_wind/data/'
dir_v = '/home/data/lab_hardik/data/ERA5/daily_means/v_component_of_wind/data/'
dir_t = '/home/data/lab_hardik/data/ERA5/daily_means/temperature/data/'
dir_w = '/home/lab_hardik/daily_means/vertical_velocity/'

out_dir = '/home/data/lab_hardik/data/ERA5/daily_means/data/daily_heat_fluxes/data/'

era5_name_map = {
    
    'u':'u_component_of_wind', 
    'v':'v_component_of_wind', 
    'w':'vertical_velocity', 
    't':'temperature', 
    'z':'geopotential',     
    
}

def read_files(year, month, day, incl_geopotential=True, incl_w = True, incl_u=True, incl_v=True, incl_temp = True):
# def read_files(year, month, day, var_lst=[]):
    
#     for variable in var_lst:
#         globals()[f'ds_{variable}'] = \
#         xr.open_dataset(
#             globals()[f'dir_{variable}'] + f'era5_{variable}_{year}_{month}_{day}.nc'
#         )
#         globals()[f'ds_{variable}']['date'] = globals()[f'ds_{variable}']['date'].astype('datetime64[ns]')
#         if 'level' in globals()[f'ds_{variable}'].dims:
#             globals()[f'ds_{variable}'] = globals()[f'ds_{variable}'].rename({'level':'isobaricInhPa'})
    
    if incl_w == True:
        global ds_w
        ds_w = xr.open_dataset(dir_w + 'era5_vertical_velocity_{}_{}_{}.nc'.format(year, month, day))
        ds_w['date'] = ds_w.date.astype('datetime64[ns]')
        if 'level' in ds_w.dims:
            ds_w = ds_w.rename({'level':'isobaricInhPa'})

    if incl_geopotential==True:
        global ds_z
        ds_z = xr.open_dataset(dir_z + 'era5_geopotential_{}_{}_{}.nc'.format(year, month, day))
        ds_z['date'] = ds_z.date.astype('datetime64[ns]')
        # only if p decreases with index will z increase with index, so that z_diff is positive
        ds_z = ds_z.sortby(['isobaricInhPa'], ascending=False)
        print(ds_z.isobaricInhPa[0] > ds_z.isobaricInhPa[1])
        
        if 'date' not in ds_z.dims:
            ds_z = ds_z.expand_dims('date')

    ### U, V, files
    if incl_u==True:
        global ds_u
        print('here in u land')
        ds_u = xr.open_dataset(dir_u + 'era5_u_component_of_wind_{}_{}_{}.nc'.format(year, month, day))
        ds_u['date'] = ds_u.date.astype('datetime64[ns]')
        if 'level' in ds_u.dims:
            ds_u = ds_u.rename({'level':'isobaricInhPa'})
#         print('ds_u prepped')
        print(ds_u.dims)

    if incl_v==True:
        global ds_v
        ds_v = xr.open_dataset(dir_v + 'era5_v_component_of_wind_{}_{}_{}.nc'.format(year, month, day))
        ds_v['date'] = ds_v.date.astype('datetime64[ns]')
        if 'level' in ds_v.dims:
            ds_v = ds_v.rename({'level':'isobaricInhPa'})
#         print(ds_v.dims)

    if incl_temp == True:
        global ds_t
        ## T, W files
        ds_t = xr.open_dataset(dir_t + 'era5_temperature_{}_{}_{}.nc'.format(year, month, day))
        ds_t['date'] = ds_t.date.astype('datetime64[ns]')


# -
len({'a':'a'})


def merge_files_2(filter_ni, press_levs, var_lst = [], rgn_bnds={}, filter_coords={}):
    global ds_bflx, filtered_ds
    
    if len(var_lst)>1:
        selected_datasets = [globals()[map_invar_ds[var]] for var in var_lst]
        
        ds_bflx = xr.merge(selected_datasets)\
        .sortby(['latitude','longitude','isobaricInhPa','date'])
        
        ds_bflx = ds_bflx.sel(isobaricInhPa = press_levs)
    else:
        ds_bflx = globals()[map_invar_ds[var_lst[0]]]
    
    print(ds_bflx.data_vars)
               
    if filter_ni == True:
        ds_bflx = ds_bflx.sel(
            latitude=slice(nibox_lat[0], nibox_lat[1]), 
            longitude=slice(nibox_lon[0], nibox_lon[1]), 
        ).load()
        return ds_bflx
    if len(filter_coords)>0:
        print('filtering coords using filter_coords{}')
        return ds_bflx.sel(
            latitude=slice(filter_coords['lat']['min'], filter_coords['lat']['max']), 
            longitude=slice(filter_coords['lon']['min'], filter_coords['lon']['max']),
            isobaricInhPa = press_levs            
        )    
    if len(rgn_bnds) > 0:
        print('len(rgn_bnds) > 0')
        filtered_ds = {}
        for region, bounds in rgn_bnds.items():
            ds_bflx = ds_bflx.sortby(['latitude', 'isobaricInhPa'], ascending=True)
            ds_filt = ds_bflx.sel(
            latitude=slice(bounds["lat_min"], bounds["lat_max"]),
            longitude=slice(bounds["lon_min"], bounds["lon_max"]),
                isobaricInhPa = press_levs
            )
            filtered_ds[region] = ds_filt
        print(filtered_ds.keys())
        return filtered_ds
    else: 
        print(ds_bflx.sel(isobaricInhPa = press_levs).dims)
        ds_bflx = ds_bflx.sel(isobaricInhPa = press_levs).load()
        return ds_bflx

    print('isobaricInhPa in ds_bflx is now sorted in ascending, as are lat, lon and date')


var_map = {
    
    'v_component_of_wind':'v',
    'u_component_of_wind':'u', 
    'vertical_velocity':'w', 
    'temperature':'t', 
    'geopotential':'z'
}


# +
# globals()[f'ds_{var_lst[i]}']

# +
def data_prep(start_date_str = '2010-03-01', end_date_str = '2015-06-30',
              incl_geopotential = True,
              incl_w = True,
              incl_u = True,
              incl_v = True,
              incl_hor_vel = True,              
              incl_temp = True,
#               var_lst = ['v_component_of_wind', 'u_component_of_wind', 'vertical_velocity', 'temperature', 'geopotential']
              create_vars = ['theta','theta_grad_z'], 
             mon_lst = [3,4,5,6],
              filter_ni = True, 
              filter_coords={},
              rgn_bnds = {},
              press_levs = [], 
              var_lst=[]
             ):

#     if incl_w == True:
#         global ds_w
#     if incl_temp == True:
#         global ds_t
#     if incl_geopotential==True:
#         global ds_z
#     if incl_hor_vel==True:
#         global ds_u, ds_v
        
    for variable in var_lst:
        globals()[f'ds_{variable}'] = xr.Dataset()

    k=0
    for element in list(pd.date_range(start_date_str, end_date_str)):
    # for element in list(pd.date_range('2010-03-01','2022-06-30')):
        year = element.year
        month = element.month
        day = element.day

        if (month in mon_lst):
            print(year, month, day)
            
            read_files(
                year, month, day, incl_geopotential=incl_geopotential, 
                incl_w=incl_w, incl_u=incl_u, incl_v=incl_v, incl_temp=incl_temp
            )
            
            #### Density
            R = 287 # J K-1 kg-1

            ### Merge all
            # Here we sort ALL DIMS (esp. isobaricInhPa) in ascending
            global ds_bflx, filtered_ds
#             ds_bflx = merge_files(
#                 incl_geopotential=incl_geopotential, incl_hor_vel=incl_hor_vel, incl_temp=incl_temp, 
#                 filter_ni = filter_ni, press_levs = press_levs
#             )

            ds_bflx = merge_files_2(
                var_lst = var_lst, filter_ni = filter_ni, press_levs = press_levs, rgn_bnds = rgn_bnds
            )
    
            if len(rgn_bnds)>0:
                print(filtered_ds.items())
                for region, data in filtered_ds.items():
                    if 't' in data.data_vars:
                        data['density'] = (data['isobaricInhPa']*100/(R*data['t'])).compute()
                        filtered_ds[region]['density'] = data['density']
                    
                    ### Variable creation
                    if incl_geopotential==True:
                        filtered_ds[region] = compute_zdiff_fields(ds = filtered_ds[region], create_vars = create_vars)
                        
                    filtered_ds[region] = filtered_ds[region].compute()

                if k==0:
                    filtered_ds_mst = filtered_ds
                else:
                    for region, data in filtered_ds.items():
                        print(type(data))
                        print(type(filtered_ds_mst[region]))
                        filtered_ds_mst[region] = xr.merge([filtered_ds_mst[region], data])
#                     for i in range(len(filtered_ds_mst)):
#                         filtered_ds_mst.values()[i] = xr.merge([filtered_ds_mst.values()[i], filtered_ds])

            else:
                if 't' in ds_bflx.data_vars:
                    ds_bflx['density'] = (ds_bflx['isobaricInhPa']*100/(R*ds_bflx['t'])).compute()

                ### Variable creation
                if incl_geopotential==True:
                    ds_bflx = compute_zdiff_fields(ds = ds_bflx, create_vars = create_vars)

                #### NOTE: Think of each 'gridbox' as the first quadrant in the xy plane with 'gridpoint' at origin

                ds_bflx = ds_bflx.compute()

                if k==0:
                    ds_bflx2 = ds_bflx
                else:
                    ds_bflx2 = xr.merge([ds_bflx2, ds_bflx])

            gc.collect()
            gc.collect()

            k = k+1

#             ds_bflx.to_netcdf(out_dir + 'heat_flxs_{}_{}_{}_nibox.nc'.format(year, month, day), 
#                        mode='w')
    if len(rgn_bnds)>0:
        return filtered_ds_mst
    else: 
        return ds_bflx2


# +
# dir_clmt = '/home/data/lab_hardik/data/ERA5/climatology/'

# def compute_anomalies(ds, incl_geopotential=False, incl_hor_vel=False):
#     ds['t_anom'] = \
#     ds['t'].groupby(ds['t'].date.dt.strftime('%d-%b')) - \
#     xr.open_dataset(dir_clmt + 't_3D/dly_clmtlgy_stdPlevs_prmnsn_temperature_1990_2022.nc')\
#     .rename({'t':'t_clmt'})['t_clmt']

#     if incl_geopotential==True:
#         ds['z_anom'] = ds['z'].groupby(ds['z'].date.dt.strftime('%d-%b')) - \
#         xr.open_dataset(dir_clmt + 'z_3D/dly_clmtlgy_stdPlevs_prmnsn_geopotential_1990_2022.nc')\
#         .rename({'z':'z_clmt'})['z_clmt']

#     if incl_hor_vel == True:
#         ds['u_anom'] = ds['u'].groupby(ds['z'].date.dt.strftime('%d-%b')) - \
#         xr.open_dataset(dir_clmt + 'u_3D/dly_clmtlgy_stdPlevs_prmnsn_u_component_of_wind_1995_2022.nc')\
#         .rename({'u':'u_clmt'})['u_clmt']

#         ds['v_anom'] = ds['v'].groupby(ds['v'].date.dt.strftime('%d-%b')) - \
#         xr.open_dataset(dir_clmt + 'v_3D/dly_clmtlgy_stdPlevs_prmnsn_v_component_of_wind_1995_2020.nc')\
#         .rename({'v':'v_clmt'})['v_clmt']
    
#     return ds

# -

def compute_x_y_step_lengths():
    global del_x, del_y
    del_y = xr.DataArray(np.repeat(np.pi*6378*10**3/721, ds_bflx.dims['longitude']), 
                         dims='longitude',
                         coords = {'longitude': ds_bflx.longitude.data})

    # del_x is step length associated with each latitude
    del_x = 2*np.pi*(6378*10**3*np.cos(ds_bflx.latitude*np.pi/180))/1440
    del_x.name = 'del_x'


steps_x = 4*10

# +
Cp = 1005 # J K-1 kg-1
g = 9.80665

Rearth = 6378*10**3
del_lambda = 0.25*np.pi/180
del_phi = 0.25*np.pi/180

# omega_to_w() returns w_up or simply w 
def omega_to_w(ds = xr.Dataset(), Rlat = xr.DataArray(), Rearth = int, del_lambda = float, del_phi = float):
    if ds.isobaricInhPa[0] < ds.isobaricInhPa[1]:
        print('pres ord asc')

        del_z_p = (ds.z.shift(isobaricInhPa=-1) - ds.z)/g
        del_p = 100*(ds.isobaricInhPa.shift(isobaricInhPa=-1) - ds.isobaricInhPa)
        
        dp_dz = del_p/del_z_p

    dz_dx_p = (ds.z.shift(longitude=-1) - ds.z)/(g*Rlat*del_lambda)
    dz_dy_p = (ds.z.shift(latitude=-1) - ds.z)/(g*Rearth*del_phi)
    dz_dt = (ds.z.shift(date=-1) - ds.z)/(g*86400)

    adv_u = -ds['u']*dp_dz*dz_dx_p

    adv_v = -ds['v']*dp_dz*dz_dy_p

    time_deriv = -dp_dz*dz_dt
    
    return (ds['w'] - adv_u - adv_v - time_deriv)/dp_dz # this is w_up

# -



# ## Climatology

from datetime import datetime as dt, timedelta
import datetime, time


# +
def assign_dates_climatology(ds, year=1900, strftime_col = 'strftime'):
    
#     date_arr = [dt.strptime(dt.strptime(ds[strftime_col].data[i], '%d-%b').strftime('{}-%m-%d'.format(year)), '%Y-%m-%d')
#                     for i in range(len(ds[strftime_col]))]
    date_arr = \
    [dt.strptime(dt.strptime(ds[strftime_col].data[i] + '-' + str(year), '%d-%b-%Y')
                .strftime('{}-%m-%d'.format(year)), '%Y-%m-%d')
    for i in range(len(ds[strftime_col]))]

    return ds.assign_coords(
                {'date': 
                 (strftime_col, date_arr)
                }
            ).sortby('date')
# \
#     .to_dataset()
# .swap_dims({'strftime':'date'})


# +
# Assign 2016 dates to ds so that it can be multiplied with 2016 masses for weighted sum 
# (although it doesn't really make much of a difference)

def assign_dates_climatology_2(ds):
    date_arr = [dt.strptime(dt.strptime(ds.strftime.data[i], '%d-%b').strftime('1800-%m-%d'), '%Y-%m-%d')
                    for i in range(len(ds.strftime))]
    return ds.assign_coords(
        {'date': 
         ('strftime', date_arr)
        }
    )\
    .to_dataset().sortby('date').swap_dims({'strftime':'date'})\
    *globals()[f'mass_wgts_{year_sample}']


# -

import variables
from variables import *




