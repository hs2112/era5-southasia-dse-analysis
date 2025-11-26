import pandas as pd, os, xarray as xr, numpy as np, gc
from datetime import datetime as dt, timedelta
import datetime, time
import h5netcdf
from dask.diagnostics import ProgressBar 


# +
def day_sort(s):
    return int(s.split('_')[-1].split('.')[0])

def mon_sort(s):
    return int(s.split('_')[-2])

def year_sort(s):
    return int(s.split('_')[-3])


# -

def prepare_z_files_1982_4():
    ds_1982_4 = xr.open_mfdataset('/home/data/lab_hardik/data/ERA5/daily_means/geopotential/data/era5_geopotential_1982_4_*.nc')
#     ds_1982_4 = ds_1982_4.compute()
    ds_1982_4 = ds_1982_4.rename({'level':'isobaricInhPa'})
    
    return ds_1982_4



import re 
def starts_with_substring_and_number(s, substring): 
    pattern = rf"^{re.escape(substring)}\d+" 
    return re.match(pattern, s) is not None 



# +
def prepare_z_ds(year_min = 1980, year_max = 1994): # not to be used for any other period but 1980-1994
    from datetime import datetime as dt
    import calendar
    import zarr, s3fs, xarray as xr, os
    # AWS S3 path
    minio_path = 'http://192.168.1.237:9000/' 
    # Initilize the S3 file system
    mn =s3fs.S3FileSystem(key='d0d250b2541ac33f4660',
                          secret='2fb32d964768bc94a3c0',
                         client_kwargs={'endpoint_url': minio_path})
    bucket = 'era5'
    store = s3fs.S3Map(root=bucket, s3=mn, check=False)
    print(mn.ls(bucket))

    ds_tst = xr.open_zarr(store=store, group='era5/pressure_levels/geopotential', consolidated=True)
    ds_tst['year'] = ds_tst['time.year']
    
    
    ds_z = ds_tst.sel(
        time = (ds_tst.time.dt.year.isin(range(year_min, year_max + 1)) & 
                (ds_tst.time.dt.month.isin([3,4,5,6]))), 
    )

    ds_z1 = ds_z['z'].groupby(ds_z.time.dt.date).mean()
    ds_z1['date'] = ds_z1.date.astype('datetime64')
    
    ds_1982_4 = prepare_z_files_1982_4()
    
    ds_z2 = xr.concat([ds_z1.drop(['number','step']), ds_1982_4['z']], dim='date')
    ds_z2 = ds_z2.sortby('date')
    return ds_z2

#     ds_z2 = ds_z2.persist()
# -


dir_v = '/home/data/lab_hardik/data/ERA5/daily_means/v_component_of_wind/data/'
dir_u = '/home/data/lab_hardik/data/ERA5/daily_means/u_component_of_wind/data/'
dir_t = '/home/data/lab_hardik/data/ERA5/daily_means/temperature/data/'
dir_z = '/home/data/lab_hardik/data/ERA5/daily_means/geopotential/data/'
# variable can be 'u', 'v','t','z'
def prepare_files(variable = 'v'):
    globals()[f'{variable}_files'] = os.listdir(globals()[f'dir_{variable}'])

    globals()[f'{variable}_files'] = [s for s in globals()[f'{variable}_files'] if (('.grib' in s) | ('.nc' in s)) & ('idx' not in s)]
    
    globals()[f'{variable}_files'] = \
    [
        s for s in globals()[f'{variable}_files'] if int(s.split('_')[-3])>= (1980 if variable != 'z' else 1995) 
         and (int(s.split('_')[-3]) <= 2022) 
         and int(s.split('_')[-2]) in [3,4,5,6]
    ]

    globals()[f'{variable}_files'].sort(key = day_sort)
    globals()[f'{variable}_files'].sort(key = mon_sort)
    globals()[f'{variable}_files'].sort(key = year_sort)

    print(len(globals()[f'{variable}_files']))
    print(globals()[f'{variable}_files'][:6])

    # globals()[f'{variable}_files']_2 = [s for s in globals()[f'{variable}_files'] if int(s.split('_')[-2]) in [3,4,5,6]]

    globals()[f'df_{variable}_files'] = \
    pd.DataFrame(
        {
            'filename': globals()[f'{variable}_files'], 
            'date': [dt(int(s.split('_')[-3]), int(s.split('_')[-2]), int(s.split('_')[-1].split('.')[0])) for s in globals()[f'{variable}_files']]
        }
    ).sort_values('date')





# +
# ds_z_1980_1994 is prefiltered for 600 to 900 when passed to the function

def filt_comp_files(variable = 'v', dates = [], press = 300, ds_z_1980_1994 = xr.DataArray(), 
                   ymin=10, ymax=60, xmin=0, xmax=120, compute=True):
    
    print(globals()[f'df_{variable}_files'].shape)

    dates_1980_1994 = [pd.to_datetime(s).to_datetime64() for s in dates if s.year in range(1980,1995)]
    dates_1995_2022 = [pd.to_datetime(s).to_datetime64() for s in dates if s.year in range(1995, 2023)]
    
    print(len(dates_1980_1994), 'dates bw 1980 and 1994')
    print(len(dates_1995_2022), 'dates bw 1995 and 2022')
    
    if variable == 'z':
        if len(dates_1995_2022) > 0:
            if compute == True:
                ds2 = filt_dly_mean_files(
                    variable = 'z', 
                    dates = dates_1995_2022, 
                    press = press
                )\
                .sel(latitude=slice(ymax,ymin), longitude=slice(xmin,xmax))\
                .compute()
            else:
                with ProgressBar():
                    ds2 = filt_dly_mean_files(
                        variable = 'z', 
                        dates = dates_1995_2022, 
                        press = press
                    )\
                    .sel(latitude=slice(ymax,ymin), longitude=slice(xmin,xmax))
#                     \
#                     .persist()
            
            print('ds w/ z dates b/w 95 and 22 is ready')

        if len(dates_1980_1994) > 0:
            print('dates_1980_1994', dates_1980_1994)
            # ds_z_1980_1994 is prefiltered for 600 to 900 when passed to the function
            ds1 = ds_z_1980_1994.sel(
                date = ds_z_1980_1994.date.isin(dates_1980_1994), 
                isobaricInhPa=press
            )\
            .sel(latitude=slice(ymax,ymin), longitude=slice(xmin,xmax))\
            .to_dataset()
            
            print('persist ds1 -- z dates b/w 1980 to 1994')
#             with ProgressBar():
#                 ds1 = ds1.persist()
#                 print('ds1 is persisted')

            if len([s for s in dir() if 'ds2' in s]) > 0:
                globals()[f'ds_{variable}_comp'] = xr.concat([ds1, ds2], dim='date')       
            else: 
                globals()[f'ds_{variable}_comp'] = ds1

        if len([s for s in dir() if 'ds1' in s]) == 0 and len([s for s in dir() if 'ds2' in s]) > 0:
            globals()[f'ds_{variable}_comp'] = ds2
        
        print(globals()[f'ds_{variable}_comp'].date.data)
        print(globals()[f'ds_{variable}_comp'].dims)
    
    else: 
        if compute == True:
            globals()[f'ds_{variable}_comp'] = filt_dly_mean_files(variable = variable, dates = dates, press = press)\
            .sel(latitude=slice(ymax,ymin), longitude=slice(xmin,xmax))\
            .compute()
        else:
            with ProgressBar():
                globals()[f'ds_{variable}_comp'] = filt_dly_mean_files(variable = variable, dates = dates, press = press)\
                .sel(latitude=slice(ymax,ymin), longitude=slice(xmin,xmax))
#                 \
#                 .persist()

#     with ProgressBar(): 
#         globals()[f'ds_{variable}_comp'] = globals()[f'ds_{variable}_comp'].compute()

    gc.collect()
    gc.collect()
    gc.collect()
    
    return globals()[f'ds_{variable}_comp']



# +
def filt_dly_mean_files(variable = 'v', dates = [], press = 300, chunks_lon = 481, chunks_lat = 201):
    globals()[f'df_{variable}_files2'] = \
    globals()[f'df_{variable}_files']\
    .loc[pd.to_datetime(globals()[f'df_{variable}_files'].date).dt.date.astype('datetime64[ns]').isin(dates)]
    
    print(globals()[f'df_{variable}_files2'].shape)
    
    for i in range(len(globals()[f'df_{variable}_files2']['filename'])):
        print('i=',i)
        print(globals()[f'df_{variable}_files2']['filename'].iloc[i])
        
#         tmp = xr.open_dataset(
#                 globals()[f'dir_{variable}'] + 
#                 globals()[f'df_{variable}_files2']['filename'].iloc[i]
#             )
#         os.chdir(globals()[f'dir_{variable}'])

        tmp = xr.open_dataset(
            globals()[f'dir_{variable}'] + 
            globals()[f'df_{variable}_files2']['filename'].iloc[i], 
            engine='h5netcdf',
            chunks = {'longitude':chunks_lon, 'latitude':chunks_lat}
            )
            
        if 'level' in tmp.dims:
            tmp = tmp.rename({'level':'isobaricInhPa'})

        tmp = tmp.sortby('isobaricInhPa', ascending=False).sel(isobaricInhPa = press)
        if 'date' not in tmp.dims:
            tmp = tmp.expand_dims('date')
        
#         with ProgressBar():
#             tmp = tmp.compute()

        if i==0: 
#             globals()[f'ds_{variable}_comp'] = tmp
            ds_var = tmp
            
        else:             
#             globals()[f'ds_{variable}_comp'] = xr.concat([globals()[f'ds_{variable}_comp'], tmp], dim='date')
            ds_var = xr.concat([ds_var, tmp], dim='date')
    
    return ds_var


# +
# press = slice(600,900) will return pressure as a dimension with these p levels

def create_anoms(variable='z', press=300, ds=xr.Dataset(), clmt_file_map = dict()):
    clmt_file = '/home/data/lab_hardik/data/ERA5/climatology/{}_3D/'.format(variable) + clmt_file_map[variable]
    
    da_clmt = xr.open_dataset(clmt_file)
    
    print(list(da_clmt.data_vars))
    print(da_clmt['{}_dly_clmt_10D_roll'.format(variable)].dims)
    
#     da_clmt.rename({variable:'{}_clmt'.format(variable)}).sel(isobaricInhPa=press)['{}_clmt'.format(variable)]

    ds['{}_anom_10D'.format(variable)] =\
    ds[variable].groupby(ds.date.dt.strftime('%d-%b')) - \
    da_clmt['{}_dly_clmt_10D_roll'.format(variable)].rename({'clmt_roll_strftime':'strftime'})

#     ds = ds['{}_anom_10D'.format(variable)].to_dataset()

    return ds


# +
# clmt_file_map = {
    
#     'u': 'dly_clmtlgy_stdPlevs_prmnsn_u_component_of_wind_1995_2022.nc',
#     'v': 'dly_clmtlgy_stdPlevs_prmnsn_v_component_of_wind_1995_2022.nc',
#     'z': 'dly_clmtlgy_stdPlevs_prmnsn_geopotential_1990_2022.nc', 
#     't': 'dly_clmtlgy_stdPlevs_prmnsn_temperature_1990_2022.nc'    
# }

# rolling_clmt_file_map = {
    
#     'u':'global_rolling_clmtlgy_prmnsn_u_component_of_wind_1980_2022.nc',
#     'v': 'global_rolling_clmtlgy_prmnsn_v_component_of_wind_1980_2022.nc', 
#     'z': 'Globe_rolling_clmtlgy_prmnsn_geopotential_1980_2022.nc', # feb added version remains
#     't': 'SA_rolling_clmtlgy_prmnsn_temperature_1980_2022.nc', # feb added version remains
#     'w': 'global_rolling_clmtlgy_prmnsn_vertical_velocity_1980_2022.nc'
# }

# +
# # press = slice(600,900) will return pressure as a dimension with these p levels

# def create_anoms(variable='z', press=300):
#     clmt_file = '/home/data/lab_hardik/data/ERA5/climatology/{}_3D/'.format(variable) + clmt_file_map[variable]
    
#     da_clmt = xr.open_dataset(clmt_file)\
#     .rename({variable:'{}_clmt'.format(variable)}).sel(isobaricInhPa=press)['{}_clmt'.format(variable)]

#     globals()[f'ds_{variable}_comp']['{}_anom_10D'.format(variable)] =\
#     globals()[f'ds_{variable}_comp'][variable]\
#     .groupby(globals()[f'ds_{variable}_comp'][variable].date.dt.strftime('%d-%b')) - \
#     da_clmt

# #     globals()[f'ds_{variable}_comp'] = globals()[f'ds_{variable}_comp']['{}_anom_10D'.format(variable)].to_dataset()

#     return globals()[f'ds_{variable}_comp']
# -




from dask.diagnostics import ProgressBar 


# +
def inner_plot_composite(
    da_plt = xr.DataArray(), da_cont=xr.DataArray(), da_u=xr.DataArray(), da_v=xr.DataArray(), 
    plot_quivs = True, 
    plot_conts = True,
    xmin = 0, xmax = 120, ymin = 5, ymax = 65, 
    cont_divisions=100, 
    coarsen_pts = 8,
    t_type = 'anomaly',
    cbar_lab = '',
    scale_quiv= 2, 
    ax=np.ndarray(shape=(3,2)), i = int(), j=int(), vmin_in = np.nan, vmax_in = np.nan,
    
    cmapp = 'coolwarm',
    title_str = '',
    save_str = '',
):
    
    da_plt = da_plt.coarsen(latitude=coarsen_pts, boundary='trim', side='right').mean()\
        .coarsen(longitude=coarsen_pts, boundary='trim', side='right').mean()
    da_plt = da_plt.assign_coords(latitude=np.round(da_plt.latitude), longitude=np.round(da_plt.longitude))
    
    with ProgressBar(): 
        da_plt = da_plt.compute()
        
    vmin = da_plt.quantile(0.05) if vmin_in != vmin_in else vmin_in
    vmax = da_plt.quantile(0.95) if vmax_in != vmax_in else vmax_in
    
    print('vmin = ', vmin)
    print('vmax = ', vmax)

    da_plt.plot(
        ax=ax, 
#         vmax = vmax if vmax > np.abs(vmin) else np.abs(vmin), 
        vmax = vmax, 
        cmap = cmapp, 
#         vmin = vmin if np.abs(vmin) > vmax else -1*vmax, 
        vmin = vmin, 
        add_colorbar = True, 
        cbar_kwargs = dict(
            orientation='horizontal',
            fraction=0.05, 
            pad=0.05, 
            label = 'Temperature anomaly (Celsius)' if t_type == 'anomaly' else 
            'Temp (Celsius)' if t_type == 'full' else 's\' (J kg-1)' if t_type=='dse_anom' else 
            cbar_lab if t_type=='' else np.nan
        )
    )
#                        label = 'v component of wind (m $\mathregular{s^{-1}}$)'))
    
    if plot_quivs == True:

        da_u = da_u.coarsen(latitude=coarsen_pts, boundary='trim', side='right').mean()\
            .coarsen(longitude=coarsen_pts, boundary='trim', side='right').mean()
        da_u = da_u.assign_coords(latitude=np.round(da_u.latitude), longitude=np.round(da_u.longitude))

        da_v = da_v.coarsen(latitude=coarsen_pts, boundary='trim', side='right').mean()\
                .coarsen(longitude=coarsen_pts, boundary='trim', side='right').mean()
        da_v = da_v.assign_coords(latitude=np.round(da_v.latitude), longitude=np.round(da_v.longitude))
    
        with ProgressBar(): 
            da_u = da_u.compute()
            da_v = da_v.compute()

        print("u_min = ", da_u.min().values, "u_max =", da_u.max().values)
        print("v_min = ", da_v.min().values, "v_max =", da_v.max().values)
    
        ax.quiver(da_u.longitude, da_u.latitude, da_u.values, da_v.values)

    if plot_conts == True:
        da_cont = da_cont.coarsen(latitude=coarsen_pts, boundary='trim', side='right').mean()\
            .coarsen(longitude=coarsen_pts, boundary='trim', side='right').mean()
        da_cont = da_cont.assign_coords(latitude=np.round(da_cont.latitude), longitude=np.round(da_cont.longitude))

        da_cont = da_cont.compute()        
        contmin = da_cont.quantile(0)
        contmax = da_cont.quantile(1)
        
        plot_contours(da_cont = da_cont, lev_min=contmin, lev_max=contmax, 
                      lev_diff = (contmax-contmin)/cont_divisions, ax=ax)

    ax.add_patch(mpatches.Rectangle(xy=[68, 24], width=10, height=7,
                                    facecolor='none', 
                                    edgecolor='black', 
                                    linewidth=2, 
                                    transform=ccrs.PlateCarree()
                                   ))
    
    ax.set_extent((xmin,xmax,ymin,ymax), crs=ccrs.PlateCarree())

    # map_df.boundary.plot(ax=ax)
    ax.set_title(title_str)
    #     ax.set_title('T2m anomalies > $2^\circ$C' if i==0 else 'T2m anomalies < $0^\circ$C' if i==1 else np.nan)

    ax.coastlines()
    
#     del da_plt, da_u, da_v, da_cont
    gc.collect()
    gc.collect()
#     plt.close()
#     plt.cla()
# -





import numpy as np
np.arange(5, 65+1, 20)

# +
import matplotlib.ticker as mticker

from matplotlib.ticker import FixedLocator


# -

def plot_composite(
    da_plt = xr.DataArray(), da_cont=xr.DataArray(), da_u=xr.DataArray(), da_v=xr.DataArray(), 
    xmin = 0, xmax = 120, ymin = 5, ymax = 65, figsize=(16,5), 
    contours = 'on', # 'off'
    cbar_pad = 0.05,
    cont_divisions=100,
    coarsen_pts = 8,
    vmax_in = np.nan, 
    vmin_in = np.nan,
    t_type = 'anomaly',
    cmapp = 'coolwarm',
    scale_quiv= 2, 
    title_str = '', 
    save_str = '',
    latlon_tickcol = 'dimgray', 
    latlon_ticksz = 15, 
    dir_out = '/home/data/lab_hardik/heatwaves/ERA5/analyses/phase_relations/'
):
    
    global vmin, vmax

    fig, ax = plt.subplots(nrows=1, ncols=1, 
                        sharey=False,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=figsize)
    
    vmin = vmin_in if vmin_in == vmin_in else da_plt.quantile(0)
    vmax = vmax_in if vmax_in == vmax_in else da_plt.quantile(1)
    
    print('vmin = ',vmin)
    print('vmax = ',vmax)

    da_plt.plot(ax=ax, 
                vmax = vmax if vmax > np.abs(vmin) else np.abs(vmin), 
                cmap = cmapp, 
                vmin = vmin if np.abs(vmin) > vmax else -1*vmax, 
                add_colorbar=True, 
                cbar_kwargs=dict(
                    orientation='horizontal',
                    fraction=0.05, 
                    pad=cbar_pad, 
                    label = 'Temperature anomaly (Celsius)' if t_type == 'anomaly' 
                    else 'Temp (Celsius)' if t_type == 'full' 
                    else '$\mathcal{S}\'$ (J kg-1)' if t_type=='dse_anom' 
                    else np.nan)
               )
#                  label = 'v component of wind (m $\mathregular{s^{-1}}$)'))

    if contours == 'on':
        contmin = da_cont.quantile(0)
        contmax = da_cont.quantile(1)

        plot_contours(da_cont = da_cont, lev_min=contmin, lev_max=contmax, lev_diff = (contmax-contmin)/cont_divisions, ax=ax)


    da_u = da_u.coarsen(latitude=coarsen_pts, boundary='trim', side='right').mean()\
            .coarsen(longitude=coarsen_pts, boundary='trim', side='right').mean()
    da_u = da_u.assign_coords(latitude=np.round(da_u.latitude), longitude=np.round(da_u.longitude))

    da_v = da_v.coarsen(latitude=coarsen_pts, boundary='trim', side='right').mean()\
            .coarsen(longitude=coarsen_pts, boundary='trim', side='right').mean()
    da_v = da_v.assign_coords(latitude=np.round(da_v.latitude), longitude=np.round(da_v.longitude))

    ax.quiver(da_u.longitude, da_u.latitude, da_u.values, da_v.values, scale=scale_quiv)

    ax.add_patch(mpatches.Rectangle(xy=[68, 24], width=10, height=7,
                                    facecolor='none', 
                                    edgecolor='black', 
                                    linewidth=2, 
                                    transform=ccrs.PlateCarree()
                                   ))
    # map_df.boundary.plot(ax=ax)
    ax.set_title(title_str)
    #     ax.set_title('T2m anomalies > $2^\circ$C' if i==0 else 'T2m anomalies < $0^\circ$C' if i==1 else np.nan)

    ax.coastlines()
    ax.set_xticks(np.arange(xmin, xmax+1, 20), crs=ccrs.PlateCarree())
    ax.set_xticklabels(ax.get_xticks(), rotation=90)

    ax.set_yticks(np.arange(ymin, ymax+1, 20), crs=ccrs.PlateCarree())
    ax.set_extent([xmin,xmax,ymin,ymax], crs=ccrs.PlateCarree())
    
    # Format the tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}°E'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}°N'))

    # Change the color of tick labels
    ax.tick_params(axis='x', which='major', labelcolor='dimgray')  # Change x-axis tick label color
    ax.tick_params(axis='y', which='major', labelcolor='dimgray')  # Change y-axis tick label color

    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.tight_layout()
    
    if save_str != '':
        plt.savefig(dir_out + '{}'.format(save_str), bbox_inches='tight')
    # plt.savefig('/home/data/lab_hardik/analysis/VertStruct/prmnsn_v_comps_{}_{}hPa.png'.format(year, press), bbox_inches='tight')

import numpy as np
xmin = 30; xmax = 120; ymin = 10; ymax = 70
print(np.arange(xmin,xmax,(xmax-xmin)/3))
print(np.arange(ymin,ymax,(ymax-ymin)/3))


# +
from matplotlib.axes import Axes
def plot_composite_final_plot(
    da_plt = xr.DataArray(), da_cont=xr.DataArray(), da_u=xr.DataArray(), da_v=xr.DataArray(), 
    xmin = 0, xmax = 120, ymin = 5, ymax = 65, 
#     figsize=(16,5), 
    cbar_on = True,
    axi = Axes,
    contours = 'on', # 'off'
    cbar_pad = 0.05,
    cont_divisions=100,
    coarsen_pts = 8,
    vmax_in = np.nan, 
    vmin_in = np.nan,
    t_type = 'anomaly',
    cmapp = 'coolwarm',
    scale_quiv= 2, 
    title_str = '', 
#     save_str = '',
    latlon_tickcol = 'dimgray', 
    latlon_ticksz = 15, 
    xtickrot = 90,
    ytickrot=0,
    num_xticks=3,
    xtick_offset = 30,
    
    num_yticks=3,
    ytick_offset = 0,
#     dir_out = '/home/data/lab_hardik/heatwaves/ERA5/analyses/phase_relations/'
):
    
    global vmin, vmax

    vmin = vmin_in if vmin_in == vmin_in else da_plt.quantile(0)
    vmax = vmax_in if vmax_in == vmax_in else da_plt.quantile(1)
    
    print('vmin = ',vmin)
    print('vmax = ',vmax)

    if cbar_on == True:
        da_plt.plot(
            ax=axi, 
            vmax = vmax if vmax > np.abs(vmin) else np.abs(vmin), 
            cmap = cmapp, 
            vmin = vmin if np.abs(vmin) > vmax else -1*vmax, 
            add_colorbar=cbar_on, 
            cbar_kwargs=dict(
                orientation='horizontal',
                fraction=0.05, 
                pad=cbar_pad, 
                label = 'Temperature anomaly (Celsius)' if t_type == 'anomaly' 
                else 'Temp (Celsius)' if t_type == 'full' 
                else '$\mathcal{S}\'$ (J kg-1)' if t_type=='dse_anom' 
                else np.nan),
            edgecolor='black',
        )
    else:
        da_plt.plot(
            ax=axi, 
            vmax = vmax if vmax > np.abs(vmin) else np.abs(vmin), 
            cmap = cmapp, 
            vmin = vmin if np.abs(vmin) > vmax else -1*vmax, 
            add_colorbar=False
       )


    if contours == 'on':
        contmin = da_cont.quantile(0)
        contmax = da_cont.quantile(1)
        plot_contours(
            da_cont = da_cont, 
            lev_min=contmin, 
            lev_max=contmax, 
            lev_diff = (contmax-contmin)/cont_divisions, 
            ax=ax
        )

    da_u = da_u.coarsen(latitude=coarsen_pts, boundary='trim', side='right').mean()\
            .coarsen(longitude=coarsen_pts, boundary='trim', side='right').mean()
    da_u = da_u.assign_coords(latitude=np.round(da_u.latitude), longitude=np.round(da_u.longitude))

    da_v = da_v.coarsen(latitude=coarsen_pts, boundary='trim', side='right').mean()\
            .coarsen(longitude=coarsen_pts, boundary='trim', side='right').mean()
    da_v = da_v.assign_coords(latitude=np.round(da_v.latitude), longitude=np.round(da_v.longitude))

    axi.quiver(da_u.longitude, da_u.latitude, da_u.values, da_v.values, scale=scale_quiv)

    axi.add_patch(mpatches.Rectangle(xy=[68, 24], width=10, height=7,
                                    facecolor='none', 
                                    edgecolor='black', 
                                    linewidth=2, 
                                    transform=ccrs.PlateCarree()
                                   ))
    # map_df.boundary.plot(ax=ax)
    
    ## Format them all
    axi.set_title(title_str)
    axi.coastlines()
    axi.set_xticks(np.arange(xmin, xmax, (xmax - (xmin + xtick_offset))/num_xticks), crs=ccrs.PlateCarree())
    axi.set_xticklabels(axi.get_xticks(), rotation=xtickrot, **{'fontsize': latlon_ticksz})

#     axi.set_yticks(np.arange(ymin, ymax+1, 20), crs=ccrs.PlateCarree())
    if num_yticks == 3:
        axi.set_yticks([10, 30, 50], crs=ccrs.PlateCarree())
    
    axi.set_yticklabels(axi.get_yticks(), rotation=ytickrot, **{'fontsize': latlon_ticksz})
    axi.set_extent([xmin,xmax,ymin,ymax], crs=ccrs.PlateCarree())
    
    # Format the tick labels
    axi.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}°E'))
    axi.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}°N'))

    # Change the color of tick labels
    axi.tick_params(axis='x', which='major', labelcolor=latlon_tickcol, )  # Change x-axis tick label color
    axi.tick_params(axis='y', which='major', labelcolor=latlon_tickcol, )  # Change y-axis tick label color

    axi.set_xlabel('')
    axi.set_ylabel('')

#     plt.tight_layout()
    
#     if save_str != '':
#         plt.savefig(dir_out + '{}'.format(save_str), bbox_inches='tight')

# +
import cartopy.crs as ccrs, matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

def plot_contours(da_cont = xr.DataArray(), lev_min=-1500, lev_max=1500, lev_diff = 100, ax=np.ndarray(shape=(3,2)), 
                 plot_dims=['longitude','latitude'], geoaxes=True, color = 'green', lw=1):
    
    num_levs = int((lev_max-lev_min)/lev_diff + 1)
    print('num_levs = ', num_levs)
    cs2 = ax.contour(
        da_cont[plot_dims[0]], 
        da_cont[plot_dims[1]], 
        da_cont, 
        levels=np.linspace(lev_min, lev_max, num_levs),
        colors=color,
        linewidths=lw, 
#         linestyles='solid', 
        transform=ccrs.PlateCarree() if geoaxes==True else None
    )

    for line, lvl in zip(cs2.collections, cs2.levels):
        if lvl < 0:
            line.set_linestyle('--')
        elif lvl == 0:
            line.set_linestyle(':')
        else:
            # Optional; this is the default.
            line.set_linestyle('-')

    plt.clabel(cs2, 
               levels=cs2.levels, 
               fontsize=8,
               inline=1, 
               inline_spacing=3, 
               fmt='%.1f',
               rightside_up=True, 
               use_clabeltext=True
              )
# -




# +
import matplotlib.animation as animation
import geopandas as gpd
import shapefile as shp
import seaborn as sns
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
from shapely.geometry import Point
sns.set_style('whitegrid')


Writer = animation.writers['ffmpeg']
writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)

def plot_contf(da_plt=xr.DataArray(), ax=np.ndarray(shape=(3,2)), vmax=5, vmin=-5):
    return ax.contourf(
    da_plt.longitude, 
    da_plt.latitude, 
    da_plt.values, 
    levels=np.linspace(vmin, vmax, 21), 
    vmax = vmax if vmax > np.abs(vmin) else -1*vmin, 
    cmap = 'coolwarm', 
    vmin = -1*vmax if vmax > np.abs(vmin) else vmin,
)


# +
import matplotlib as mpl

mpl.rc('image', cmap='coolwarm')


# +
# prereq: r must be greater than 1
def bicol_panel_rownum(i):
    rownum = int(np.floor(i/2))
    return rownum

def bicol_panel_colnum(i):
    rownum = bicol_panel_rownum(i)
    colnum = i - 2*rownum
    return colnum
# -





# +
def path_dates_movie(ds, path_desc, mon_str, 
                     xmin=0, xmax=120, ymin=5, ymax=50,
                     cont_min=-1500, cont_max=1500, cont_spacing=200,
                     v_field = 'v', 
                     u_field = 'u', 
                     t_field = 't_anom_10D', 
                     z_field = 'z_anom_10D',
                     dir_out = '/home/data/lab_hardik/heatwaves/ERA5/analyses/EDA/movies/'):
    
    da_plt0 = ds[t_field]
    vmin = da_plt0.quantile(0)
    vmax = da_plt0.quantile(1)

    da_cont0 = ds[z_field]
    da_u0 = ds[u_field]
    da_v0 = ds[v_field]

    coarsen_pts = 8

    da_u0 = da_u0.coarsen(latitude=coarsen_pts, boundary='trim', side='right').mean()\
            .coarsen(longitude=coarsen_pts, boundary='trim', side='right').mean()
    da_u0 = da_u0.assign_coords(latitude=np.round(da_u0.latitude), longitude=np.round(da_u0.longitude))

    da_v0 = da_v0.coarsen(latitude=coarsen_pts, boundary='trim', side='right').mean()\
            .coarsen(longitude=coarsen_pts, boundary='trim', side='right').mean()
    da_v0 = da_v0.assign_coords(latitude=np.round(da_v0.latitude), longitude=np.round(da_v0.longitude))

    # %matplotlib notebook
    # %matplotlib notebook

    xmin = xmin
    xmax = xmax
    ymin = ymin
    ymax = ymax

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(projection=ccrs.PlateCarree())

    def animate(i):
        print(i)
        da_plt = da_plt0.isel(date=i)

        da_cont = da_cont0.isel(date=i)
        da_u = da_u0.isel(date=i)
        da_v = da_v0.isel(date=i)

        print(da_plt.date.data)
        plt.cla()

        if i==0:
            cont = plot_contf(da_plt, ax=ax, vmax=vmax, vmin=vmin)
            
            colormap = plt.cm.get_cmap('coolwarm') # 'plasma' or 'viridis'
#             sm = plt.cm.ScalarMappable(cmap=colormap)
#             sm.set_clim(vmin=vmin, vmax=vmax)
            
#             cbar = plt.colorbar(sm)

            cbar = plt.colorbar(cont, orientation='horizontal', shrink=0.5, aspect=10, 
                         label='s\' (J kg-1)', 
                         format='%.0f', pad=0.25, ax=ax)

            plt.setp(cbar.ax.get_xticklabels(), rotation=90)


        else:
            cont = plot_contf(da_plt, ax=ax, vmax=vmax, vmin=vmin)

        ax.add_patch(
            mpatches.Rectangle(
                xy=[68, 24], width=10, height=7,
                facecolor='none', 
                edgecolor='black', 
                linewidth=2,
                transform=ccrs.PlateCarree())
        )
            
        plot_contours(da_cont = da_cont, lev_min=cont_min, lev_max=cont_max, lev_diff = cont_spacing, ax=ax)
        ax.quiver(da_u.longitude, da_u.latitude, da_u.values, da_v.values)
        
        ax.set_xticks(np.arange(xmin, xmax+1, 10), crs=ccrs.PlateCarree())
        ax.set_xticklabels(ax.get_xticks(), rotation=90)
        ax.set_yticks(np.arange(ymin, ymax+1, 10), crs=ccrs.PlateCarree())
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_extent((xmin,xmax,ymin,ymax), crs=ccrs.PlateCarree())
        ax.coastlines()
        
        title_str = '{},\n Z\' = {}'\
        .format(da_plt.date.dt.date.values, 
                np.round(da_cont.sel(latitude=slice(31,24), longitude=slice(68,78)).mean(['latitude','longitude']).values),0)

        ax.set_title(title_str)

        print('{} plot done'.format(i))

    ani = animation.FuncAnimation(fig, animate, 
                                  frames=np.arange(0,len(ds.date.data),1), 
                                  interval=1000, repeat=False, init_func=lambda: None)

    ani.save(dir_out + 'Path_{}_{}'.format(path_desc, mon_str) + '.mp4', 
             writer=writer)
