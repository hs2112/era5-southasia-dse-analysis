

from datetime import datetime as dt, timedelta

import os, xarray as xr, numpy as np, pandas as pd, datetime, dask, random, time, netCDF4


import cartopy.crs as ccrs, matplotlib.pyplot as plt
import regionmask
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# +
import matplotlib.animation as animation

import geopandas as gpd
import shapefile as shp
import seaborn as sns
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
from shapely.geometry import Point
sns.set_style('whitegrid')
# -

Writer = animation.writers['ffmpeg']
writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)

import sys
sys.path.insert(1, '/home/data/lab_hardik/udf/')

# +
# vel_dim_map = {'u':'x', 'v':'y', 'w':'z'}
# full_wind_cols = []
# for vel in ['u','v','w']:
#     filt_cols = [s for s in df_mlr0.columns if s.startswith('{}'.format(vel))]
#     print(filt_cols)
#     col_str = '{}_AdvFlx_{}_dse'.format(vel, vel_dim_map[vel])
#     df_mlr0[col_str] = df_mlr0[filt_cols].sum(axis=1)
#     full_wind_cols.append(col_str)

#     for dsetyp in ['Anom', 'Clmt']:
#         filt_cols = [s for s in df_mlr0.columns if s.startswith('{}'.format(vel)) and s.endswith('{}'.format(dsetyp))]
#         print(filt_cols)
#         col_str = '{}_AdvFlx_{}_dse{}'.format(vel, vel_dim_map[vel], dsetyp)
#         df_mlr0[col_str] = df_mlr0[filt_cols].sum(axis=1)
#         full_wind_cols.append(col_str)
    
    
# df_mlr0.columns
# # = pd.merge(df_mlr0, df_mlr0_fullWinds, left_index=True, right_index=True)
# # [['adv_recon', 'coready_dse_del1']]
# -

def plot_contours(da_cont = xr.DataArray(), lev_min=-1500, lev_max=1500, lev_diff = 100, ax=np.ndarray(shape=(3,2)), 
                 lw = 1, col = 'grey'):
    
    num_levs = int((lev_max-lev_min)/lev_diff + 1)
    print('num_levs = ', num_levs)
    cs2 = ax.contour(
        da_cont.longitude, 
        da_cont.latitude, 
        da_cont, 
        levels=np.linspace(lev_min, lev_max, num_levs),
        colors=col,
        linewidths=lw, transform=ccrs.PlateCarree()
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



from matplotlib.axes import Axes

import pydot

import variables
from variables import *


def get_ticks(minimum, maximum, levels):
    step = np.ceil((maximum - minimum)/levels)
    return np.arange(minimum, maximum, step)



lev_min = np.nan
lev_min != lev_min


# +
def plot_composite(da_colplt, da_cont, da_u, da_v, xmin, xmax, ymin, ymax, xticks='on',
                   cont_divisions=100, cont_min=np.nan, cont_max=np.nan, quivers='on',
                   axi = Axes, col_levs=6, cbar_pad=.15, lev_min=np.nan, lev_max = np.nan, cbar_lab='xyz', title=''):

#     fig, ax = plt.subplots(nrows=1, ncols=1, 
#                         sharey=True,
#                         subplot_kw={'projection': ccrs.PlateCarree()},
#                         figsize=(16,5))

#     vmin = da_colplt.quantile(0)
#     vmax = da_colplt.quantile(1)
  
#     da_colplt.plot(ax=ax, 
#                 vmax=vmax, 
#                 cmap = 'Reds',
#                 vmin = vmin, 
#                 add_colorbar=True, 
#                 cbar_kwargs=dict(orientation='horizontal',
#                                  fraction=0.05, 
#                                  pad=0.25, 
#                                  label = 'temperature (Kelvin)'))

    cont = axi.contourf(
        da_colplt.longitude, da_colplt.latitude, da_colplt.values,
        cmap='Reds', 
        levels=get_ticks(
            np.round(da_colplt.quantile(0)/10**4)*10**4, 
            np.round(da_colplt.quantile(1)/10**4)*10**4 + 1.5*10**4, 
            col_levs
        ) if lev_min != lev_min else np.linspace(
            lev_min, 
            lev_max, 
            col_levs
        )
    )

    shrink_factor=0.7
    cbar = plt.colorbar(cont, orientation='horizontal', shrink=shrink_factor,
                 aspect=30, label=cbar_lab, format='%.1f', pad=cbar_pad)
    
    cbar.ax.tick_params(labelrotation=90)
    
    if quivers=='on':
        axi.quiver(da_u.longitude, da_u.latitude, da_u.values, da_v.values)
    
    if da_cont:
        contmin = da_cont.quantile(0) if cont_min != cont_min else cont_min
        contmax = da_cont.quantile(1) if cont_max != cont_max else cont_max
        plot_contours(da_cont = da_cont, lev_min=contmin, lev_max=contmax, lev_diff = (contmax-contmin)/cont_divisions, 
                      ax=axi)
    
    axi.add_patch(mpatches.Rectangle(xy=[68, 24], width=10, height=7,
                                    facecolor='none', 
                                    edgecolor='black', 
                                    linewidth=2, 
                                    transform=ccrs.PlateCarree()
                                   ))
    # map_df.boundary.plot(ax=ax)
    if xticks == 'on':
        axi.set_xticks(np.arange(xmin, xmax+1, 10), crs=ccrs.PlateCarree())
        axi.set_xticklabels(axi.get_xticks(), rotation=90)

    axi.set_yticks(np.arange(ymin, ymax+1, 10), crs=ccrs.PlateCarree())
    axi.set_extent([xmin,xmax,ymin,ymax], crs=ccrs.PlateCarree())
    #         plt.xticks(rotation=45, fontsize=6)
#             plt.yticks(fontsize=6)

    axi.set_title(title if title != '' else np.nan)

    axi.coastlines()
#     plt.tight_layout()
# -






def plot_contf(da_plt=xr.DataArray(), ax=np.ndarray(shape=(3,2))):
    return ax.contourf(
    da_plt.longitude, 
    da_plt.latitude, 
    da_plt.values, 
    levels=np.linspace(-50, 50, 101), 
    cmap='RdYlGn', 
    vmax= 15,
    vmin=-15
)


# +
def path_dates_movie(ds, path_desc, mon_str, 
                     xmin=0, xmax=120, ymin=5, ymax=70,
                     cont_min=-1500, cont_max=1500, cont_spacing=200, 
                     dir_out = '/home/data/lab_hardik/data/ERA5/climatology/'):
    
    da_plt0 = ds['v']
    da_cont0 = ds['z_anom_10D']

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

        print(da_plt.date.data)
        plt.cla()

        if i==0:
            cont = plot_contf(da_plt, ax=ax)

            cbar = plt.colorbar(cont, orientation='horizontal', shrink=0.5, aspect=30, 
                         label='v ($\mathregular{m}$ $\mathregular{s^{-1}}$)', 
                         format='%.0f', pad=cbar_pad, ax=ax)

            plt.setp(cbar.ax.get_xticklabels(), rotation=90)

#             plot_contours(da_cont = da_cont0.isel(date=i), 
#                           lev_min=-1000, lev_max=1000, lev_diff = cont_spacing, ax=ax)

#             ax.add_patch(mpatches.Rectangle(xy=[68, 24], width=10, height=7,
#                                             facecolor='none', 
#                                             edgecolor='black', 
#                                             linewidth=2,
#                                         transform=ccrs.PlateCarree()))

        else:
            cont = plot_contf(da_plt, ax=ax)

        ax.add_patch(
            mpatches.Rectangle(
                xy=[68, 24], width=10, height=7,
                facecolor='none', 
                edgecolor='black', 
                linewidth=2,
                transform=ccrs.PlateCarree())
        )
            
        plot_contours(da_cont = da_cont, lev_min=cont_min, lev_max=cont_max, lev_diff = cont_spacing, ax=ax)
        ax.set_xticks(np.arange(xmin, xmax+1, 10), crs=ccrs.PlateCarree())
        ax.set_xticklabels(ax.get_xticks(), rotation=90)
        ax.set_yticks(np.arange(ymin, ymax+1, 10), crs=ccrs.PlateCarree())
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_extent((xmin,xmax,ymin,ymax), crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_title(da_plt.date.dt.date.values)
        print('{} plot done'.format(i))

    ani = animation.FuncAnimation(fig, animate, 
                                  frames=np.arange(0,len(ds.date.data),1), 
                                  interval=1000, repeat=False, init_func=lambda: None)

    ani.save(dir_out + 'Path_{}_{}'.format(path_desc, mon_str) + '.mp4', 
             writer=writer)



# +
# def assign_dates_climatology_2(ds, year=1904):
# #     date_arr = [
# #         dt.strptime(dt.strptime(ds.strftime.data[i], '%d-%b').strftime('{}-%m-%d'.format(year)), '%Y-%m-%d')
# #         for i in range(len(ds.strftime))
# #     ]
#     date_arr = [
#         dt.strptime(dt.strptime('1904-' + ds['strftime'].data[i], '%Y-%d-%b').strftime('1904-%m-%d'), '%Y-%m-%d')
#         for i in range(len(ds['strftime']))
#     ]  

#     return ds.assign_coords({'date': ('strftime', date_arr)}).sortby('date')

# -




# +
# def assign_dates_climatology(ds, to_replace='strftime'):
    
#     date_arr = [
#         dt.strptime(dt.strptime('1904-' + ds[to_replace].data[i], '%Y-%d-%b').strftime('1904-%m-%d'), '%Y-%m-%d')
#         for i in range(len(ds[to_replace]))
#     ]  
    
#     return ds.assign_coords({'date': (to_replace, date_arr)}).sortby('date')
# -



# +
def plot_vert_struc(cbar_lab='d'r'$\theta/dz$ [K $\mathregular{km^{-1}}$]', 
                     da_plt = xr.DataArray(),
                     sp_plt_series = xr.DataArray(), # surface pressure series
                     t2m_series = xr.DataArray(), # t2m anom series
                     plot_name = '',
                     cmap='YlGn', 
                    ):
    # Start with a square Figure
    # %matplotlib inline
    fig = plt.figure(figsize=(16,6))
    plt.tight_layout()
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions
    # Also adjust the subplot parameters for a square plot
    gs = fig.add_gridspec(2, 2,  height_ratios=(1, 4), width_ratios=(4, 1),
                          left=0.1, right=0.9, 
                          bottom=0.1, top=0.9,
                          hspace=0.1, wspace=0.05)
    # Create the Axes
    ax = fig.add_subplot(gs[1,0])
    ax_t2m = fig.add_subplot(gs[0,0], sharex=ax)

    da_plt\
    .transpose()\
    .plot(ax=ax, add_colorbar=True, cmap=cmap,
          cbar_kwargs=dict(orientation='horizontal', label=cbar_lab,
                           fraction=0.05, pad=0.25),
     )

    ax.invert_yaxis()
    ax.set_ylim(1000,300)
    (sp_plt_series/100)\
    .plot(ax=ax, color='black')

    ax.set_ylabel('Pressure (hPa)')
    ax.set_xlabel('')

    t2m_min = t2m_series.quantile(0)
    t2m_max = t2m_series.quantile(1)
    (t2m_series)\
    .plot(ax=ax_t2m)
    ax_t2m.set_ylim([t2m_min, t2m_max])
    ax_t2m.get_xaxis().set_visible(False)
    ax_t2m.axhline(0, color='black',linewidth=1)
    ax_t2m.axhline(2, color='black',linewidth=1)
    ax_t2m.set_ylabel('T2m \n ($^\circ$C)')

    plt.tight_layout()
    save_str = '/home/data/lab_hardik/data/ERA5/climatology/plots/' + plot_name
    plt.savefig(save_str)
    
    
