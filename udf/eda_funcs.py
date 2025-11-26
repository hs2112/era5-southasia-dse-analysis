# +
import os, xarray as xr, pandas as pd
import cartopy.crs as ccrs, matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.patches as mpatches
import gc
from matplotlib.gridspec import GridSpec


# -

def plot_hw_v_lines(axis, plot_dates = list(), plot_dates_1 = list(), plot_dates_2 = list(), lw=1): 
    for dat in plot_dates:
        axis.axvline(dat, color='black', linewidth=lw, linestyle='--' if dat in plot_dates_1 else '-' if dat in plot_dates_2 else np.nan)



# +
def vert_struct_plot(
    cbar_lab='', # d'r'$\theta/dz$ [K $\mathregular{km^{-1}}$] 
    da_plt = xr.DataArray(),
    sp_plt_series = xr.DataArray(), # surface pressure series
    t2m_anom_series = xr.DataArray(), # t2m anom series
    save_str = 'abc',
    plot_dates = list(),
    plot_dates_1 = list(),
    plot_dates_2 = list(),
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
    .sel(isobaricInhPa=slice(300,1000))\
    .mean(['latitude','longitude'])\
    .transpose()\
    .plot(ax=ax, add_colorbar=True, cmap=cmap,
          cbar_kwargs=dict(orientation='horizontal', label=cbar_lab,
                           fraction=0.05, pad= 0.05 if 't' in da_plt.name else 0.25),
     )

    ax.invert_yaxis()
    ax.set_ylim(1000,300)
    ax.axhline(950, c='dimgray')

    if 't' in da_plt.name: 
        plt.gca().set_xticks([])
    
#     (sp_plt_series/100)\
#     .plot(ax=ax, color='black')

    ax.set_ylabel('Pressure\n (hPa)')
    ax.set_xlabel('')

    (t2m_anom_series)\
    .plot(ax=ax_t2m)
    ax_t2m.set_ylim([-5,5])
    ax_t2m.get_xaxis().set_visible(False)
    ax_t2m.axhline(0, color='black',linewidth=1)
    ax_t2m.axhline(1, color='black',linewidth=1)
    ax_t2m.set_ylabel('T2m\' \n ($^\circ$C)')

    for axis in [ax, ax_t2m]:
        plot_hw_v_lines(axis, plot_dates, plot_dates_1, plot_dates_2, lw=1)

    plt.tight_layout()
    plt.savefig(save_str)
# -



def ave_cor_sctrplt(var = 'shfanom_mon_anom', 
                    dv = 't2m_anom_del_mon_anom', 
#                     cor_var = 'ave2D', # Lag1 
                    col_var = None,
                   mon_filt_lst = [],
                    add_0_0 = True,
                   df = pd.DataFrame(), 
                   xy_labs = []):
    
    df_cor = df.copy()
    
    if len(mon_filt_lst) >0:
        df_cor = df_cor.loc[df_cor.month.isin(mon_filt_lst)]
        
    iv = var

    fig, ax = plt.subplots(figsize=(7,4))
    
    if col_var in [None, '']:
        df_cor.plot.scatter(x=iv, y=dv, s=1, ax=ax)
    else: 
        df_cor.plot.scatter(x=iv, y=dv, s=1, c=col_var,  colormap='coolwarm', ax=ax)
    if add_0_0 == True:
        ax.axhline(0)
        ax.axvline(0)
    
    if len(xy_labs) > 0:
        ax.set_xlabel(xy_labs[0])
        ax.set_ylabel(xy_labs[1])
    print(df_cor[[iv,dv]].corr())


def create_classes(y_var = 't2m_anom_del', num_classes = 3, df=pd.DataFrame()):
#     global df_raw
    
    clss_bound_abs = int((num_classes-1)/2)
    y_classed = y_var + '_clss'

    labs = [n for n in range(-1*clss_bound_abs, clss_bound_abs+1)]
    q_arr = [np.round(n*1/num_classes,2) for n in range(num_classes+1)]
    print(labs, q_arr)

    df_tmp = df.assign(
        **{
            y_classed: pd.qcut(df[y_var], q=q_arr, labels=labs,
    #                                                       right=False
               )
        }
    )

    print(y_classed, 'created')
    print(df_tmp['{}_clss'.format(y_var)].value_counts().sort_index())

    lab_desc_map = \
    ['({},{}]'.format(np.round(df_tmp[y_var].quantile(q_arr[s]),1), 
                             np.round(df_tmp[y_var].quantile(q_arr[s+1]),1)
                            ) 
     for s in range(len(q_arr)-1)
    ]
    
    globals()[f'{y_var}_class_map'] =  dict(zip(labs, lab_desc_map))
    dict_name = '{}_class_map'.format(y_var)

    print(dict_name, 'created')
    print(globals()[f'{y_var}_class_map'])

    return df_tmp
