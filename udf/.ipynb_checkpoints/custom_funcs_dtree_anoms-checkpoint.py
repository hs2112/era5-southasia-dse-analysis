# +
import os, xarray as xr, numpy as np, pandas as pd, dask, random, netCDF4
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from datetime import datetime as dt, timedelta
import datetime, time
import pylab

import matplotlib.patches as mpatches
import gc
from matplotlib.gridspec import GridSpec
dask.config.set({"array.slicing.split_large_chunks": True})

params = {
#     'legend.fontsize': 10,
#           'legend.title_fontsize': 10,
#           'figure.figsize': (15, 5),
         'axes.labelsize': 15, # this controls labelsize of both x and y axis of main plot as well as colorbar  
         'axes.titlesize':25, # pot title size
         'xtick.labelsize':15,
         'ytick.labelsize':15 # this controls yticks labelsize of both main plot and colorbar 
}


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

def tot_mass_bw_levs(M = xr.Dataset(), p_low = 400, p_high = 500):
    return M.sel(
        isobaricInhPa = slice(p_low, p_high), 
          latitude=slice(24,30.75), longitude=slice(68,77.75)
        ).sum(['isobaricInhPa','latitude','longitude'])


def vert_MassAve_qty(M = xr.Dataset(), qty = xr.DataArray(), p_low=600, p_high=900): # qty = -1*(u_gradx_dse + v_grady_dse)*86400
    return (M*qty).sel(
        isobaricInhPa = slice(p_low, p_high), latitude=slice(24,30.75), longitude=slice(68,77.75)
    ).sum(['isobaricInhPa','latitude','longitude'])/(tot_mass_bw_levs(M, p_low, p_high))




# +
def plot_hw_v_lines(plot_dates, axis, lw=2):
    for dat in plot_dates:
        axis.axvline(
        dat, 
        color='black', linewidth=lw, linestyle='--')


def plot_3d_t2m(ds = xr.DataArray(), bin_plevs = True, 
                bin_arr = range(300,1100,100), 
                bin_agg = 'sum', 
                legend_lab = 'W', 
                t2m_variant='vanilla', 
                year=2016,
                sp_plt_series = xr.DataArray(),
                t2m_anom_series = xr.DataArray(),
                plot_dates = list()
               ):

    
    # Start with a square Figure
    fig = plt.figure(figsize=(16,5))
    # plt.rcdefaults()
    pylab.rcParams.update(params)
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

    # ((ds_bflx['net_heat_flux_z'])/ds_bflx['p_diff_pscl'])\

    if bin_plevs==True:
        if bin_agg == 'sum':
            ds.sel(date = ds.date.dt.year==year)\
            .sel(isobaricInhPa=slice(300,None))\
            .groupby_bins('isobaricInhPa', bin_arr, right=False).sum()\
            .plot(ax=ax, add_colorbar=True, 
#                   vmin=-3*10**14,
#                   vmax=3*10**14,
#                   cmap='coolwarm',
                  cbar_kwargs=dict(orientation='horizontal', 
                                   label=legend_lab, 
                                   fraction=0.05, pad=0.3))
        elif bin_agg == 'mean':
            ds.sel(date = ds.date.dt.year==year)\
            .sel(isobaricInhPa=slice(300,None))\
            .groupby_bins('isobaricInhPa', bin_arr, right=False).mean()\
            .plot(ax=ax, add_colorbar=True, 
                  cbar_kwargs=dict(orientation='horizontal', 
                                   label=legend_lab, 
                                   fraction=0.05, pad=0.3))
            
    else:
        ds.sel(date = ds.date.dt.year==year)\
        .sel(isobaricInhPa=slice(300,None))\
        .plot(ax=ax, add_colorbar=True, 
              cbar_kwargs=dict(orientation='horizontal', 
                               label=legend_lab, 
                               fraction=0.05, pad=0.3))

    ax.invert_yaxis()
    ax.set_ylim(1000,300)
    (sp_plt_series/100)\
    .plot(ax=ax, color='black')

    ax.set_ylabel('Pressure\n (hPa)')
    ax.set_xlabel('')

    t2m_int = t2m_anom_series.sel(date = t2m_anom_series.date.dt.year == year)
    if t2m_variant=='vanilla':
        ds_t2m_plt = t2m_int
    if t2m_variant=='del':
        ds_t2m_plt = t2m_int - t2m_int.shift(date=1) 
        
    (ds_t2m_plt)\
    .plot(ax=ax_t2m, color='black')
    ax_t2m.set_ylim([-5,5])
    ax_t2m.get_xaxis().set_visible(False)
    ax_t2m.axhline(0, color='black',linewidth=1)
    ax_t2m.axhline(2, color='black',linewidth=1)
    ax_t2m.set_ylabel('t2m\' ($^\circ$C)' if t2m_variant=='vanilla' else 'del t2m\' ($^\circ$C)' if t2m_variant=='del' else np.nan)

    for axis in [ax,ax_t2m]:
        plot_hw_v_lines(plot_dates, axis, lw=1)
    # plt.savefig("/home/data/lab_hardik/analysis/DSE/curvedArea_dse_fluxes_z_2016.png", bbox_inches="tight")    
# -



def calc_cor(flxVar = xr.DataArray(), #  point qty * timestep
             dlyVar = xr.DataArray(), #  point qty
             dlyVar_shifted = xr.DataArray(), #  point qty
             M = xr.DataArray(),
             mon_lst = [3,4,5],
             pres_range = [300,975],
             plots = 'on', 
             xlimits = np.nan, ylimits = np.nan,
             figsize=(10,6)
            ):
    global flxVar_net_colSum, flxVar_lag1d_colSum, dseVar_diffColSum, \
    coready_dse_change, coready_DSEflx_lag1d, coready_DSEflx
    
    p_low = pres_range[0]
    p_high = pres_range[1]
    
    #  dlyVar is pt qty
    with ((dlyVar - dlyVar_shifted)) as dseVar_diff:
        if 6 not in mon_lst:
            dseVar_diff = dseVar_diff.sel(date = dseVar_diff.date.dt.month.isin(mon_lst)).compute() # 
        else: 
            dseVar_diff = dseVar_diff.sel(date = 
                                          (dseVar_diff.date.dt.month.isin([s for s in mon_lst if s != 6])) | 
                                         ((dseVar_diff.date.dt.day <= 15) & (dseVar_diff.date.dt.month == 6))
                                         ).compute() # 

        print(len(dseVar_diff))
    
        coready_dse_change = vert_MassAve_qty(
                M = M, 
                qty = dseVar_diff.where(np.abs(dseVar_diff) <= np.abs(dseVar_diff).quantile(.98), drop=True).compute(),
                p_low=p_low, p_high=p_high
        )
        
        print(len(coready_dse_change))
    
    # Flux variables
    coready_DSEflx = vert_MassAve_qty(
        M = M, 
        qty = flxVar.sel(
            date = coready_dse_change.date.dt.date).compute(), 
        p_low=p_low, p_high=p_high
    )

    coready_DSEflx_lag1d = coready_DSEflx.shift(date=1)
    coready_DSEflx_lag1d.loc[dict(date = (coready_DSEflx_lag1d.date.dt.day==1) & 
                                      (coready_DSEflx_lag1d.date.dt.month==3))
                                ] = np.nan
    coready_DSEflx_lag1d = coready_DSEflx_lag1d.dropna('date')
        
    ds_mst = xr.merge(
        [
            coready_dse_change.rename('coready_dse_change'), 
            coready_DSEflx.rename('coready_DSEflx'), 
            coready_DSEflx_lag1d.rename('coready_DSEflx_lag1d')
        ])\
    .dropna('date')
    
    flx_ave2d = (ds_mst['coready_DSEflx_lag1d']+ds_mst['coready_DSEflx'])/2
    
    cor_lag1d = np.round(xr.corr(ds_mst['coready_dse_change'], ds_mst['coready_DSEflx_lag1d']).data, 2)
    cor_sameday = np.round(xr.corr(ds_mst['coready_dse_change'], ds_mst['coready_DSEflx']).data,2)
    cor_ave = np.round(xr.corr(ds_mst['coready_dse_change'], flx_ave2d).data,2)
    

    if plots=='on':
        fig,ax = plt.subplots(figsize=figsize)
        ax.scatter(flx_ave2d, ds_mst['coready_dse_change'], s=1)
        
        if xlimits == xlimits:
            ax.set_xlim(xlimits[0], xlimits[1])
        if ylimits == ylimits:
            ax.set_ylim(ylimits[0], ylimits[1])
        
        ax.axhline(ds_mst['coready_dse_change'].mean())
        ax.axvline(flx_ave2d.mean())

        ax.set_ylabel('Daily change in \nDSE (J $\mathregular{kg^{-1}}$)')
        ax.set_xlabel('Net Advective Flux \nDSE (J $\mathregular{kg^{-1}}$)')

        
    print('cor w/ lag1d sum', cor_lag1d)
    print('cor w/ same day sum', cor_sameday)
    print('cor w/ both days\' avg sum', cor_ave)
    
    print('flxVar_net_colSum, flxVar_lag1d_colSum, dseVar_diffColSum, coready_dse_change, coready_DSEflx_lag1d, coready_DSEflx created')
    return ds_mst['coready_DSEflx'], ds_mst['coready_DSEflx_lag1d'], ds_mst['coready_dse_change']



# +
import pydot, re
from graphviz import Source
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
# from dtreeviz.trees import dtreeviz
import dtreeviz

def make_tree(df_iv = pd.DataFrame(), 
              df_comb = pd.DataFrame(), 
              y_var = 'coready_dse_diff_clss',
              
              depth_param=4, month='Mar_Apr', vert_lev_str_append='', num_classes=3, y_str='DSEdiff',
              
              dir_mods = '', 
              entropy_decrease_thresh=0.03, fontsize=12, 
              col_param=True, k = np.int(), rotation=True,
              labels = list(), 
              
              testing ='on',
              test_size=0.1,
              min_samples = 2,
             ):
    
    global mod, expected_y, predicted_y, df_mod_iv, iv_list, viz
    
    depth = depth_param
    
    df_mod_iv = df_iv
    iv_list = df_mod_iv.columns
    print(iv_list)
    
    X = df_mod_iv.values
    np.random.seed(k)

    y = df_comb[y_var].values

    if testing=='on':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify = y, random_state=k)
    else: 
        X_train, y_train = X, y
        X_test, y_test = X, y

    mod = DecisionTreeClassifier(
        random_state=k, 
        criterion='entropy',
        max_depth=depth_param,
        min_samples_leaf= min_samples if min_samples == min_samples else 2, 
    #     min_samples_split=20, 
        min_impurity_decrease= entropy_decrease_thresh
    )

    mod.fit(X_train, y_train)
    out_str = "{}_{}_{}_{}Cat_depth{}.dot".format(vert_lev_str_append,month,y_str,num_classes,depth_param)

    # Plot the decision tree graph
    export_graphviz(
        mod,
        out_file=out_str,
        feature_names=df_mod_iv.columns,
        class_names=labels,
        rotate=rotation,
        rounded=True,
        filled=col_param, # leaf node colors 
        impurity=False,
    #     impurity=True,
        label='all',
        fontname='verdana'
     )
#     pydot_graph.set_size('"10,8!"')

    expected_y  = y_test
    predicted_y = mod.predict(X_test)

    print('mod accuracy score with criterion entropy: {0:0.4f}'.format(accuracy_score(expected_y, predicted_y)))
    print(classification_report(expected_y, predicted_y))
    print(confusion_matrix(expected_y, predicted_y))
    
    df_varimp = pd.DataFrame({'var_name':iv_list, 'imp': mod.feature_importances_}).sort_values('imp', ascending=False)

    print(df_varimp)

    fig,ax = plt.subplots(figsize=(15, 10)) # Resize figure

    tree.plot_tree(mod, 
                   filled=col_param, 
                   rounded=True,
                   impurity=False,
                   class_names=labels,
                   feature_names=df_iv.columns,
                   ax=ax, 
                   fontsize=fontsize
    )

#     viz = dtreeviz.model(mod, df_mod_iv,df_comb[y_var],
#                target_name=y_var,
#                feature_names=iv_list, 
#                class_names=labels)
#     viz.view() 
    plt.show()
#     print(expected_y, predicted_y)
    
    
    ### for modified file for easy reading, uncomment below lines, set impurity=False above, 
    # and below that, open(mod_out_str) as f, Source(dot_graph).render(mod_out_str.replace('.dot', '')

#     if accuracy_score(expected_y, predicted_y) > 0.3:
    PATH = dir_mods + out_str if dir_mods != '' else out_str
    f = pydot.graph_from_dot_file(PATH)[0].to_string()
    f = re.sub('(\\\\nsamples = [0-9]+)', '', f) # (\\\\nvalue = \[[0-9]+, [0-9]+, [0-9]+\])
    f = re.sub('(samples = [0-9]+)\\\\n', '', f) # (\\\\nvalue = \[[0-9]+, [0-9]+, [0-9]+\])

#     f.write_png('original_tree.png')
#     f.set_size('"5,5!"')
#     f.write_png('resized_tree.png')

    mod_out_str = '{}_modified.dot'.format(out_str.replace('.dot',''))

    with open(mod_out_str, 'w') as file:
        file.write(f)

    with open(mod_out_str) as f:
        dot_graph = f.read()

    Source(dot_graph)
    Source(dot_graph).render(mod_out_str.replace('.dot', ''),format='svg', view=False)
    Source(dot_graph).render(mod_out_str.replace('.dot', ''),format='png', view=False)
    
#     print(mod_out_str.replace('.dot', ''))
    print(os.getcwd())
    k=k+1
#     plt.savefig("test.svg", format="svg")
    return df_varimp


# +
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

def results_summary_to_dataframe(results, print_res=True):
#     global results_df, results_df2
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int().iloc[:,0]
    conf_higher = results.conf_int().iloc[:,1]

    results_df = pd.DataFrame({
#         "variable":pvals.index,
        "pvals":pvals,
        "coeff":coeff,
        "conf_lower":conf_lower,
        "conf_higher":conf_higher
                                })\
    .sort_values(['coeff'], ascending=False)
    
    print(results_df.columns)
    results_df2 = results_df
#     .loc[(results_df.pvals<0.05) & (np.abs(results_df.coeff) > 0.05)]

    if print_res==True:
        print(results_df2)
    
    return results_df, results_df2

def standardize_df(X = pd.DataFrame(), y = pd.Series(dtype=np.float64)):
    X_iv = pd.DataFrame(StandardScaler().fit_transform(X), columns = X.columns, index = X.index)
    if len(y) > 0:
        y_dv = pd.DataFrame(StandardScaler().fit_transform(np.array(y).reshape(-1,1)), columns = [y.name], index = y.index)
        return X_iv, y_dv
    else:
        return X_iv

def mlr(X = pd.DataFrame(), y = pd.Series(dtype=np.float64), test_size = 0.1, figsize=(16,5), print_res=True):
    global X_iv, x_train, x_test, y_train, y_test, regr, df_mod, results_df, results_df2
    
    X_iv, y = standardize_df(X, y)
#     pd.DataFrame(StandardScaler().fit_transform(X), columns = X.columns)
#     y = pd.DataFrame(StandardScaler().fit_transform(np.array(y).reshape(-1,1)), columns = [y.name])

    x_train, x_test, y_train, y_test = train_test_split(X_iv, y, test_size = test_size)

    #add constant to predictor variables
    x_train = sm.add_constant(x_train)

    #fit linear regression model
    model = sm.OLS(y_train, x_train).fit()
    
    #view model summary
    r_sq = np.round(model.rsquared_adj, 2)
    modsum = model.summary()
    
    if print_res==True:
        print(modsum.tables[0])
    
    results_df, results_df2 = results_summary_to_dataframe(model, print_res=print_res)
    
#     if r_sq > 0.5:
        # fig,ax = plt.subplots()
    plt.figure(figsize = figsize)
    plt.plot(results_df2.index, results_df2.coeff)
#         plt.axhline(0.2)
    # ax.set_xlabel(ax.get_xlabel(),rotation='vertical')
    _ = plt.xticks(results_df2.index,rotation=90) # , fontsize=14

    return results_df, results_df2


# +
def create_classes(df = pd.DataFrame(), y_var = 'del_t2m_anom', num_classes = 3):
    global dict_name
    
#     clss_bound_abs = int((num_classes-1)/2)
    y_classed = y_var + '_clss'

#     labs = [n for n in range(-1*clss_bound_abs, clss_bound_abs+1)]
    labs = [n for n in range(num_classes)]

    q_arr = [np.round(n*1/num_classes,2) for n in range(num_classes+1)]
    print(labs, q_arr)

    df2 = df.assign(
        **{
            y_classed: pd.qcut(df[y_var], q=q_arr, labels=labs,
    #                                                       right=False
               )
        }
    )

    print(y_classed, 'created')
    print(df2['{}_clss'.format(y_var)].value_counts().sort_index())

    lab_desc_map = \
    ['({},{}]'.format(np.round(df2['{}'.format(y_var)].quantile(q_arr[s]),1), 
                             np.round(df2['{}'.format(y_var)].quantile(q_arr[s+1]),1)
                            ) 
     for s in range(len(q_arr)-1)
    ]
    
    globals()[f'{y_var}_class_map'] =  dict(zip(labs, lab_desc_map))
    dict_name = '{}_class_map'.format(y_var)

    print(dict_name, 'created')
    print(globals()[f'{y_var}_class_map'])

    return df2, globals()[f'{y_var}_class_map']
# -



import cartopy.crs as ccrs, matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def plot_contours(da_cont = xr.DataArray(), lev_min=-1500, lev_max=1500, lev_diff = 100, ax=np.ndarray(shape=(3,2))):
    
    num_levs = int((lev_max-lev_min)/lev_diff + 1)
    print('num_levs = ', num_levs)
    cs2 = ax.contour(
        da_cont.longitude, 
        da_cont.latitude, 
        da_cont, 
        levels=np.linspace(lev_min, lev_max, num_levs),
        colors='grey',
        linewidths=1, linestyles='solid', transform=ccrs.PlateCarree()
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


def plot_composite(da_plt, da_cont, xmin, xmax, ymin, ymax, figsize=(16,5)):

    fig, ax = plt.subplots(nrows=1, ncols=1, 
                        sharey=True,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=figsize)

    da_plt.plot(ax=ax, 
                vmax=7, 
                cmap = 'RdYlGn', 
                vmin = -7, 
                add_colorbar=True, 
                cbar_kwargs=dict(orientation='horizontal',
                                 fraction=0.05, 
                                 pad=0.25, 
                                 label = 'v component of wind (m $\mathregular{s^{-1}}$)'))


    plot_contours(da_cont = da_cont, lev_min=-1500, lev_max=1500, lev_diff = 100, ax=ax)

    ax.add_patch(mpatches.Rectangle(xy=[68, 24], width=10, height=7,
                                    facecolor='none', 
                                    edgecolor='black', 
                                    linewidth=2, 
                                    transform=ccrs.PlateCarree()
                                   ))
    # map_df.boundary.plot(ax=ax)
    ax.set_extent([xmin,xmax,ymin,ymax], crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(xmin, xmax+1, 10), crs=ccrs.PlateCarree())
    ax.set_xticklabels(ax.get_xticks(), rotation=90)

    ax.set_yticks(np.arange(ymin, ymax+1, 10), crs=ccrs.PlateCarree())
    #         plt.xticks(rotation=45, fontsize=6)
    #         plt.yticks(fontsize=6)

    #     ax.set_title('T2m anomalies > $2^\circ$C' if i==0 else 'T2m anomalies < $0^\circ$C' if i==1 else np.nan)

    ax.coastlines()
    plt.tight_layout()
    # plt.savefig('/home/data/lab_hardik/analysis/VertStruct/prmnsn_v_comps_{}_{}hPa.png'.format(year, press))

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
                     dir_out = '/home/data/lab_hardik/HW/'):
    
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
                         format='%.0f', pad=0.15, ax=ax)

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


