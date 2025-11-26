

# +
### Extra for generating +-5D of given date, but current strategy is to plot all mar, apr dates
import pandas as pd

def generate_date_range(input_date = str()):
    td = pd.Timedelta(1, "d")
    num_days = 5  # Number of days before and after the input date
    date_range = pd.date_range(
        pd.to_datetime(input_date) - 5*td, periods= 2*num_days+1)
    
    return date_range.tolist() # .strftime("%Y-%m-%d")

# Example usage:
input_date = "2004-03-19" # input_dates[1] #  # Replace with your desired input date (in "YYYY-MM-DD" format)
result_dates = generate_date_range(input_date)
# for date_str in result_dates:
#     print(date_str)
# -
2385*2

2370*2

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
dask.config.set({"array.slicing.split_large_chunks": False})

params = {
#     'legend.fontsize': 10,
#           'legend.title_fontsize': 10,
#           'figure.figsize': (15, 5),
         'axes.labelsize': 15, # this controls labelsize of both x and y axis of main plot as well as colorbar  
         'axes.titlesize':25, # pot title size
         'xtick.labelsize':15,
         'ytick.labelsize':15 # this controls yticks labelsize of both main plot and colorbar 
}
# -



# +
month_map = {
    3:'March',4:'April',5:'May',6:'June'
}

t2m_lag1_lab_map = {
    
    0:'negative',
    1:'neutral',
    2:'positive',    
}

# -

from functools import reduce


def merge_multiple_dfs(data_frames = list()):
    df_merged = reduce(lambda  left,right: pd.merge(left,right,left_index=True, right_index=True,
                                            how='inner'), data_frames)
    return df_merged


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
def plot_hw_v_lines(plot_dates, axis, lw=2):
    for dat in plot_dates:
        axis.axvline(
        dat, 
        color='black', linewidth=lw, linestyle='--')

        
        
def plot_3d_t2m(
    ds = xr.DataArray(), bin_plevs = True, 
    bin_arr = range(300,1100,100), 
    bin_agg = 'sum', 
    legend_lab = 'W', 
    colmap = 'coolwarm_r',
    t2m_variant='vanilla', 
    year=2016,
    sp_plt_series = xr.DataArray(),
    t2m_anom_series = xr.DataArray(),
    plot_del_t2m = False,
    plot_dates = list(),
    plot_dates_1 = list(),
    plot_dates_2 = list(),
):

    
    # Start with a square Figure
    fig = plt.figure(figsize=(16,6))
    # plt.rcdefaults()
#     pylab.rcParams.update(params)
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
        .plot(ax=ax, add_colorbar=True, cmap= colmap,
              cbar_kwargs=dict(orientation='horizontal', 
                               label=legend_lab, 
                               fraction=0.05, pad=0.25))

    ax.invert_yaxis()
    ax.set_ylim(1000,300)
    (sp_plt_series/100)\
    .plot(ax=ax, color='black')

    ax.set_ylabel('Pressure\n (hPa)')
    ax.set_xlabel('')
#     plt.gca().set_xticks([])
    
    t2m_int = t2m_anom_series.sel(date = t2m_anom_series.date.dt.year == year)
    if t2m_variant=='vanilla':
        ds_t2m_plt = t2m_int
    if t2m_variant=='del':
        ds_t2m_plt = t2m_int - t2m_int.shift(date=1) 
        
    (ds_t2m_plt)\
    .plot(ax=ax_t2m, color='black')
    if plot_del_t2m == True:
        ds_del_t2m = t2m_int - t2m_int.shift(date=1) 
        (ds_del_t2m).plot(ax=ax_t2m, color='blue')
    
    ax_t2m.set_ylim([-5,5])
    ax_t2m.get_xaxis().set_visible(False)
#     ax_t2m.axhline(0, color='black',linewidth=1)
    ax_t2m.axhline(1, color='black',linewidth=1)
    ax_t2m.axhline(-1, color='black',linewidth=1)
    
    ax_t2m.set_ylabel('t2m\' ($^\circ$C)' if t2m_variant=='vanilla' else 'del t2m\' ($^\circ$C)' if t2m_variant=='del' else np.nan)
    
    for axis in [ax,ax_t2m]:
        plot_hw_v_lines(plot_dates, plot_dates_1, plot_dates_2, axis, lw=1)
    # plt.savefig("/home/data/lab_hardik/analysis/DSE/curvedArea_dse_fluxes_z_2016.png", bbox_inches="tight")


# -



def tot_mass_bw_levs(M = xr.Dataset(), p_low = 400, p_high = 500, filt_ni = True, ):
    if filt_ni == True:
        return M.sel(
            isobaricInhPa = slice(p_low, p_high), 
              latitude=slice(24,30.75), longitude=slice(68,77.75)
            ).sum(['isobaricInhPa','latitude','longitude'])
    else:
        return M.sel(
            isobaricInhPa = slice(p_low, p_high)
            ).sum(['isobaricInhPa'])


# mass weighted average of point qantity
def vert_MassAve_qty(M = xr.Dataset(), qty = xr.DataArray(), p_low=600, p_high=900, sp_ave = True, filt_ni = True): 
    # qty = -1*(u_gradx_dse + v_grady_dse)*86400
    if sp_ave == True:
        if filt_ni == True:
            return (M*qty).sel(
                isobaricInhPa = slice(p_low, p_high), latitude=slice(24,30.75), longitude=slice(68,77.75)
            ).sum(['isobaricInhPa','latitude','longitude'])/(tot_mass_bw_levs(M, p_low, p_high))
    if (sp_ave == False) and (filt_ni == False):
        return (M*qty).sel(
            isobaricInhPa = slice(p_low, p_high)
        ).sum(['isobaricInhPa'])/(tot_mass_bw_levs(M, p_low, p_high, filt_ni = False))
    if (sp_ave == True) and (filt_ni == False):
        return ((M*qty).sel(
            isobaricInhPa = slice(p_low, p_high)
        ).sum(['isobaricInhPa'])/(tot_mass_bw_levs(M, p_low, p_high, filt_ni = False))).mean(['latitude','longitude'])



def calc_cor(flxVar = xr.DataArray(), #  point qty * timestep
             dlyVar = xr.DataArray(), #  point qty
             dlyVar_shifted = xr.DataArray(), #  point qty
             M = xr.DataArray(),
             mon_lst = [3,4,5],
             pres_range = [300,975],
             plots = 'on', 
             xlimits = np.nan, ylimits = np.nan, trim_quant = 1, jun_dmax = 15,
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
            dseVar_diff = dseVar_diff.sel(
                date = 
                (
                    dseVar_diff.date.dt.month.isin([s for s in mon_lst if s != 6])) | 
                (
                    (dseVar_diff.date.dt.day <= jun_dmax) & (dseVar_diff.date.dt.month == 6)
                )
            ).compute()

        print(len(dseVar_diff))

        coready_dse_change = vert_MassAve_qty(
                M = M, 
                qty = dseVar_diff.where(np.abs(dseVar_diff) <= np.abs(dseVar_diff).quantile(trim_quant), drop=True).compute(),
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
    return ds_mst['coready_DSEflx'], ds_mst['coready_DSEflx_lag1d'], ds_mst['coready_dse_change'], cor_sameday, cor_lag1d, cor_ave





# +

def df_update(i):
    global df
    df = pd.DataFrame({'A':[0,i]}).copy()
    return df


# +

# function to convert to superscript 
def get_super(x): 
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s)) 
    return x.translate(res)


# +

# function to convert to subscript 
def get_sub(x): 
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s)) 
    return x.translate(res) 


# -

def round_tree_values(tree):
    for i in range(tree.node_count):
        tree.value[i] = np.round(tree.value[i], 0)



# +
import pydot, re
from graphviz import Source
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import dtreeviz

# del mod, expected_y, predicted_y, df_mod_iv, iv_list, viz, df_vi, labels
# gc.collect()
global mod, expected_y, predicted_y, df_mod_iv, iv_list, viz, df_vi, x_train, y_train

def make_tree(df_iv = pd.DataFrame(), 
              df_comb = pd.DataFrame(), 
              y_var = 'coready_dse_diff_clss',
              month='Mar_Apr', 
              y_str='DSEdiff',

              vert_lev_str_append='', 
              depth_param=4, 
              num_classes=3, 
              
              dir_mods = '', 
              fontsize=12, 
              col_param=True, 
              k = int, 
              rotation=True,
              labels = list(), 
              precision = 0,
              
              nodes_add_samples = False,
              nodes_add_value = True,
              nodes_add_class = True,
              show_proportn = False,
              node_id_show = False,

              testing ='on',
              test_size = 0.1,
              
              entropy_decrease_thresh=0.03, 
              min_samples_split = 50,
              min_samples = 2,
              ccp_alpha = 0.00001,
              
              return_vimps = False,
              return_preds = False
             ):

#     global mod, expected_y, predicted_y, df_mod_iv, iv_list, viz, df_vi

    depth = depth_param
    
    df_mod_iv = df_iv
    iv_list = df_mod_iv.columns
    print(iv_list)
    
    X = df_mod_iv.values
    np.random.seed(k)

    y = df_comb[y_var].values

    if testing=='on':
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify = y, random_state=k)
    else: 
        x_train, y_train = X, y
        x_test, y_test = X, y

    mod = DecisionTreeClassifier(
        random_state=k, 
        criterion='entropy',
        max_depth=depth_param,
        ccp_alpha=ccp_alpha,
        min_samples_split=min_samples_split,
        min_samples_leaf= min_samples if min_samples == min_samples else 2, 
    #     min_samples_split=20, 
        min_impurity_decrease= entropy_decrease_thresh
    )

    mod.fit(x_train, y_train)
    out_str = "{}_{}_{}_{}Cat_depth{}.dot".format(vert_lev_str_append,month,y_str,num_classes,depth_param)
    
    expected_y  = y_test
    predicted_y = mod.predict(x_test)

    print('mod accuracy score with criterion entropy: {0:0.4f}'.format(accuracy_score(expected_y, predicted_y)))
    print(classification_report(expected_y, predicted_y))
    print(confusion_matrix(expected_y, predicted_y))
    
    df_vi = pd.DataFrame({'var_name':iv_list, 'imp': np.round(mod.feature_importances_, 2)})\
    .sort_values('imp', ascending=False).copy()
    print('df_vi = ', df_vi)

    # Round the values in the tree
#     round_tree_values(mod.tree_)
    
#     for i in range(len(mod.tree_.value)):
#         print(mod.tree_.value[i])
#         print(np.round(mod.tree_.value[i][0]*mod.tree_.n_node_samples[i], 0))
#         mod.tree_.value[i][0] = np.round(mod.tree_.value[i][0]*mod.tree_.n_node_samples[i], 0)

    # Plot the decision tree graph
    export_graphviz(
        mod,
        out_file=out_str,
        feature_names=df_mod_iv.columns,
        class_names=labels,
        rotate=rotation,
        rounded=True,
        precision=1, # change this for num decimals on node text 
        filled=col_param, # leaf node colors 
        special_characters=True,
        node_ids=node_id_show,
        impurity=False,
        proportion=show_proportn,
        leaves_parallel=False,
        
    #     impurity=True,
        label='all',
        fontname='verdana'
     )

#     pydot_graph.set_size('"10,8!"')

    fig,ax = plt.subplots(figsize=(15, 10)) # Resize figure

    _ = tree.plot_tree(
        mod, 
        filled=col_param, 
        rounded=True,
        impurity=False,
        precision=precision,
        class_names=labels,
        proportion=show_proportn, 
        node_ids=True,
                   
        feature_names=df_iv.columns,
        ax=ax, 
        fontsize=fontsize, 
#       cmap = 'coolwarm'
    )
    
    plt.savefig(out_str.replace('.dot','_sklearn.png'))
    plt.show()
    
#     print(expected_y, predicted_y)
    
    ### for modified file for easy reading, uncomment below lines, set impurity=False above, 
    # and below that, open(mod_out_str) as f, Source(dot_graph).render(mod_out_str.replace('.dot', '')

#     if accuracy_score(expected_y, predicted_y) > 0.3:

# https://stackoverflow.com/questions/44821349/python-graphviz-remove-legend-on-nodes-of-decisiontreeclassifier
    PATH = dir_mods + out_str if dir_mods != '' else out_str
    f = pydot.graph_from_dot_file(PATH)[0].to_string()

    if nodes_add_samples == False:
        f = re.sub('(\\\\nsamples = [0-9]+)', '', f) #(\\\\nvalue = \[[0-9]+, [0-9]+, [0-9]+\])
        f = re.sub('(samples = [0-9]+)\\\\n', '', f) # (\\\\nvalue = \[[0-9]+, [0-9]+, [0-9]+\])
    if nodes_add_value == False:
        f = re.sub('(\\\\nvalue = \[[0-9]+, [0-9]+, [0-9]+\])', '', f) #
        f = re.sub('(\\\\nvalue = \[[0-9]+, [0-9]+, [0-9]+\])\\\\n', '', f) # (\\\\nvalue = \[[0-9]+, [0-9]+, [0-9]+\])
    if nodes_add_class == False:
        f = re.sub('(\\\\nclass = \([0-9]+, [0-9]+\])', '', f) #
        f = re.sub('(\\\\nclass = \([0-9]+, [0-9]+\])\\\\n', '', f) # (\\\\nvalue = \[[0-9]+, [0-9]+, [0-9]+\])
        
#     f.write_png('original_tree.png')
#     f.set_size('"5,5!"')
#     f.write_png('resized_tree.png')

    mod_out_str = '{}_modified.dot'.format(out_str.replace('.dot',''))

    with open(mod_out_str, 'w') as file:
        file.write(f)

    with open(mod_out_str) as f:
        dot_graph = f.read()
    
    Source(dot_graph).render(mod_out_str.replace('.dot', ''), format='svg', view=False)
    Source(dot_graph).render(mod_out_str.replace('.dot', ''), format='pdf', view=False)
    
    print(mod_out_str.replace('.dot', ''))

#     if return_vimps == True:
#         return iv_list, df_vi
#     elif return_preds == True:
#         return expected_y, predicted_y
    return iv_list, df_vi, expected_y, predicted_y, x_train, y_train, mod

# -




# +
def standardize_col(y_var = str, df = pd.DataFrame()):
    return ((df[y_var] - df[y_var].mean())/df[y_var].std()).values

def standardize_df(X = pd.DataFrame(), y = pd.Series(dtype=np.float64)):
    if len(X.columns) > 1:
        X_iv = pd.DataFrame(StandardScaler().fit_transform(X), columns = X.columns, index = X.index)
    elif len(X.columns) == 1:
        X_iv = pd.DataFrame(StandardScaler().fit_transform(np.array(X.iloc[:,0]).reshape(-1,1)), columns = [X.iloc[:,0].name], index = X.index)
    if len(y) > 0: 
        y_dv = pd.DataFrame(StandardScaler().fit_transform(np.array(y).reshape(-1,1)), columns = [y.name], index = y.index)
        return X_iv, y_dv
    else: 
        return X_iv
# -



g = 9.80665

# +
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

def results_summary_to_dataframe(results, print_res=True):
#     global results_df, results_df2
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = np.round(results.params, 2)
#     print(pd.DataFrame(results.conf_int())
    
    conf_lower = np.round(results.conf_int().iloc[:,0], 2)
    conf_higher = np.round(results.conf_int().iloc[:,1], 2)

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

def mlr(X = pd.DataFrame(), y = pd.Series(dtype=np.float64), testing = True, 
        test_size = 0.1, figsize=(16,5), print_res = True, coef_thresh = 0.2, plot_coefs = True,
       standardize=True, mean_center = True):
    
    global X_iv, x_train, x_test, y_train, y_test, regr, df_mod, results_df, results_df2, model
    
    if standardize==True:
        if mean_center == True:
            X_iv, y = standardize_df(X, y)
#             if len(X.columns) > 1:
#                 X_iv, y = standardize_df(X, y)
#             else: 
#                 xcol = X.columns[0]
#                 X_iv = X[xcol].tolist()
#                 y = y.tolist()
                
#     pd.DataFrame(StandardScaler().fit_transform(X), columns = X.columns)
#     y = pd.DataFrame(StandardScaler().fit_transform(np.array(y).reshape(-1,1)), columns = [y.name])
        else: 
            tmp = pd.concat([X, y], axis=1)/pd.concat([X, y], axis=1).std()
            X_iv = tmp.iloc[:,:-1]
            y = tmp.iloc[:,-1]
    else:
        X_iv, y = X, y
    
    if testing == True:
        x_train, x_test, y_train, y_test = train_test_split(X_iv, y, test_size = test_size)
    else:
        x_train, y_train = X_iv, y
        x_test, y_test = X_iv, y

    #add constant to predictor variables
    x_train = sm.add_constant(x_train)

    #fit linear regression model
    model = sm.OLS(y_train, x_train).fit()
    
    #view model summary
    r_sq = np.round(model.rsquared_adj, 2)
    modsum = model.summary()
    
    if print_res == True:
        print(modsum.tables[0])
        
    results_df, results_df2 = results_summary_to_dataframe(model, print_res=print_res)
    
#     if r_sq > 0.5:
        # fig,ax = plt.subplots()
    df_plt = results_df2.loc[np.abs(results_df2.coeff) > coef_thresh].copy()
    
    if plot_coefs == True:
        plt.figure(figsize = figsize)
        plt.plot(df_plt.index, df_plt.coeff)
#         plt.axhline(0.2)
    # ax.set_xlabel(ax.get_xlabel(),rotation='vertical')
        _ = plt.xticks(df_plt.index,rotation=90) # , fontsize=40

    return results_df, results_df2, model, y_train


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

def path_inspect(ds = xr.Dataset(), var_a = 'w_down', var_b = 'v_anom'):
    params = {
#     'legend.fontsize': 10,
#           'legend.title_fontsize': 10,
#           'figure.figsize': (15, 5),
         'axes.labelsize': 15, # this controls labelsize of both x and y axis of main plot as well as colorbar  
         'axes.titlesize':25, # pot title size
         'xtick.labelsize':12,
         'ytick.labelsize':12 # this controls yticks labelsize of both main plot and colorbar 
    }

    plt.rcParams.update(params)

    path1_conds = ((ds_analysis.wAnom_AdvFlx_z_dseClmt > 44) & 
        (ds_analysis.uclmt_AdvFlx_x_dseAnom > -300))
    # positive anomalous fluxes in zonal and meridional dirs

    pos_path1 = ds_analysis.where(path1_conds, drop=True)
    print(pos_path1.dims, 'number of dates in this path')

    nrow=2; ncol=2
    fig,ax = plt.subplots(nrow,ncol,figsize=(16,12))

    pos_path1.plot.scatter(
        x = 'gradz_dseAnom', y = 'w_down_anom', ax=ax[0,0], 
                          )
    ax[0,0].set_title(pos_path1.dims)

    pos_path1.plot.scatter(x = 'gradz_dseClmt', y = 'w_down_clmt', ax=ax[0,1])
    ax[0,1].set_title(pos_path1.dims)

    pos_path1.plot.scatter(x = 'wAnom_AdvFlx_z_dseClmt', y = 'wclmt_AdvFlx_z_dseAnom', ax=ax[1,0])
    ax[1,0].set_title(pos_path1.dims)

    pos_path1.plot.scatter(x = 'w_down_anom', y = 'wAnom_AdvFlx_z_dseAnom', ax=ax[1,1])
    ax[1,1].set_title(pos_path1.dims)

    for i in range(nrow):
        for j in range(ncol):
            ax[i,j].axhline(0)
            ax[i,j].axvline(0)


    plt.tight_layout()


import seaborn as sn

# +
sn.set_style(style='white') 

def corplot(df_cormat = pd.DataFrame(), vmin = -1, vmax = 1, ax = list(), figsize=(6,4)):
    mask = np.zeros_like(df_cormat, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=figsize)
    with sn.axes_style('white'):
        sn.heatmap(
            df_cormat.round(2),
            mask=mask,
            cbar=False,
            vmin=vmin, vmax=vmax,
            square=False,annot=True,annot_kws={"size": 10},fmt='g',
            cmap = 'RdYlGn',linewidths=0.5,ax=ax
        )
#     plt.grid(None)
#     plt.rcParams["axes.grid"] = False
    ax.set_facecolor('white')


    plt.tight_layout()

# -



# +
def prep_corrdata(ni_var = str, nh_var=str, plev=300, corr_dict = dict, nh_dict_to_use = dict, ds_NI_LL = xr.Dataset(), 
                 ds_out = xr.Dataset(), max_lag = 10):

    for i in range(-1*max_lag, max_lag+1):
        str_i = str(i).replace("-", "neg")
        lag_str = f"lag{str_i}"
        print(lag_str)

        # Create corr_dict dictionary to calculate corrs for 10 lats at a time

        corr_dict[f'{lag_str}'] = {} 
        for j in range(len(nh_dict_to_use[nh_var])):

            ds_tmp = nh_dict_to_use[nh_var][j][nh_var].sel(isobaricInhPa=plev)
            ds_tmp = ds_tmp.shift(date=i)

            ## positive shift in date: previous days
            ## negative shift in date: next days

            if i > 0:
                ds_tmp = \
                ds_tmp.where(
                    ds_tmp.date.dt.day > i 
                )
            if i < 0:
                ds_tmp = \
                ds_tmp.where(
                    ds_tmp.date.dt.day < 30 + i 
                )
            print('calc corr')
            corr_dict[f'{lag_str}'][j] = \
            xr.corr(
                ds_NI_LL[ni_var], 
                ds_tmp, 
                dim='date'
            ).compute().round(3)

            print("max_corr", corr_dict[f'{lag_str}'][j].max().data)

        ## merge vaues of corr_dict dictionary

        globals()[f'corr_{nh_var}_{plev}_{lag_str}'] = \
        xr.concat(list(corr_dict[f'{lag_str}'].values()), dim='longitude')
        
        ds_out[f'{lag_str}'] = globals()[f'corr_{nh_var}_{plev}_{lag_str}']        
                  
        del ds_tmp
        gc.collect()

#     globals()[f'corr_dict_{nh_var}_{plev}'] = copy.deepcopy(corr_dict)
#     del corr_dict
    return ds_out
    gc.collect()
# -








