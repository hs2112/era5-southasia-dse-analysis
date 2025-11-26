# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python [conda env:my_conda]
#     language: python
#     name: conda-env-my_conda-py
# ---

import numpy as np

# +

Cp = 1005 # J K-1 kg-1
g = 9.80665
R = 287

Rearth = 6378*10**3
steps_x = 4*10
del_lambda = 0.25*np.pi/180
del_phi = 0.25*np.pi/180


# +
# def Rlat_fn(ds):
#     return Rearth*np.cos(ds.latitude*np.pi/180)

# Rlat = Rlat_fn(ds = ds_mst)

# Ayz = (Rearth*del_phi)*(ds_mst['z_diff_down']/g) # select lon of choice for all corresponding NI lats
# Axz = (Rlat*del_lambda)*(ds_mst['z_diff_down']/g) # select latitude of choice for all corresponding NI lons
def Axy_fn(ds): 
    Rlat = Rearth*np.cos(ds.latitude*np.pi/180)
    
    return Rearth**2*del_lambda*(
    np.sin((Rlat.latitude + 0.25)*np.pi/180) - np.sin(Rlat.latitude*np.pi/180)
)

# M = (Axy_2*ds_mst['z_diff_down']/g*.5*(ds_mst['density'] + ds_mst['density'].shift(isobaricInhPa=-1)))\
# .sel(latitude=slice(24,30.75), longitude=slice(68,77.75))

# mass_wgts = M/(M.sum(['latitude','longitude']))
# print(mass_wgts.isel(isobaricInhPa=0, date=0).sum().data)

# mass_wgts = mass_wgts.rename('mass_wgts')

# V = (Axy_2*ds_mst['z_diff_down']/g).sel(latitude=slice(24,30.75), longitude=slice(68,77.75))
# vol_wgts = V/(V.sum(['latitude','longitude']))
# vol_wgts = vol_wgts.rename('vol_wgts')
# print(vol_wgts.isel(isobaricInhPa=0, date=0).sum().data)

## M and V defined for each gridpoint
# -



# +
nebox_lon = [85, 90]
nebox_lat = [22.5, 27.5]

sebox_lon = [79, 83]
sebox_lat = [14, 18]

nibox_lon = [68, 78]
nibox_lat = [24, 31]

# -
lon_coords_EW = [s/4 for s in range(0,180*4 )] + [s/4 for s in range(-180*4,0)]

import custom_funcs
from custom_funcs import *

redcolor = '#ca4842' #'#67001f'
bluecolor = '#3884bb' #'#053061'


t2m_lag1_lab_map = {
    
    0:'negative',
    1:'neutral',
    2:'positive',    
}

mon_map = {
    3: 'mar', 
    4: 'apr', 
    5: 'may', 
    6: 'jun'
}

rolling_clmt_file_map = {
    
    'u':'global_rolling_clmtlgy_prmnsn_u_component_of_wind_1980_2022.nc',
    'v': 'global_rolling_clmtlgy_prmnsn_v_component_of_wind_1980_2022.nc', 
    'z': 'global_rolling_clmtlgy_prmnsn_geopotential_1980_2022.nc', # final version
#     't': 'SA_rolling_clmtlgy_prmnsn_temperature_1980_2022.nc', # doesn't have feb normalized rolling clmt
    't': 'global_rolling_clmtlgy_prmnsn_temperature_1980_2022.nc', # final version
    'w': 'global_rolling_clmtlgy_prmnsn_vertical_velocity_1980_2022.nc'
}


# +
clmt_file_map = {
    
#     'u': 'dly_clmtlgy_stdPlevs_prmnsn_u_component_of_wind_1995_2022.nc',
    'u':'dly_clmtlgy_stdPlevs_prmnsn_u_component_of_wind_1980_2022.nc',
#     'v': 'dly_clmtlgy_stdPlevs_prmnsn_v_component_of_wind_1995_2022.nc',
    'v': 'dly_clmtlgy_stdPlevs_prmnsn_v_component_of_wind_1980_2022.nc', 
    'z': 'dly_clmtlgy_stdPlevs_prmnsn_geopotential_1990_2022.nc', 
    't': 'dly_clmtlgy_stdPlevs_prmnsn_temperature_1990_2022.nc', 
#     'w': 'SA_dly_clmtlgy_stdPlevs_prmnsn_vertical_velocity_2000_2021.nc'
    'w': 'dly_clmtlgy_stdPlevs_prmnsn_vertical_velocity_1980_2022.nc'
}
# -

name_dict = {
    'uAnom_AdvFlx_x_dseAnom': 'u\'_s\'',
    'vAnom_AdvFlx_y_dseAnom': 'v\'_s\'',
    'wAnom_AdvFlx_z_dseAnom': 'w\'_s\'',
    
    'uAnom_AdvFlx_x_dseClmt': 'u\'_' + u's\u0305',
    'vAnom_AdvFlx_y_dseClmt': 'v\'_' + u's\u0305',
    'wAnom_AdvFlx_z_dseClmt': 'w\'_' + u's\u0305',
    
    'uclmt_AdvFlx_x_dseAnom': u'u\u0305' + '_s\'',
    'vclmt_AdvFlx_y_dseAnom': u'v\u0305' + '_s\'',
    'wclmt_AdvFlx_z_dseAnom': u'w\u0305' + '_s\'',
    
    'uclmt_AdvFlx_x_dseClmt': u'u\u0305_' + u's\u0305',
    'vclmt_AdvFlx_y_dseClmt': u'v\u0305_' + u's\u0305', 
    'wclmt_AdvFlx_z_dseClmt': u'w\u0305_' + u's\u0305' 
}

print('$\mathregular{\mathcal{S}\'_{x}}$')

# +
# dx_s_anom = 's'+'\''+get_sub('x')
# dy_s_anom = 's'+'\''+get_sub('y')
# # dz_s_anom = 's'+'\''+get_sub('z')
# dz_s_anom = '$s_{z}$'+'\''
# dx_sbar = u's\u0305'+ get_sub('x')
# dy_sbar = u's\u0305'+ get_sub('y')
# # dz_sbar = u's\u0305'+ get_sub('z')
# -




# $\mathrm{u'}$

# $\overline{u}$

# $\overline{\mathcal{S}}_{x}$

# $\mathcal{S}'_{x}$

# +
# ## mathrm for u, v, w version

# dx_s_anom = '$\mathcal{S}\'_{x}$'
# dy_s_anom = '$\mathcal{S}\'_{y}$'
# dz_s_anom = '$\mathcal{S}\'_{z}$'

# dx_sbar = '$\overline{\mathcal{S}}_{x}$'
# dy_sbar = '$\overline{\mathcal{S}}_{y}$'
# dz_sbar = '$\overline{\mathcal{S}}_{z}$'

# ubar = '$\overline{u}$'
# vbar = '$\overline{v}$'
# wbar = '$\overline{w}$'

# uanom = '$\mathrm{u\'}$'
# vanom = '$\mathrm{v\'}$'
# wanom = '$\mathrm{w\'}$' 


# +
## mathtex version for u, v, w

dx_s_anom = '$\mathcal{S}\'_x$'
dy_s_anom = '$\mathcal{S}\'_y$'
dz_s_anom = '$\mathcal{S}\'_z$'

dx_sbar = u'$\mathcal{S}\u0305_x$'
dy_sbar = u'$\mathcal{S}\u0305_y$'
dz_sbar = u'$\mathcal{S}\u0305_z$'


ubar = u'${u\u0305}$'
vbar = u'${v\u0305}$'
wbar = u'${w\u0305}$'

uanom = '$u\'$'
vanom = '$v\'$'
wanom = '$w\'$'

name_dict_final = {
    
     'u\'_s\'': uanom + dx_s_anom,
     'v\'_s\'': vanom + dy_s_anom,
    'w\'_s\'': wanom + dz_s_anom,
    
     'u\'_' + u's\u0305': uanom + dx_sbar,
     'v\'_' + u's\u0305': vanom + dy_sbar,
     'w\'_' + u's\u0305': wanom + dz_sbar,
    
     u'u\u0305' + '_s\'': ubar + dx_s_anom,
     u'v\u0305' + '_s\'': vbar + dy_s_anom,
     u'w\u0305' + '_s\'': wbar + dz_s_anom,
    
     u'u\u0305_' + u's\u0305': ubar + dx_sbar, 
     u'v\u0305_' + u's\u0305': vbar + dy_sbar, 
     u'w\u0305_' + u's\u0305': wbar + dz_sbar 

}


# +
# print(u'${u\u0305}$')

# print(dx_sbar)

# +
# [name_dict_final[key] for key in name_dict.values()]
# -





# +
# name_dict_2 = {

#     'u_anom': 'u\'', 
#     'v_anom': 'v\'', 
#     'w_anom': 'w\'', 
#     'gradx_dseAnom': 'dx(s\')', 
#     'grady_dseAnom': 'dy(s\')', 
#     'gradz_dseAnom': 'dz(s\')', 
#     'u_clmt': u'u\u0305', 
#     'v_clmt': u'v\u0305', 
#     'w_clmt': u'w\u0305', 
#     'gradx_dseClmt': 'dx({})'.format(u's\u0305'), 
#     'grady_dseClmt': 'dy({})'.format(u's\u0305'), 
#     'gradz_dseClmt': 'dz({})'.format(u's\u0305') 
# }
# -

name_dict_2 = {

    'u_anom': uanom, 
    'v_anom': vanom, 
    'w_anom': wanom, 
    'gradx_dseAnom': dx_s_anom, 
    'grady_dseAnom': dy_s_anom, 
    'gradz_dseAnom': dz_s_anom, 
    'u_clmt': ubar, 
    'v_clmt': vbar, 
    'w_clmt': wbar, 
    'gradx_dseClmt': dx_sbar, 
    'grady_dseClmt': dy_sbar, 
    'gradz_dseClmt': dz_sbar 
}



x_RD_vars = [uanom + dx_s_anom, uanom + dx_sbar, ubar + dx_s_anom , ubar + dx_sbar]
y_RD_vars = [vanom + dy_s_anom, vanom + dy_sbar, vbar + dy_s_anom , vbar + dy_sbar]
z_RD_vars = [wanom + dz_s_anom, wanom + dz_sbar, wbar + dz_s_anom , wbar + dz_sbar]


RD_dict = {
    'u': x_RD_vars, 
    'v': y_RD_vars, 
    'w': z_RD_vars,    
}

x_main = ubar + dx_s_anom
z_main = wanom + dz_sbar
y_main = vanom + dy_sbar

Tmain_easy = {
    'y_main': y_main,
    'z_main': z_main,
    'x_main': x_main,
}



var_lst_fund = list(name_dict_2.values())


print(var_lst_fund)

print(name_dict_final.values())

var_lst_adv = list(name_dict.values())
var_lst_adv_trtd = list(name_dict_final.values())


var_lst_adv_trtd

# +
set_main = list(Tmain_easy.values()) #[name_dict_final[key] for key in list()]
print(set_main)

eddy_eddy = [uanom + dx_s_anom, vanom + dy_s_anom, wanom + dz_s_anom]
print(eddy_eddy)

mean_mean = [ubar + dx_sbar, vbar + dy_sbar, wbar + dz_sbar]
print(mean_mean)

set_insig = [s for s in name_dict_final.values() if s not in set_main + eddy_eddy + mean_mean]
print(set_insig)


# +
# set(name_dict.values()).intersection(set(eddy_eddy))
# name_dict.values()

# +
# eddy_eddy
# set(name_dict_final.values()).difference(set(set_main)) 
# -

regime_cls_map = {
    
    'Neutral':'Neutral', 
    'amplif_QL_med':'amplif_QL', 
    'amplif_QL_lrg':'amplif_QL', 
    'amplif_QL_NL_negus':'amplif_QL_NL', 
    'amplif_NL_y':'amplif_NL',
    'amplif_NLsat_x':'amplif_NLsat', 
    'amplif_NL_x':'amplif_NL',
    'amplif_QL_NL_posus':'amplif_QL_NL', 
    'amplif_NLsat_y':'amplif_NLsat',
    'amplif_NL_z': 'amplif_NL',
    'decay_NLsat_z':'decay_NLsat', 
    'decay_QL':'decay_QL',
    'decay_NL_x':'decay_NL', 
    'decay_QL_NL_y':'decay_QL_NL', 
    'decay_NL_y':'decay_NL',
    'decay_QL_NL_x':'decay_QL_NL', 
    'decay_NLsat_y':'decay_NLsat', 
    'decay_NLsat_x':'decay_NLsat',
    'nan':'nan'
}



