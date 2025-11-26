import numpy as np, xarray as xr, gc
from datetime import datetime as dt


# +

Cp = 1005 # J K-1 kg-1
g = 9.80665 # J kg-1

Rearth = 6378*10**3
steps_x = 4*10
del_lambda = 0.25*np.pi/180
del_phi = 0.25*np.pi/180


da_lat = xr.DataArray(
    data = [24.  , 24.25, 24.5 , 24.75, 25.  , 25.25, 25.5 , 25.75, 26.  , 26.25,
       26.5 , 26.75, 27.  , 27.25, 27.5 , 27.75, 28.  , 28.25, 28.5 , 28.75,
       29.  , 29.25, 29.5 , 29.75, 30.  , 30.25, 30.5 , 30.75, 31.  ],
    dims = ['latitude'],
    coords = dict(latitude = [24.  , 24.25, 24.5 , 24.75, 25.  , 25.25, 25.5 , 25.75, 26.  , 26.25,
       26.5 , 26.75, 27.  , 27.25, 27.5 , 27.75, 28.  , 28.25, 28.5 , 28.75,
       29.  , 29.25, 29.5 , 29.75, 30.  , 30.25, 30.5 , 30.75, 31.  ]),
    name = 'Rlat'
)

Rlat = Rearth*np.cos(da_lat*np.pi/180)

# -
def Rlat_fn(ds):
    return Rearth*np.cos(ds.latitude*np.pi/180)


# ## lat, lon, pressure levs should all be sorted ascending for the following funcs

# +
# this is for taking points outside the ni box
# and for any arrangement of lons

# Fwd diff = next - current
def deriv_fwddiff(sclr = xr.DataArray(), dim = 'isobaricInhPa'): 
    dim_ord = 'dsc' if sclr[dim][0] > sclr[dim][1] else 'asc' 
    print(dim, dim_ord)
    
    shift_by = 1 if dim_ord == 'dsc' else -1
    
    print('term_hi shifted by {}'.format(shift_by))
    term_hi = eval('sclr.shift({} = {})'.format(dim, shift_by)) 
    term_lo = sclr 
    sclr_diff = term_hi - term_lo 

    ind_hi = 0 if dim_ord == 'dsc' else 1 
    ind_lo = 1 if ind_hi == 0 else 0 

    if dim == 'longitude':
        Rlat = Rearth*np.cos(sclr[dim]*np.pi/180)
        del_lambda = (sclr[dim][ind_hi] - sclr[dim][ind_lo])*np.pi/180
        del_x = Rlat*del_lambda
        denom = del_x
    
    if dim== 'latitude':
        del_phi = ((sclr[dim][ind_hi] - sclr[dim][ind_lo])*np.pi/180)
        del_y = Rearth*del_phi
        denom = del_y

    if dim == 'isobaricInhPa':
        term_p_hi = eval('sclr[dim].shift({} = {})'.format(dim, shift_by))
        term_p_lo = sclr[dim]
        
        del_p = (term_p_hi - term_p_lo)*100 
        denom = del_p
        print(del_p)
    
    return sclr_diff/denom


# +
# this is for taking points outside the ni box
# and for any arrangement of lons

# Fwd diff = next - current
def grad_x_fwddiff(sclr = xr.DataArray()):
    lon_ord = 'dsc' if sclr.longitude[0] > sclr.longitude[1] else 'asc'
    
    term_hi = sclr.shift(longitude = 1) if lon_ord == 'dsc' else sclr.shift(longitude = -1)
    term_lo = sclr
    sclr_diff = term_hi - term_lo

    ind_hi = 0 if lon_ord == 'dsc' else 1 
    ind_lo = 1 if lon_ord == 'dsc' else 0

    Rlat_times_del_lambda = (Rearth*np.cos(sclr.latitude*np.pi/180)*((sclr.longitude[ind_hi] - sclr.longitude[ind_lo])*np.pi/180))

    return sclr_diff/Rlat_times_del_lambda


# +
# this is for taking points outside the ni box
# and for any arrangement of lons

# Fwd diff = next - current
def grad_y_fwddiff(sclr = xr.DataArray()):
    lat_ord = 'dsc' if sclr.latitude[0] > sclr.latitude[1] else 'asc'
    
    term_hi = sclr.shift(latitude = 1) if lat_ord == 'dsc' else sclr.shift(latitude = -1)
    term_lo = sclr
    sclr_diff = term_hi - term_lo

    ind_hi = 0 if lat_ord == 'dsc' else 1 
    ind_lo = 1 if lat_ord == 'dsc' else 0

    Rlat_times_del_phi = (Rearth*np.cos(sclr.latitude*np.pi/180)*((sclr.latitude[ind_hi] - sclr.latitude[ind_lo])*np.pi/180))

    return sclr_diff/Rlat_times_del_phi


# +
# this is for taking points outside the ni box

# Fwd diff = next - current
def grad_p_omega_fwddiff(sclr = xr.DataArray()):
    p_ord = 'dsc' if sclr.isobaricInhPa[0] > sclr.isobaricInhPa[1] else 'asc'
    
    term_hi = sclr.shift(isobaricInhPa = 1) if p_ord == 'dsc' else sclr.shift(isobaricInhPa = -1)
    term_lo = sclr
    sclr_diff = term_hi - term_lo

    ind_hi = 0 if p_ord == 'dsc' else 1 
    ind_lo = 1 if p_ord == 'dsc' else 0

    del_p = ((sclr.isobaricInhPa[ind_hi] - sclr.isobaricInhPa[ind_lo])*100)

    return sclr_diff/del_p



# +
# this is for taking points outside the ni box
# and for any arrangement of lons

# Bck diff = current - prv

def grad_x_bckdiff(sclr = xr.DataArray()):
    lon_ord = 'dsc' if sclr.longitude[0] > sclr.longitude[1] else 'asc'
    term_hi = sclr
    term_lo = sclr.shift(longitude = -1) if lon_ord == 'dsc' else sclr.shift(longitude = 1)    
    sclr_diff = term_hi - term_lo
    
    ind_hi = 0 if lon_ord == 'dsc' else 1 
    ind_lo = 1 if lon_ord == 'dsc' else 0
    
    Rlat_times_del_phi = (Rearth*np.cos(sclr.latitude*np.pi/180)*((sclr.longitude[ind_hi] - sclr.longitude[ind_lo])*np.pi/180))

    return sclr_diff/Rlat_times_del_phi
# -



# this is for taking points outside the ni box
# and for any arrangement of lons
def grad_x_2(sclr = xr.DataArray()):
    if sclr.longitude[0] > sclr.longitude[1]: # desc
        Rlat_times_del_phi = (Rearth*np.cos(sclr.latitude*np.pi/180)*((sclr.longitude[0] - sclr.longitude[1])*np.pi/180))
        sclr_diff = sclr.shift(longitude = 1) - sclr
    if sclr.longitude[0] < sclr.longitude[1]: # asc
        Rlat_times_del_phi = (Rearth*np.cos(sclr.latitude*np.pi/180)*((sclr.longitude[1] - sclr.longitude[0])*np.pi/180))
        sclr_diff = sclr.shift(longitude = -1) - sclr    
    return sclr_diff/Rlat_times_del_phi

# +
# # this is for taking points outside the ni box
# # and for any arrangement of lons
# def grad_y_2(sclr = xr.DataArray()):
#     if sclr.longitude[0] > sclr.longitude[1]: # desc
#         Rlat_times_del_phi = (Rearth*np.cos(sclr.latitude*np.pi/180)*((sclr.longitude[0] - sclr.longitude[1])*np.pi/180))
#         sclr_diff = sclr.shift(longitude = 1) - sclr
#     if sclr.longitude[0] < sclr.longitude[1]: # asc
#         Rlat_times_del_phi = (Rearth*np.cos(sclr.latitude*np.pi/180)*((sclr.longitude[1] - sclr.longitude[0])*np.pi/180))
#         sclr_diff = sclr.shift(longitude = -1) - sclr    
#     return sclr_diff/Rlat_times_del_phi
# -



# +
# this is for taking points outside the ni box
# and for any arrangement of lons
# def grad_x_2(sclr):
#     if sclr.latitude[0] > sclr.latitude[1]:
#         return (sclr - sclr.shift(longitude = 1))/(Rearth*np.cos(sclr.latitude*np.pi/180)*((sclr.latitude[0] - sclr.latitude[1])*np.pi/180))
#     else:
#         return (sclr.shift(longitude = -1) - sclr)/(Rearth*np.cos(sclr.latitude*np.pi/180)*((sclr.latitude[1] - sclr.latitude[0])*np.pi/180))


# -

# lon is always arranged in ascending so this works
# but Rlat is defined only within the ni box
def grad_x(sclr):
    return 1/Rlat*(sclr.shift(longitude=-1) - sclr)/((sclr.longitude[1] - sclr.longitude[0])*np.pi/180)

# forward difference
def grad_y(sclr):
    if sclr.latitude[1] > sclr.latitude[0]: # ascending
        return 1/Rearth*(sclr.shift(latitude=-1) - sclr)/((sclr.latitude[1] - sclr.latitude[0])*np.pi/180)
    if sclr.latitude[0] > sclr.latitude[1]: # descending
        return 1/Rearth*(sclr.shift(latitude=1) - sclr)/((sclr.latitude[0] - sclr.latitude[1])*np.pi/180)

def grad_z(sclr, zdiff):
    return (sclr.shift(isobaricInhPa=-1) - sclr)/(zdiff)


# This is 'actual' backward difference based first derivative
# Works only because zdiff is passed as backward difference (Z(p0) - Z(p0 + delP)) irrespective of ordering of pressure dim
def grad_z_act(sclr, zdiff):
    if (sclr.isobaricInhPa[0] < sclr.isobaricInhPa[1]):
        return (sclr - sclr.shift(isobaricInhPa=-1))/(zdiff) # scalar(400 hPa) - scalar(500 hPa)
    if (sclr.isobaricInhPa[0] > sclr.isobaricInhPa[1]):
        return (sclr - sclr.shift(isobaricInhPa=1))/(zdiff) # scalar(400 hPa) - scalar(500 hPa)




import sys
sys.path.insert(1, '/home/data/lab_hardik/udf/')

import read_data_DSE_flux_funcs
from read_data_DSE_flux_funcs import * 


# +
def u_gradSclr(u = xr.DataArray(), #ds_bflx['u_anom'], 
               v = xr.DataArray(), #ds_bflx['v_anom'], 
               w_down = xr.DataArray(), #ds_bflx['w_down_anom'], 
               scalar = xr.DataArray(), #ds_bflx['pt_dse'], 
               u_type = 'daily', 
               scalar_type = 'climatology', 
               z_diff = xr.DataArray(), #ds_bflx['z_diff']/g
#                Rearth = 6378*10**3, Rlat = xr.DataArray(), del_phi = , del_lambda
              ):
    
    u_gradx = xr.DataArray(); v_grady = xr.DataArray(); w_gradz = xr.DataArray()

    gc.collect()
    if u_type == 'daily':
        if scalar_type == 'climatology':

            for name, grp in u.groupby(u.date.dt.strftime('%d-%b')):
                temp = grp*grad_x(scalar.sel(strftime=name))
                u_gradx = temp if len(u_gradx.shape)==0 else xr.concat([u_gradx, temp], dim='date')
                del temp
                gc.collect()
            
            for name, grp in v.groupby(v.date.dt.strftime('%d-%b')):
                temp = grp*grad_y(scalar.sel(strftime=name))
                v_grady = temp if len(v_grady.shape)==0 else xr.concat([v_grady, temp], dim='date')
                del temp
                gc.collect()

            for name, grp in w_down.groupby(w_down.date.dt.strftime('%d-%b')):
                temp = grp*grad_z(
                    scalar.sel(strftime=name), 
                    z_diff.sel(date=z_diff.date.dt.strftime('%d-%b') == name)
                )
                
                w_gradz = temp if len(w_gradz.shape)==0 else xr.concat([w_gradz, temp], dim='date')
                del temp
                gc.collect()
                
            gc.collect()

        elif scalar_type == 'daily':
            u_gradx = u*grad_x(scalar)
            v_grady = v*grad_y(scalar)
            w_gradz = w_down*grad_z(scalar, z_diff)
            gc.collect()
        
        u_gradx = u_gradx.sortby('date'); v_grady = v_grady.sortby('date'); w_gradz = w_gradz.sortby('date')
        gc.collect()
        
    elif u_type == 'climatology':
        if scalar_type == 'climatology':
            u_gradx = u*grad_x(scalar)
#             /Rlat*((scalar.shift(longitude=-1) - scalar)/del_lambda)
            v_grady = v*grad_y(scalar)
#     /Rearth*((scalar.shift(latitude=-1) - scalar)/del_phi)
            
            da_z_diff = z_diff #ds_bflx['z_diff']/g #.sel(date=ds_bflx.date.dt.year.isin([2016, 2017])) # for trial
            nrep = len(np.unique(da_z_diff.date.dt.year))
            
            gc.collect()

            for name, grp in da_z_diff.groupby(da_z_diff.date.dt.strftime('%d-%b')):
                temp = w_down.sel(strftime=name)*grad_z(scalar.sel(strftime=name), grp)
#                 ((scalar.sel(strftime=name).shift(isobaricInhPa=-1) - scalar.sel(strftime=name))/(grp/g))
                w_gradz = temp if len(w_gradz.shape)==0 else xr.concat([w_gradz, temp], dim='date')
                del temp
                gc.collect()
                gc.collect()
           
            w_gradz = w_gradz.sortby('date')
            
            u_gradx2 = \
            assign_dates_climatology(ds = u_gradx).swap_dims({'strftime':'date'})
            u_gradx2 = u_gradx2.transpose('date', 'isobaricInhPa', 'latitude', 'longitude')
            u_gradx3 = xr.DataArray(
                data = np.tile(u_gradx2.sel(isobaricInhPa = w_gradz.isobaricInhPa), (nrep,1,1,1)),
                dims = [ 'date', 'isobaricInhPa', 'latitude', 'longitude'],
                coords = dict(

                    isobaricInhPa = w_gradz.isobaricInhPa.data,
                    latitude = w_gradz.latitude.data,
                    longitude = w_gradz.longitude.data,
                    date = w_gradz.date.data
                )
            )
            
            gc.collect()

            # both values should be the same
            with u_gradx3.isel(date=[0,122])[:,0,0,0] as a:
                print(a[0].data == a[1].data)
    
            v_grady2 = \
            assign_dates_climatology(ds = v_grady).swap_dims({'strftime':'date'})
            v_grady2 = v_grady2.transpose('date', 'isobaricInhPa', 'latitude', 'longitude')
            
            v_grady3 = xr.DataArray(
                data = np.tile(v_grady2.sel(isobaricInhPa = w_gradz.isobaricInhPa), (nrep,1,1,1)),
                dims = [ 'date', 'isobaricInhPa', 'latitude', 'longitude'],
                coords = dict(

                    isobaricInhPa = w_gradz.isobaricInhPa.data,
                    latitude = w_gradz.latitude.data,
                    longitude = w_gradz.longitude.data,
                    date = w_gradz.date.data
                )

            )
            # both values should be the same
            with v_grady3.isel(date=[0,122])[:,0,0,0] as a:
                print(a[0].data == a[1].data)

            u_gradx3.name = 'u_gradx'
            v_grady3.name = 'v_grady'
            w_gradz.name = 'w_gradz'
            return u_gradx3, v_grady3, w_gradz
            exit()
                
        elif scalar_type == 'daily':
            for name, grp in scalar.groupby(scalar.date.dt.strftime('%d-%b')):
                
                u_temp = u.sel(strftime=name)*grad_x(grp)
#                 /Rlat*((grp.shift(longitude=-1) - grp)/del_lambda)
                
                u_gradx = u_temp if len(u_gradx.shape)==0 else xr.concat([u_gradx, u_temp], dim='date')
            
                v_temp = v.sel(strftime=name)*grad_y(grp)
#                 /Rearth*((grp.shift(latitude=-1) - grp.sel(strftime=name))/del_phi)
                
                v_grady = v_temp if len(v_grady.shape)==0 else xr.concat([v_grady, v_temp], dim='date')

                w_temp = w_down.sel(strftime=name)*\
            grad_z(grp, z_diff.sel(date=z_diff.date.dt.strftime('%d-%b') == name))

#             ((grp.shift(isobaricInhPa=-1) - grp)/\
#                             (ds_bflx['z_diff'].sel(date=ds_bflx.date.dt.strftime('%d-%b') == name)/g))
                w_gradz = w_temp if len(w_gradz.shape)==0 else xr.concat([w_gradz, w_temp], dim='date')
                u_gradx = u_gradx.sortby('date'); v_grady = v_grady.sortby('date'); w_gradz = w_gradz.sortby('date')

    u_gradx.name = 'u_gradx'
    v_grady.name = 'v_grady'
    w_gradz.name = 'w_gradz'
    return u_gradx, v_grady, w_gradz
# -


