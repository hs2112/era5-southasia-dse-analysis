# +
# a=xr.open_dataset('/home/data/lab_project1/daily_means/potential_vorticity/era5_potential_vorticity_2021_3_1.nc')
# a['pv'][0,0,0,0].compute()
# a.close()
# -

import cdsapi, os, sys, calendar, xarray as xr, pandas as pd, numpy as np, dask
from datetime import date,datetime,timedelta


n = len(sys.argv)
print("Total arguments passed:", n)
print("Name of Python script:", sys.argv[0])
print("Arguments passed:", [s for s in sys.argv])


# var = 'geopotential'
year = int(sys.argv[1])
month = int(sys.argv[2])
var = sys.argv[3]

num_days = calendar.monthrange(year, month)[1]

print(var)

dir_hourly_dl = '/home/data/lab_hardik/{}/'.format(var)
os.chdir(dir_hourly_dl)

dir_daily_mean = '/home/data/lab_hardik/data/ERA5/daily_means/{}/data/'.format(var)

def download_era5yrdata(variable,pressure_level,year,month,day,time):
    c = cdsapi.Client()
    c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': variable,
        'pressure_level': pressure_level,
        'year': str(year),
        'month': str(month),
        'day': str(day),
        'time': time,
    },
    'era5_'+ variable+'_'+ str(year)+'_'+ str(month)+'_'+str(day)+'.nc')

def main():
    def preprocess(ds):
        print(ds.time.dt.date.max().data.item())
        ds = ds.groupby('time.date').mean()
        ds['date'] = ds.date.astype('datetime64[ns]')
        return ds

    time = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00', '16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']
    
    pressure_level = ['100', '125', '150', '175', '200', '225', '250', '300', '400', '500', '600', '700', '750', '775', '800', '825', '850', '875', '900', '925','950', '975', '1000']

    for element in list(pd.date_range('{}-{}-01'.format(year, month),'{}-{}-{}'.format(year, month, num_days))):
        if ((element.month==2) and (element.day<25)) or ((element.month==7) and (element.day>5)):
            print('skipping', element)
            continue
        
        file_str = 'era5_'+ var +'_'+ str(element.year)+'_'+ str(element.month)+'_'+str(element.day)+'.nc'
        print(file_str)
        if file_str != file_str: 
            break

        if file_str in os.listdir(dir_daily_mean): 
            print(file_str,'exists in /daily_mean')
            continue
        else:
            # download hourly file if not present, then proceed to write the hourly file to daily
            if file_str not in os.listdir(dir_hourly_dl):
                print(file_str, 'downloading')
                download_era5yrdata(var, pressure_level, element.year, element.month, element.day, time)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("Current Time = ", current_time)
                print(file_str, 'downloaded to', os.getcwd())
                
            print(os.getcwd())

            ds_in = xr.open_mfdataset(file_str,
#                     backend_kwargs={'indexpath':''}, 
                    preprocess=preprocess, 
                    coords='minimal', compat='override', data_vars = 'minimal', combine_attrs = 'override'
                    )
            ds_in.to_netcdf(dir_daily_mean + file_str)
            print('daily mean .nc written')
            ds_in.close()    
            del ds_in
            os.remove(dir_hourly_dl + file_str)
            print('hourly file removed')



if __name__ == "__main__":
    main()




