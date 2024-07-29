import xarray as xr
import numpy as np
import requests
import matplotlib.pyplot as plt
from metpy.units import units
import metpy.calc as mpcalc
import pickle


def calculate_tsi(dataset):
    '''
    Calculate the Thunderstorm Index (TSI) based on the TTI, KI, and the relative humidity
    :param dataset: xarray dataset with temperature and relative humidity
    :return: xarray dataset with the TSI
    '''

    # Create a grid of zeros
    grid = np.zeros((dataset.sizes['latitude'], dataset.sizes['longitude']))
    dataset = dataset.metpy.quantify()

    # Calculate Dewpoint
    dew = mpcalc.dewpoint_from_relative_humidity(dataset['t'], dataset['r']).metpy.dequantify()
    dew_850 = dew.sel(isobaricInhPa=850)
    dew_700 = dew.sel(isobaricInhPa=700)

    temp_850 = dataset['t'].sel(isobaricInhPa=850).metpy.convert_units('degC').metpy.dequantify()
    temp_700 = dataset['t'].sel(isobaricInhPa=700).metpy.convert_units('degC').metpy.dequantify()
    temp_500 = dataset['t'].sel(isobaricInhPa=500).metpy.convert_units('degC').metpy.dequantify()
    r_850 = dataset['r'].sel(isobaricInhPa=850).metpy.dequantify()
    r_700 = dataset['r'].sel(isobaricInhPa=700).metpy.dequantify()
    r_500 = dataset['r'].sel(isobaricInhPa=500).metpy.dequantify()

    TTI = temp_850 + dew_850 - (2 * temp_500)
    KI = (temp_850 - temp_500) + dew_850 - (temp_700 - dew_700)

    grid[(r_500 > 90) & (TTI > 44) & (KI > 25)] = 1

    tsi = xr.DataArray(grid, coords=[dataset['latitude'], dataset['longitude']], dims=['latitude', 'longitude'])
    return tsi


def calculate_ww(tp, vis, tcc, tsi):
    '''
    Calculate the Weather Category (WW) based on the TP, VIS, TCC, and TSI
    :param tp: xarray dataset with total precipitation
    :param vis: xarray dataset with visibility
    :param tcc: xarray dataset with total cloud cover
    :param tsi: xarray dataset with thunderstorm index
    :return: xarray dataset with the WW
    '''
    # Ensure all inputs are DataArrays
    tp = (tp.to_array() if isinstance(tp, xr.Dataset) else tp).squeeze()
    vis = (vis.to_array() if isinstance(vis, xr.Dataset) else vis).squeeze()
    tcc = (tcc.to_array() if isinstance(tcc, xr.Dataset) else tcc).squeeze()
    tsi = (tsi.to_array() if isinstance(tsi, xr.Dataset) else tsi).squeeze()

    # Create a grid of zeros
    grid = np.zeros((tp.sizes['latitude'], tp.sizes['longitude']))

    grid[(tsi == 1) & (tp <= 1)] = 17
    grid[(tsi == 1) & (tp > 1)] = 95
    grid[(tsi == 0) & (tp > 10)] = 65
    grid[(tsi == 0) & (tp > 5) & (tp <= 10)] = 63
    grid[(tsi == 0) & (tp > 1) & (tp <= 5)] = 61
    grid[(tsi == 0) & (tp <= 1) & (vis < 1000)] = 45
    grid[(tsi == 0) & (tp <= 1) & (vis <= 5000) & (vis >= 1000)] = 10
    grid[(tsi == 0) & (tp <= 1) & (vis > 5000) & (tcc < 10)] = 0
    grid[(tsi == 0) & (tp <= 1) & (vis > 5000) & (tcc < 60) & (tcc >= 10)] = 1
    grid[(tsi == 0) & (tp <= 1) & (vis > 5000) & (tcc < 90) & (tcc >= 60)] = 2
    grid[(tsi == 0) & (tp <= 1) & (vis > 5000) & (tcc > 90)] = 3

    ww = xr.DataArray(grid, coords=[tp['latitude'], tp['longitude']], dims=['latitude', 'longitude'])
    return ww

# Load the data
filename = 'sample/gfs.t12z.pgrb2.0p25.f003_all_R20240726120000_20240726150000_20240726150000.grib2'

# Domain slice
lats = slice (-13, 9)
lons = slice (90, 143)

# Load lats lons from the harmonization
with open('D:\Projects\df-auto\src\df_latitudes.pkl', 'rb') as f:
    df_lats = pickle.load(f)

with open('D:\Projects\df-auto\src\df_longitudes.pkl', 'rb') as f:
    df_lons = pickle.load(f)

# Load data for TSI
ds_tsi = xr.load_dataset(filename, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})[['t','r']].sel(latitude=lats, longitude=lons).interp(latitude=df_lats, longitude=df_lons, method='linear')

# Load data for TP
ds_tp = xr.load_dataset(filename, engine='cfgrib', backend_kwargs={'filter_by_keys': {'stepType': 'accum','typeOfLevel': 'surface','shortName':'tp'}}).sel(latitude=lats, longitude=lons).interp(latitude=df_lats, longitude=df_lons, method='linear')

# Load data for Vis
ds_vis = xr.load_dataset(filename, engine='cfgrib', backend_kwargs={'filter_by_keys': {'stepType': 'instant','typeOfLevel': 'surface','shortName':'vis'}}).sel(latitude=lats, longitude=lons).interp(latitude=df_lats, longitude=df_lons, method='linear')

#Total Cloud Cover
ds_tcc = xr.load_dataset(filename, engine='cfgrib', backend_kwargs={'filter_by_keys': {'stepType': 'instant','typeOfLevel': 'atmosphere', 'shortName':'tcc'}}).sel(latitude=lats, longitude=lons).interp(latitude=df_lats, longitude=df_lons, method='linear')

tsi = calculate_tsi(ds_tsi)
ww = calculate_ww(ds_tp, ds_vis, ds_tcc, tsi)

# Load additional data
ds_surface = xr.load_dataset(filename, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround','level':2.0}}).sel(latitude=lats, longitude=lons).interp(latitude=df_lats, longitude=df_lons, method='linear')
ds_wind = xr.load_dataset(filename, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround','level':10.0}}).sel(latitude=lats, longitude=lons).interp(latitude=df_lats, longitude=df_lons, method='linear')

coords = ds_tp.coords

# Create dataset
ds_fc = xr.Dataset(
    {
        'tp': (['latitude','longitude'],ds_tp['tp'].values),
        'tsi': (['latitude','longitude'],tsi.values),
        't2m': (['latitude','longitude'], ds_surface['t2m'].values),
        'tcc': (['latitude','longitude'], ds_tcc['tcc'].values),
        'vis': (['latitude','longitude'], ds_vis['vis'].values),
        'r2': (['latitude','longitude'], ds_surface['r2'].values),
    },
    coords=coords,
    attrs={
        'GRIB_edition': 2,
        'GRIB_centre': 'wiix',
        'GRIB_centreDescription': 'Indonesia Meteorological Climatological and Geophysical Agency - BMKG',
        'GRIB_subCentre': 0,
        'Conventions': 'CF-1.7',
        'institution': 'Indonesia Meteorological Climatological and Geophysical Agency - BMKG',
    }
)

ds_fc_wind = xr.Dataset(
    {
        'u10': (['latitude','longitude'],ds_wind['u10'].values),
        'v10': (['latitude','longitude'],ds_wind['v10'].values),
    },
    coords=coords,
    attrs={
        'GRIB_edition': 2,
        'GRIB_centre': 'wiix',
        'GRIB_centreDescription': 'Indonesia Meteorological Climatological and Geophysical Agency - BMKG',
        'GRIB_subCentre': 0,
        'Conventions': 'CF-1.7',
        'institution': 'Indonesia Meteorological Climatological and Geophysical Agency - BMKG',
    }
)

ds_af = xr.Dataset(
    {
        'ww': (['latitude','longitude'],ww.values),
    },
    coords=coords,
    attrs={
        'GRIB_edition': 2,
        'GRIB_centre': 'wiix',
        'GRIB_centreDescription': 'Indonesia Meteorological Climatological and Geophysical Agency - BMKG',
        'GRIB_subCentre': 0,
        'Conventions': 'CF-1.7',
        'institution': 'Indonesia Meteorological Climatological and Geophysical Agency - BMKG',
    }
)

# Save the dataset
ds_complete = xr.merge([ds_fc, ds_fc_wind, ds_af],compat='override')
ds_complete.to_netcdf('sample/df_auto_gfs_test.nc')