from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import requests
import matplotlib.pyplot as plt
from metpy.units import units
import metpy.calc as mpcalc
import pickle
import re
import os
import eccodes

MODEL_DIR = 'D:/Data/sample_ifs'
DF_LATITUDES = 'D:/Projects/df-auto/src/df_latitudes.pkl'
DF_LONGITUDES = 'D:/Projects/df-auto/src/df_longitudes.pkl'
OUTPUT_DIR = 'D:/Projects/df-auto/output'

FILE_PATTERN_DICT = {
        'ECMWF__0.1': r'A1D{init_date:%m%d%H00}[0-9]+_R{init_date:%Y%m%d%H0000}_{valid_date:%Y%m%d%H0000}_{valid_date:%Y%m%d%H0000}.grib',
        'AROME_SUS__0.025': r'arome_indo_{init_date:%Y%m%d}_{init_date:%H}00_{step:02d}.grib',
    }

FILE_PATH_DICT = {
        'ECMWF__0.1': 'D:/Data/sample_ifs',
        'AROME_SUS__0.025': 'D:/Data/sample_arome'
    }


def calculate_tsi(dataset):
    '''
    Calculate the Thunderstorm Index (TSI) based on the TTI, KI, and the relative humidity
    :param dataset: xarray dataset with temperature and relative humidity
    :return: xarray dataset with the TSI

    # TO DO: ECMWF has its own precalculated TTI and KI, try to use them and not recalculate
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


def make_datelist(init_date, step, days):
    '''
    Create a list of datetime objects for the forecast
    :param init_date: initial datetime object
    :param step: timedelta object
    :param days: number of days to forecast
    :return: list of datetime objects
    '''
    date_list = [init_date + timedelta(hours=x) for x in range(0, 24*days, step)]
    return date_list


def find_file(model, init_date, valid_date):
    '''
    Find the file for the given model, initial date, and step
    :param model: model name
    :param init_date: initial datetime object
    :param valid_date: step in hours
    :return: filename
    '''

    if model == 'ECMWF__0.1':
        filepath = FILE_PATH_DICT.get(model)
        file_pattern_str = FILE_PATTERN_DICT[model].format(
            init_date=init_date,
            valid_date=valid_date
        )

    else:
        return 'Model not found'

    # Compile the regex pattern
    file_pattern = re.compile(file_pattern_str)

    for filename in os.listdir(filepath):
        if file_pattern.match(filename):
            return os.path.join(filepath, filename)
    return None


def load_data(model, filename1, filename2=None):
    '''
    Load data and interpolate to digital forecast grid

    :param model:
    :param filename:
    :return:
    '''
    # Load DF lats and lons
    with open(DF_LATITUDES, 'rb') as f:
        df_lats = pickle.load(f)

    with open(DF_LONGITUDES, 'rb') as f:
        df_lons = pickle.load(f)

    # Domain slice
    lats = slice(9, -13)
    lons = slice(90, 143)

    if model == 'ECMWF__0.1':
        # Load data for TSI
        ds_tsi = xr.load_dataset(filename1, engine='cfgrib',
                                 backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})[['t', 'r']].sel(
            latitude=lats, longitude=lons).interp(latitude=df_lats, longitude=df_lons, method='linear')
        ds_tsi = calculate_tsi(ds_tsi)

        # Load data for Vis
        ds_vis = xr.load_dataset(filename1, engine='cfgrib', backend_kwargs={
            'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'surface', 'shortName': 'vis'}}).sel(latitude=lats,
                                                                                                          longitude=lons).interp(
            latitude=df_lats, longitude=df_lons, method='linear')['vis']

        # Load data for TCC
        ds_tcc = xr.load_dataset(filename1, engine='cfgrib', backend_kwargs={
            'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'surface', 'shortName': 'tcc'}}).sel(latitude=lats,
                                                                                                          longitude=lons).interp(
            latitude=df_lats, longitude=df_lons, method='linear')['tcc'] * 100

        # Load data for TP
        if filename2:
            tp1 = xr.load_dataset(filename1, engine='cfgrib',
                                  backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'edition': 1}}).sel(
                latitude=lats, longitude=lons).interp(latitude=df_lats, longitude=df_lons, method='linear')['tp'] * 1000
            tp2 = xr.load_dataset(filename2, engine='cfgrib',
                                  backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'edition': 1}}).sel(
                latitude=lats, longitude=lons).interp(latitude=df_lats, longitude=df_lons, method='linear')['tp'] * 1000
            ds_tp = tp1 - tp2
            ds_tp = ds_tp.where(ds_tp > 0, 0)
        else:
            ds_tp = xr.load_dataset(filename1, engine='cfgrib',
                                    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'edition': 1}}).sel(
                latitude=lats, longitude=lons).interp(latitude=df_lats, longitude=df_lons, method='linear')['tp'] * 1000


        # Load additional data
        ds_wind = xr.load_dataset(filename1, engine='cfgrib',
                                  backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'edition': 1}})[
            ['u10', 'v10']].sel(latitude=lats, longitude=lons).interp(latitude=df_lats, longitude=df_lons,
                                                                      method='linear')  # Load additional data
        ds_surface = xr.load_dataset(filename1, engine='cfgrib',
                                     backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'edition': 1}})[
            ['t2m', 'd2m']].sel(latitude=lats, longitude=lons).interp(latitude=df_lats, longitude=df_lons,
                                                                      method='linear')
        ds_t2 = ds_surface['t2m']
        ds_r2 = mpcalc.relative_humidity_from_dewpoint(ds_surface['t2m'] * units.kelvin,
                                                       ds_surface['d2m'] * units.kelvin).metpy.dequantify() * 100

        return ds_tsi, ds_vis, ds_tcc, ds_tp, ds_t2, ds_r2, ds_wind
    else:
        return 'Model not found'


def main():
    init_time = datetime(2024, 7, 31, 12)
    forecast_hour = 8
    step = 1
    model = 'ECMWF__0.1'

    for fh in range(1,forecast_hour,step):
        valid_time = init_time + timedelta(hours=fh)
        file1 = find_file(model, init_time, valid_time)
        file2 = find_file(model, init_time, valid_time - timedelta(hours=1))
        if fh == step:
            ds_tsi, ds_vis, ds_tcc, ds_tp, ds_t2, ds_r2, ds_wind = load_data(model, file1)
        else:
            ds_tsi, ds_vis, ds_tcc, ds_tp, ds_t2, ds_r2, ds_wind = load_data(model, file1, file2)

        ww = calculate_ww(ds_tp, ds_vis, ds_tcc, ds_tsi)

        # Some parameters for the grib file
        points = ww.count().item()
        Nx = ww.sizes['longitude']
        Ny = ww.sizes['latitude']
        lat0, lon0 = ww['latitude'].values[0], ww['longitude'].values[0]
        lat1, lon1 = ww['latitude'].values[-1], ww['longitude'].values[-1]
        dataDate = int(init_time.strftime('%Y%m%d'))
        dataTime = int(init_time.strftime('%H%M'))

        grib_list = [{
            'edition': 2,
            'centre': '195',
            'dataDate': dataDate,
            'dataTime': dataTime,
            'stepType': 'instant',
            'stepUnits': 'h',
            'step': fh,
            'typeOfLevel': 'heightAboveGround',
            'level': 0.0,  # Surface level is typically level 0
            'gridType': 'regular_ll',
            'Nx': Nx,
            'Ny': Ny,
            'numberOfPoints': points,
            'latitudeOfFirstGridPointInDegrees': lat0,
            'longitudeOfFirstGridPointInDegrees': lon0,
            'latitudeOfLastGridPointInDegrees': lat1,
            'longitudeOfLastGridPointInDegrees': lon1,
            'iDirectionIncrementInDegrees': 0.025,
            'jDirectionIncrementInDegrees': 0.025,
            'dataType': 'af',
            'discipline': 0,
            'parameterCategory': 19,
            'parameterNumber': 25,
            'typeOfFirstFixedSurface': 1,
            'values': ww.values.flatten()},
            {
                'edition': 2,
                'centre': '195',
                'dataDate': dataDate,
                'dataTime': dataTime,
                'stepType': 'accum',
                'stepUnits': 'h',
                'step': fh,
                'typeOfLevel': 'surface',
                'level': 0.0,  # Surface level is typically level 0
                'gridType': 'regular_ll',
                'Nx': Nx,
                'Ny': Ny,
                'numberOfPoints': points,
                'latitudeOfFirstGridPointInDegrees': lat0,
                'longitudeOfFirstGridPointInDegrees': lon0,
                'latitudeOfLastGridPointInDegrees': lat1,
                'longitudeOfLastGridPointInDegrees': lon1,
                'iDirectionIncrementInDegrees': 0.025,
                'jDirectionIncrementInDegrees': 0.025,
                'dataType': 'fc',
                'localTablesVersion': 1,
                'discipline': 0,
                'parameterCategory': 1,
                'parameterNumber': 193,
                'typeOfFirstFixedSurface': 1,
                'typeOfStatisticalProcessing': 1,
                'values': ds_tp.values.flatten()
            },
            {
                'edition': 2,
                'centre': '195',
                'dataDate': dataDate,
                'dataTime': dataTime,
                'stepType': 'instant',
                'stepUnits': 'h',
                'step': fh,
                'typeOfLevel': 'surface',
                'level': 0.0,  # Surface level is typically level 0
                'gridType': 'regular_ll',
                'Nx': Nx,
                'Ny': Ny,
                'numberOfPoints': points,
                'latitudeOfFirstGridPointInDegrees': lat0,
                'longitudeOfFirstGridPointInDegrees': lon0,
                'latitudeOfLastGridPointInDegrees': lat1,
                'longitudeOfLastGridPointInDegrees': lon1,
                'iDirectionIncrementInDegrees': 0.025,
                'jDirectionIncrementInDegrees': 0.025,
                'dataType': 'fc',
                'discipline': 0,
                'parameterCategory': 19,
                'parameterNumber': 203,
                'values': ds_tsi.values.flatten()
            },
            {
                'edition': 2,
                'centre': '195',
                'dataDate': dataDate,
                'dataTime': dataTime,
                'stepType': 'instant',
                'stepUnits': 'h',
                'step': fh,
                'typeOfLevel': 'surface',
                'level': 0.0,  # Surface level is typically level 0
                'gridType': 'regular_ll',
                'Nx': Nx,
                'Ny': Ny,
                'numberOfPoints': points,
                'latitudeOfFirstGridPointInDegrees': lat0,
                'longitudeOfFirstGridPointInDegrees': lon0,
                'latitudeOfLastGridPointInDegrees': lat1,
                'longitudeOfLastGridPointInDegrees': lon1,
                'iDirectionIncrementInDegrees': 0.025,
                'jDirectionIncrementInDegrees': 0.025,
                'dataType': 'fc',
                'discipline': 0,
                'parameterCategory': 0,
                'parameterNumber': 0,
                'typeOfFirstFixedSurface': 103,
                'scaledValueOfFirstFixedSurface': 2,
                'values': ds_t2.values.flatten()
            },
            {
                'edition': 2,
                'centre': '195',
                'dataDate': dataDate,
                'dataTime': dataTime,
                'stepType': 'instant',
                'stepUnits': 'h',
                'step': fh,
                'typeOfLevel': 'surface',
                'level': 0.0,  # Surface level is typically level 0
                'gridType': 'regular_ll',
                'Nx': Nx,
                'Ny': Ny,
                'numberOfPoints': points,
                'latitudeOfFirstGridPointInDegrees': lat0,
                'longitudeOfFirstGridPointInDegrees': lon0,
                'latitudeOfLastGridPointInDegrees': lat1,
                'longitudeOfLastGridPointInDegrees': lon1,
                'iDirectionIncrementInDegrees': 0.025,
                'jDirectionIncrementInDegrees': 0.025,
                'dataType': 'fc',
                'localTablesVersion': 1,
                'discipline': 0,
                'parameterCategory': 6,
                'parameterNumber': 192,
                'typeOfFirstFixedSurface': 1,
                'typeOfSecondFixedSurface': 8,
                'values': ds_tcc.values.flatten()
            },
            {
                'edition': 2,
                'centre': '195',
                'dataDate': dataDate,
                'dataTime': dataTime,
                'stepType': 'instant',
                'stepUnits': 'h',
                'step': fh,
                'typeOfLevel': 'surface',
                'level': 0.0,  # Surface level is typically level 0
                'gridType': 'regular_ll',
                'Nx': Nx,
                'Ny': Ny,
                'numberOfPoints': points,
                'latitudeOfFirstGridPointInDegrees': lat0,
                'longitudeOfFirstGridPointInDegrees': lon0,
                'latitudeOfLastGridPointInDegrees': lat1,
                'longitudeOfLastGridPointInDegrees': lon1,
                'iDirectionIncrementInDegrees': 0.025,
                'jDirectionIncrementInDegrees': 0.025,
                'dataType': 'fc',
                'discipline': 0,
                'parameterCategory': 19,
                'parameterNumber': 0,
                'values': ds_vis.values.flatten()
            },
            {
                'edition': 2,
                'centre': '195',
                'dataDate': dataDate,
                'dataTime': dataTime,
                'stepType': 'instant',
                'stepUnits': 'h',
                'step': fh,
                'typeOfLevel': 'surface',
                'level': 0.0,  # Surface level is typically level 0
                'gridType': 'regular_ll',
                'Nx': Nx,
                'Ny': Ny,
                'numberOfPoints': points,
                'latitudeOfFirstGridPointInDegrees': lat0,
                'longitudeOfFirstGridPointInDegrees': lon0,
                'latitudeOfLastGridPointInDegrees': lat1,
                'longitudeOfLastGridPointInDegrees': lon1,
                'iDirectionIncrementInDegrees': 0.025,
                'jDirectionIncrementInDegrees': 0.025,
                'dataType': 'fc',
                'localTablesVersion': 1,
                'discipline': 0,
                'parameterCategory': 6,
                'parameterNumber': 192,
                'typeOfFirstFixedSurface': 1,
                'typeOfSecondFixedSurface': 8,
                'values': ds_tcc.values.flatten()
            },
            {
                'edition': 2,
                'centre': '195',
                'dataDate': dataDate,
                'dataTime': dataTime,
                'stepType': 'instant',
                'stepUnits': 'h',
                'step': fh,
                'typeOfLevel': 'surface',
                'level': 0.0,  # Surface level is typically level 0
                'gridType': 'regular_ll',
                'Nx': Nx,
                'Ny': Ny,
                'numberOfPoints': points,
                'latitudeOfFirstGridPointInDegrees': lat0,
                'longitudeOfFirstGridPointInDegrees': lon0,
                'latitudeOfLastGridPointInDegrees': lat1,
                'longitudeOfLastGridPointInDegrees': lon1,
                'iDirectionIncrementInDegrees': 0.025,
                'jDirectionIncrementInDegrees': 0.025,
                'dataType': 'fc',
                'discipline': 0,
                'parameterCategory': 1,
                'parameterNumber': 1,
                'typeOfFirstFixedSurface': 103,
                'scaledValueOfFirstFixedSurface': 2,
                'scaleFactorOfFirstFixedSurface': 0,
                'values': ds_r2.values.flatten()
            },
            {
                'edition': 2,
                'centre': '195',
                'dataDate': dataDate,
                'dataTime': dataTime,
                'stepType': 'instant',
                'stepUnits': 'h',
                'step': fh,
                'typeOfLevel': 'heightAboveGround',
                'level': 10.0,  # Surface level is typically level 0
                'gridType': 'regular_ll',
                'Nx': Nx,
                'Ny': Ny,
                'numberOfPoints': points,
                'latitudeOfFirstGridPointInDegrees': lat0,
                'longitudeOfFirstGridPointInDegrees': lon0,
                'latitudeOfLastGridPointInDegrees': lat1,
                'longitudeOfLastGridPointInDegrees': lon1,
                'iDirectionIncrementInDegrees': 0.025,
                'jDirectionIncrementInDegrees': 0.025,
                'dataType': 'fc',
                'discipline': 0,
                'parameterCategory': 2,
                'parameterNumber': 3,
                'typeOfFirstFixedSurface': 103,
                'scaledValueOfFirstFixedSurface': 10,
                'scaleFactorOfFirstFixedSurface': 0,
                'values': ds_wind['v10'].values.flatten()
            },
            {
                'edition': 2,
                'centre': '195',
                'dataDate': dataDate,
                'dataTime': dataTime,
                'stepType': 'instant',
                'stepUnits': 'h',
                'step': fh,
                'typeOfLevel': 'heightAboveGround',
                'level': 10.0,  # Surface level is typically level 0
                'gridType': 'regular_ll',
                'Nx': Nx,
                'Ny': Ny,
                'numberOfPoints': points,
                'latitudeOfFirstGridPointInDegrees': lat0,
                'longitudeOfFirstGridPointInDegrees': lon0,
                'latitudeOfLastGridPointInDegrees': lat1,
                'longitudeOfLastGridPointInDegrees': lon1,
                'iDirectionIncrementInDegrees': 0.025,
                'jDirectionIncrementInDegrees': 0.025,
                'dataType': 'fc',
                'discipline': 0,
                'parameterCategory': 2,
                'parameterNumber': 2,
                'typeOfFirstFixedSurface': 103,
                'scaledValueOfFirstFixedSurface': 10,
                'scaleFactorOfFirstFixedSurface': 0,
                'values': ds_wind['u10'].values.flatten()
            }
        ]

        with open(f'{OUTPUT_DIR}/df_auto_ecmwf_{fh}.grib', 'wb') as f:
            for data in grib_list:
                gid = eccodes.codes_grib_new_from_samples('regular_ll_pl_grib2')
                for key, value in data.items():
                    # print(key, value)
                    if key == 'values':
                        # print('setting values')
                        eccodes.codes_set_values(gid, value)
                    else:
                        # print('setting key')
                        eccodes.codes_set(gid, key, value)
                eccodes.codes_write(gid, f)
                eccodes.codes_release(gid)


if __name__ == '__main__':
    main()