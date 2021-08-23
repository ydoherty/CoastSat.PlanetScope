import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import copy
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import os
import pdb
from coastsat_ps.preprocess_tools import create_folder


#%%

def get_closest_datapoint(dates, dates_ts, values_ts):
    """
    Extremely efficient script to get closest data point to a set of dates from a very
    long time-series (e.g., 15-minutes tide data, or hourly wave data)
    
    Make sure that dates and dates_ts are in the same timezone (also aware or naive)
    
    KV WRL 2020

    Arguments:
    -----------
    dates: list of datetimes
        dates at which the closest point from the time-series should be extracted
    dates_ts: list of datetimes
        dates of the long time-series
    values_ts: np.array
        array with the values of the long time-series (tides, waves, etc...)
        
    Returns:    
    -----------
    values: np.array
        values corresponding to the input dates
        
    """
    
    # check if the time-series cover the dates
    if dates[0] < dates_ts[0] or dates[-1] > dates_ts[-1]: 
        raise Exception('Time-series do not cover the range of your input dates')
    
    # get closest point to each date (no interpolation)
    temp = []
    def find(item, lst):
        start = 0
        start = lst.index(item, start)
        return start
    for i,date in enumerate(dates):
        print('\rExtracting closest tide to PS timestamps: %d%%' % int((i+1)*100/len(dates)), end='')
        temp.append(values_ts[find(min(item for item in dates_ts if item > date), dates_ts)])
    values = np.array(temp)
    
    return values


#%% Tidal correction

def tidal_correction(settings, tide_settings, sl_csv):

    # Initialise
    if type(tide_settings['beach_slope']) is list:
        if len(tide_settings['beach_slope']) != len(settings['transects_load'].keys()):
            raise Exception('Beach slope list length does not match number of transects')
    
    # unpack settings
    weight = tide_settings['weighting']
    contour = tide_settings['contour']
    offset = tide_settings['offset']
    mindate = tide_settings['date_min']
    maxdate = tide_settings['date_max']
    
    # import sl data
    sl_csv_tide = copy.deepcopy(sl_csv)
    sl_csv_tide.loc[:,'Date'] = pd.to_datetime(sl_csv_tide.loc[:,'Date'], utc = True)
    
    # Filter by date
    sl_csv_tide = sl_csv_tide[sl_csv_tide['Date'] > pd.to_datetime(mindate, utc = True)]
    sl_csv_tide = sl_csv_tide[sl_csv_tide['Date'] < pd.to_datetime(maxdate, utc = True)]

    # Filter by filter
    sl_csv_tide = sl_csv_tide[sl_csv_tide['Filter'] == 1]

    # Import tide daa
    tide_data = pd.read_csv(os.path.join(settings['user_input_folder'], settings['tide_data']), parse_dates=['dates'])
    dates_ts = [_.to_pydatetime() for _ in tide_data['dates']]
    tides_ts = np.array(tide_data['tides'])
    
    # get tide levels corresponding to the time of image acquisition
    dates_sat = sl_csv_tide['Date'].to_list()
    sl_csv_tide['Tide'] = get_closest_datapoint(dates_sat, dates_ts, tides_ts)
    
    # Perform correction for each transect
    for i, ts in enumerate(settings['transects_load'].keys()):
        # Select beach slope
        if type(tide_settings['beach_slope']) is not list:
            beach_slope = tide_settings['beach_slope']
        else:
            beach_slope = tide_settings['beach_slope'][i]
        
        # Select ts data
        ps_data = copy.deepcopy(sl_csv_tide[['Date',ts, 'Tide']])
        ps_data = ps_data.set_index('Date')
        
        # apply correction
        correction = weight*(ps_data['Tide']-contour)/beach_slope + offset
        sl_csv_tide.loc[:, ts] += correction.values
    
    # save csv
    sl_csv_tide.to_csv(settings['sl_transect_csv'].replace('.csv', '_tide_corr.csv'))
    
    return sl_csv_tide


#%% Single transect plot

def ts_plot_single(settings, sl_csv, transect, savgol, x_scale):
    
    # import PS data and remove nan
    ps_data = copy.deepcopy(sl_csv[['Date',transect]])
    ps_data.loc[:,'Date'] = pd.to_datetime(sl_csv.loc[:,'Date'], utc = True)
    ps_data = ps_data.set_index('Date')
    ps_data = ps_data[np.isfinite(ps_data[transect])]
    mean_ps = np.nanmean(ps_data[transect])
    
    # Initialise figure
    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot(111)
    ax.set_title(settings['site_name'] + ' Transect ' + transect + ' Timeseries Plot')
    ax.set(ylabel='Chainage [m]')
    ax.set(xlabel='Date [UTC]')      
          
    # Mean Position line
    l2 = ax.axhline(y = mean_ps, color='k', linewidth=0.75, label='Mean PS Position', zorder = 2)

    # Number of days
    no_days = (max(ps_data.index)-min(ps_data.index)).days

    #savgol = False
    if savgol == True:
        if no_days < 16:
            raise Exception('SavGol filter requires >15 days in timeseries')
        
        # PS plot
        l1 = ax.fill_between(ps_data.index, ps_data[transect], y2 = mean_ps, alpha = 0.35, color = 'grey', label='PS Data', zorder = 3)
        #l1 = ax.scatter(ps_data.index, ps_data[transect], color = 'k', label='PS Data', marker = 'x', s = 10, linewidth = 0.5, zorder = 1)#, alpha = 0.75)
        l1, = ax.plot(ps_data.index, ps_data[transect], linewidth = 0.75, alpha = 0.4, color = 'k', label='PS Data', zorder = 4)

        # savgol plot rolling mean
        roll_days = 15
        interp_PL = pd.DataFrame(ps_data.resample('D').mean().interpolate(method = 'linear'))
        interp_PL_sav = signal.savgol_filter(interp_PL[np.isfinite(interp_PL)][transect], roll_days, 2)
        l3, = ax.plot(interp_PL.index, interp_PL_sav, linewidth = 0.75, alpha = 0.7, color = 'r', label=str(roll_days) + ' Day SavGol Filter', zorder = 5)
        #l3 = ax.fill_between(interp_PL.index, interp_PL[ts], y2 = mean_GT, alpha = 0.35, color = 'grey', label=str(roll_days) + ' Day SavGol Filter', zorder = 0)
    
        # Set legend
        ax.legend(handles = [l1, l2, l3], ncol = 3, bbox_to_anchor = (0,1), loc='upper left', framealpha = 1, fontsize = 'xx-small')
    else:
        # PS plot
        l1 = ax.fill_between(ps_data.index, ps_data[transect], y2 = mean_ps, alpha = 0.25, color = 'grey', label='PS Data', zorder = 3)
        l1, = ax.plot(ps_data.index, ps_data[transect], linewidth = 0.75, alpha = 0.6, color = 'k', label='PS Data', zorder = 4)
    
        # Set legend
        ax.legend(handles = [l1, l2], ncol = 3, bbox_to_anchor = (0,1), loc='upper left', framealpha = 1, fontsize = 'xx-small')

    
    # Find axis bounds
    if abs(max(ps_data[transect]))-mean_ps > mean_ps-abs(min(ps_data[transect])):
        bound = abs(max(ps_data[transect]))-mean_ps+5
    else:
        bound = mean_ps-abs(min(ps_data[transect]))+5
    
    # Set axis limits
    ax.set_ylim(top = mean_ps + bound)
    ax.set_ylim(bottom = mean_ps - bound)

    ax.set_xlim([min(ps_data.index)-(max(ps_data.index)-min(ps_data.index))/40,
                  max(ps_data.index)+(max(ps_data.index)-min(ps_data.index))/40])

    # Set grid and axis label ticks
    ax.grid(b=True, which='major', linestyle='-')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.tick_params(labelbottom=False, bottom = False)
    ax.tick_params(axis = 'y', which = 'major', labelsize = 6)
        
    if x_scale == 'years':
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        #ax.set_yticklabels([])
        ax.tick_params(labelbottom=True, bottom = True) 
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
    elif x_scale == 'months':
        if no_days > 100:
            raise Exception('Too many dates to render months properly, try x_ticks = years')
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.tick_params(labelbottom=True, bottom = True) 
            ax.xaxis.set_minor_locator(mdates.DayLocator())
    elif x_scale == 'days':
        if no_days > 100:
            raise Exception('Too many dates to render days properly, try x_ticks = years')
        elif no_days > 60:
            raise Exception('Too many dates to render days properly, try x_ticks = months')
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.tick_params(labelbottom=True, bottom = True) 
            ax.xaxis.set_minor_locator(mdates.DayLocator())
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
    else:
        raise Exception('Select either years, months or days for x_scale input')
      

    # save plot
    save_folder = os.path.join(settings['sl_thresh_ind'], 'Timeseries Plots')
    create_folder(save_folder)    
    fig.tight_layout()
    save_file = os.path.join(save_folder, 'transect_' + transect + '_timeseries.png')
    fig.savefig(save_file, dpi=200)

    plt.show()



