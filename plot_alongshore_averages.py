# CoastSat for PlanetScope Dove Imagery

import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
import matplotlib.dates as dt
from matplotlib import patches
import copy
from datetime import datetime,timedelta
import pytz
import numpy as np
import pandas as pd
from scipy import interpolate, stats, signal
import pickle
plt.style.use('default')
# import seaborn as sns
# sns.set_theme()
plt.ion()

fp = r'C:\Users\z5030440\Downloads\CoastSat.PlanetScope\outputs\YAGON\shoreline outputs\Coreg Off\NmB\Peak Fraction'
fn = 'YAGON_NmB_Peak Fraction_transect_SL_data_tide_corr.csv'

sl_csv_tide = pd.read_csv(os.path.join(fp,fn))

dates = sl_csv_tide['Date']
dates = [pytz.utc.localize(datetime.strptime(_[:-6],'%Y-%m-%d %H:%M:%S')) for _ in dates]
keys = [_ for _ in sl_csv_tide.columns if 'aus' in _]

with open(os.path.join(os.getcwd(),'longterm_means.pkl'),'rb') as f:
    long_means = pickle.load(f)
    
#%% divide transects in N groups
multiple = 3
groups = []
idx1 = np.floor(len(keys)/multiple).astype(int)
for k in range(multiple):
    if k < multiple-1:
        groups.append(keys[k*idx1:(k+1)*idx1])
    else: 
        groups.append(keys[k*idx1:])
    
# alongshore average
chain_av = dict([])
for i,group in enumerate(groups):
    chain_av['%d'%(i+1)] = []
    for k in range(len(sl_csv_tide)): 
        try:
            chain_av['%d'%(i+1)].append(np.nanmean(sl_csv_tide.iloc[k][group]))
        except:
            chain_av['%d'%(i+1)].append(np.nan)
            
#%% plot
            
            
roll_days = 7
fig,ax = plt.subplots(multiple+1,1,figsize=(9,9),tight_layout=True,sharex=True)
for i in range(len(groups)):
    ax[i].grid(which='major',ls=':',c='k',lw=1)
    ax[i].set(title='Group %d'%(i+1), ylabel='cross-shore [m]',)
    # ax[i].axhline(y=0,ls='--',c='k',lw=1.5)
    chainages = np.array(chain_av['%d'%(i+1)]) 
    # get longterm mean
    means = 0
    for key in groups[i]:
        means += long_means[key]
    long_mean = means / len(groups[i])
    chainages = chainages - long_mean
    idx_nan = np.isnan(chainages)
    dates2 = [dates[_] for _ in np.where(~idx_nan)[0]]
    chainages = chainages[~idx_nan]    
    ax[i].plot(dates2,chainages,'x',c='0.5',ms=6)
    
    # plot savgol
    ps_data = pd.DataFrame({'dates':dates,'chainages':np.array(chain_av['%d'%(i+1)]) })
    ps_data = ps_data.set_index('dates')
    ps_data = ps_data[np.isfinite(ps_data['chainages'])]
    ps_data = ps_data - long_mean
    interp_PL = pd.DataFrame(ps_data.resample('D').mean().interpolate(method = 'linear'))
    interp_PL_sav = signal.savgol_filter(interp_PL[np.isfinite(interp_PL)]['chainages'], roll_days, 2)    
    ax[i].plot(interp_PL.index, interp_PL_sav, linewidth =2, alpha =1, color = 'k', label=str(roll_days)+'-days moving average', zorder = 5)
    
    ymax = ax[i].get_ylim()[-1]
    ymin = ax[i].get_ylim()[0]
    ax[i].axhspan(ymin=0,ymax=ymax,fc='C2',alpha=0.5, label='wider than average')
    ax[i].axhspan(ymin=-20,ymax=0,fc='C1',alpha=0.5, label='narrower than average (0-20m)')
    if ymin < 20:
        ax[i].axhspan(ymin=ymin,ymax=-20,fc='C3',alpha=0.5, label='>20m narrower than average')
    ax[i].set(xlim=[dates2[0]-timedelta(days=7),dates2[-1]+timedelta(days=7)])
    ax[i].set(ylim=[ymin,ymax])

    if i == 0:
        ax[i].legend(loc='lower left',ncol=2)
        ax[i].set_title('Shoreline position relative to long-term mean - Northern Yagon beach')
    elif i == 1:
        ax[i].set_title('Middle Yagon beach')
    elif i == len(groups)-1:
        ax[i].set_title('Southern Yagon beach')
        
# load wave data
df_wave = pd.read_csv(os.path.join(os.getcwd(),'user_inputs','CRHDOW.CrowdyHead.Wave.csv'))
df_wave = df_wave.loc[df_wave['Date']>=sl_csv_tide['Date'][0]]
dates_waves = [pytz.utc.localize(datetime.strptime(_,'%Y-%m-%d %H:%M:%S')) for _ in df_wave['Date']]
hs = np.array(df_wave['Crowdy Head (Hs)'])
hs = np.array(df_wave['Crowdy Head (Wave Height (Hs) AUSWAVE Forecast)'])

ps_data = pd.DataFrame({'dates':dates_waves,'chainages':hs})
ps_data = ps_data.set_index('dates')
ps_data = ps_data[np.isfinite(ps_data['chainages'])]
interp_PL = pd.DataFrame(ps_data.resample('6h').mean().interpolate(method = 'linear'))
wdir = np.array(df_wave['Crowdy Head (Wave Direction AUSWAVE Forecast)'])
ax[-1].set(title='Significant Wave Height at Crowdy Head [AUSWAVE model]',
           ylabel='Hs [m]')
# ax[-1].plot(dates_waves,hs,'C3-') 
ax[-1].plot(interp_PL.index,interp_PL['chainages'],'C3-') 
# ax_twin = ax[-1].twinx()
# ax_twin.grid(False)
# ax_twin.plot(dates_waves,wdir,'C2-')
# date_form = DateFormatter("%d-%b-%y")
# ax[-1].xaxis.set_major_formatter(date_form)
ax[-1].grid(which='major',ls=':',c='k',lw=1)
ax[-1].xaxis.set_major_locator(dt.WeekdayLocator())
ax[-1].xaxis.set_major_formatter(dt.DateFormatter('%d %b'))
ax[-1].xaxis.set_minor_locator(dt.DayLocator())
ax[-1].xaxis.set_minor_formatter(ticker.NullFormatter())
ax[-1].set_xlim([datetime(2021,4,1),datetime(2021,6,15)])

fig.savefig('summary_plot_v5.jpg',dpi=500)
           
#%% individual plots
roll_days = 15
for key in keys:
    # remove nans
    ps_data = copy.deepcopy(sl_csv_tide[['Date',key]])
    ps_data.loc[:,'Date'] = pd.to_datetime(sl_csv_tide.loc[:,'Date'], utc = True)
    ps_data = ps_data.set_index('Date')
    ps_data = ps_data[np.isfinite(ps_data[key])]
    ps_data = ps_data - np.mean(ps_data)
    # savgol plot rolling mean
    interp_PL = pd.DataFrame(ps_data.resample('D').mean().interpolate(method = 'linear'))
    interp_PL_sav = signal.savgol_filter(interp_PL[np.isfinite(interp_PL)][key], roll_days, 2)    
    
    fig,ax = plt.subplots(2,1,figsize=(15,7),tight_layout=True)
    ax[0].axhline(y=0,lw=1,ls='--',c='k')
    ax[0].plot(ps_data,'-o',c='C0',ms=3,lw=1,mfc='w',label='shoreline data')
    ax[0].plot(interp_PL.index, interp_PL_sav, linewidth = 1.5, alpha = 0.7, color = 'r', label=str(roll_days)+'-days moving average', zorder = 5)
    ax[0].set(title='Transect %s'%key,ylabel='cross-shore change [m]')
    ax[0].legend(loc='upper right')
    ax[-1].set(title='Significant Wave Height at Crowdy Head buoy',
               ylabel='Hs [m]')
    ax[-1].plot(dates_waves,hs,'-',c='0.5')     
    
    fig.savefig(os.path.join(os.getcwd(),'figs','%s.jpg'%key),dpi=250)
    plt.close(fig)

                
                
            

            

