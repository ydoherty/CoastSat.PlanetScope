# CoastSat for PlanetScope Dove Imagery

# load coastsat modules
import os
from coastsat_ps.data_import import initialise_settings
from coastsat_ps.extract_shoreline import extract_shorelines, compute_intersection
from coastsat_ps.interactive import filter_shorelines                    
from coastsat_ps.preprocess import (data_extract, pre_process, select_ref_image, 
                                    add_ref_features)
from coastsat_ps.postprocess import tidal_correction, ts_plot_single
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('default')


#%% 0) User Input Settings 

settings = {
    
    ### General Settings ###
    # Site name (for output folder and files) 
    'site_name': 'YAGON',
    # Maximum image cloud cover percentage threshold
    'cloud_threshold': 10, # Default 10
    # Minimum image AOI cover percentage threshold
    'extent_thresh': 1, # Default 80
    # Desired output shoreline epsg
    'output_epsg': '28356',
    
    
    ### Reference files (in "...CoastSat.PlanetScope/user_inputs/") ###
    # Area of interest file (save as .kml file from geojson.io website)
    'aoi_kml': 'YAGON_polygon.kml',
    # Transects in geojson file (ensure same epsg as output_epsg)
    'transects': 'YAGON_transects.geojson', # False
        # If False boolean given, popup window will allow for manual drawing of transects
    # Tide csv file in MSL and UTC 
    'tide_data': 'YAGON_tides.csv',
    # Local folder planet imagery downloads location (provide full folder path)
    'downloads_folder': os.path.join(os.getcwd(),'images_Yagon',
                                      'Yagon_ChrisD_PSScene4Band_QGIS',
                                      'files'),
    # 'downloads_folder': os.path.join(os.getcwd(),'images_Yagon','test'),

    ### Processing settings ###
    # Machine learning classifier filename (in "...CoastSat.PlanetScope/classifier/models")
        # A new classifier may be re-trained after step 1.3. Refer "...CoastSat.PlanetScope/classifier/train_new_classifier.py" for instructions. 
    'classifier': 'NN_4classes_PS_NARRA.pkl',
    # Image co-registration choice ['Coreg Off', 'Local Coreg', 'Global Coreg']
    'im_coreg': 'Coreg Off', # refer https://pypi.org/project/arosics/ for details on Local vs Global coreg. Local recommended but slower. 


    ### Advanced settings ###
    # Buffer size around masked cloud pixels [in metres]
    'cloud_buffer': 3, # Default 9 (3 pixels)  
    # Max distance from reference shoreline for valid shoreline [in metres]
    'max_dist_ref': 100, # Default 75
    # Minimum area (m^2) for an object to be labelled as a beach
    'min_beach_area': 150*150, # Default 22500
    # Minimum length for identified contour line to be saved as a shoreline [in metres]
    'min_length_sl': 500, # Default 500 
    # GDAL location setting (Update 'anaconda2' to 'anaconda3' depending on installation version. Update 'coastsat_ps' to chosen environment name)
    'GDAL_location': r'C:\ProgramData\Anaconda3\envs\coastsat_ps\Library\bin',
    
    #### Additional advanced Settings can be found in "...CoastSat.PlanetScope/coastsat_ps/data_import.py"
    
    }


# Import data and updade settings based on user input
outputs = initialise_settings(settings)


#%% 1.1) Pre-processing - TOA conversion and mask extraction

data_extract(settings, outputs)


#%% 1.2) Pre-processing - Select reference image for co-registration

select_ref_image(settings)
  

#%% 1.3) Pre-Processing - image coregistration and scene merging

pre_process(settings, outputs, 
        # del_files_int = True will delete intermediate coregistration files to save space
        del_files_int = True)

# Note "Failed to delete GEOS geom" error message in console during 
    # co-registraion does not impact algorithm. Working on bug fix. 


#%% 2.1) Select georectified/merged image for classification, reference shoreline and transects

add_ref_features(settings)


#%% 2.2) Extract shoreline data
    
# Note that output shoreline .geojson file for use in GIS software is not todally corrected
    
shoreline_data = extract_shorelines(outputs, settings, 
                                    
        # del_index = True will delete water index .tif files once used to save space
        del_index = False, 
        
        # reclassify = True will reclassify images if they have been classified previously
            # useful when running again with a new classifier
            # use False to save time on re-runs with the same classifier to save processing time
        reclassify = False)
 

#%%
# plot the mapped shorelines
plt.ion()
fig = plt.figure(figsize=[15,8], tight_layout=True)
plt.axis('equal')
plt.xlabel('Eastings')
plt.ylabel('Northings')
plt.grid(linestyle=':', color='0.5')
for i in range(len(shoreline_data['shorelines'])):
    sl = shoreline_data['shorelines'][i]
    date = shoreline_data['timestamp utc'][i]
    plt.plot(sl[:,0], sl[:,1], '.', label=date.strftime('%d-%m-%Y'))
plt.legend()   

#%% 3) Manual error detection

    # Option 1:
        # manual_filter & load_csv = False
            # All images pass, creates a csv in the outputs folder
                # "...CoastSat.PlanetScope/outputs/SITE/shoreline outputs/COREG/NmB/Peak Fraction/shoreline_filter.csv"

    # Option 2:
        # manual_filter = True & load_csv = False    
            # popup window to keep or discard images (saves choices as a csv)

    # Option 3:
        # manual_filter = False & load_csv = True
            # loads and applies the csv saved from option 1 or 2
            # This file can be manually updated if desired with a text editor
         
shoreline_data = filter_shorelines(settings,
        manual_filter = True, load_csv = False)


#%% average images taken on the same day



#%% 4) Shoreline transect intersction and csv export

settings['max_std'] = 20
settings['max_range'] = 40
settings['min no. intercepts'] = 3 
sl_csv = compute_intersection(shoreline_data, settings)


#%% 5) Tidal Correction & filtering

tide_settings = {
    # select beach slope as a generic value, or list of values corresponding to each transect
        # Transect specific beach slope values can be extracted with the CoastSat beach slope tool https://github.com/kvos/CoastSat.slope
    'beach_slope': 0.07, #0.1
    
    # Reference elevation contour
    'contour': 0,
    # Tidal correction weighting
    'weighting': 1,
    # Offset correction (+ve value corrects sl seaward, ie. increases chainage)
    'offset': 0,
    
    # Date filter (minimum)
    'date_min':'2016-01-01',
    # Date filter (maximum)
    'date_max':'2022-01-01' 
    }

sl_csv_tide = tidal_correction(settings, tide_settings, sl_csv)


#%% 6) Plot transects
    
for transect in settings['transects_load'].keys():
    ts_plot_single(settings, sl_csv_tide, transect, 
                   
        # set savgol = True to plot 15 day moving average shoreline position
        # Requires > 15 day shorleine timeseries range
        savgol = True,
        
        # set x_scale for x-axis labels ['days', 'months', 'years']
        x_scale = 'years')
    
#%% 
  

import seaborn as sns
sns.set_theme()
plt.ion()

dates = list(sl_csv_tide['Date'])
keys = [_ for _ in sl_csv_tide.columns if 'aus' in _]
for key in keys:
    fig,ax = plt.subplots(1,1,figsize=(15,4),tight_layout=True)
    chainages = sl_csv_tide[key] 
    chainages = chainages - np.nanmean(chainages)
    idx_nan = np.isnan(chainages)
    dates2 = [dates[_] for _ in np.where(~idx_nan)[0]]
    chainages = chainages[~idx_nan]
    ax.plot(dates2,chainages,'o-',mfc='w')
    ax.set(title=key,ylabel='cross-shore change [m]')
        
    fig.savefig(os.path.join(os.getcwd(),'figs','%s.jpg'%key),dpi=250)
    plt.close(fig)

#%% Approximate times (for ~1000 downloaded images)
    # 1.1) 20min
    # 1.3) 2.5h coregistration, 35min merge
    # 2.2) 1h classification, 50min shoreline extraction


                
                
            

            

