# Shoreline extraction functions

import time
import pathlib
import rasterio
import numpy as np
from osgeo import gdal
import os
import pickle
import skimage.morphology as morphology
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import shutil
import pdb
from coastsat_ps.shoreline_tools import (calc_water_index, classify_single, 
                                         create_shoreline_buffer,
                                         process_shoreline, sl_extract, 
                                         sl_extract_generic, output_to_gdf_PL)

from coastsat_ps.interactive import get_ps_data, convert_world2pix, rescale_image_intensity

from coastsat_ps.plotting import (initialise_plot, initialise_plot_gen, rgb_plot, 
                                  index_plot, class_plot, histogram_plot, 
                                  histogram_plot_split)


#%% Overall shoreline extraction function

def extract_shorelines(outputs, settings, del_index = False, reclassify = False):
    # Extract shoreline crop region and calculate water index
    batch_index_and_classify(outputs, settings, reclassify = reclassify)
    index_dict_update(outputs, settings)
    
    # Threshold water index images, extract shorelines and plot
    shorelines = batch_threshold_sl(outputs, settings)
    
    if del_index:
        shutil.rmtree(settings['index_tif_out'])
        # Create blank folder for later runs
        pathlib.Path(settings['index_tif_out']).mkdir(exist_ok=True) 
    
    return shorelines
           
    
#%% Functions

def create_crop_mask(im_classif, im_ref_buffer, settings):
    
    # Extract class masks
    im_sand = im_classif == 1
    im_ww = im_classif == 2
    im_water = im_classif == 3

    # Create mask of non sand or water pixels
    class_mask = ~(im_sand + im_water + im_ww)>0
            
    # Combine all masks
    crop_mask = (class_mask + im_ref_buffer) >0
    
    # Smooth classified edges (thin other removed)
    out_mask = morphology.binary_opening(crop_mask,morphology.square(6)) # perform image opening

    # Add buffer to sl edge
    if settings['thin_beach_fix'] == True:
        crop_mask = crop_mask == 0
        # Adds buffer to sl edge (buffer around sand)
        out_mask = morphology.binary_dilation(crop_mask, morphology.square(5))
        out_mask = morphology.remove_small_objects(out_mask, 
                            min_size=3*9, 
                            connectivity=1)
        out_mask = out_mask == 0

    # Clean up image (small other removed)
    out_mask = morphology.remove_small_objects(out_mask, 
                            min_size=settings['min_beach_area_pixels'], 
                            connectivity=1)
    
    return out_mask, ~im_sand, ~im_water
    

def batch_index_and_classify(outputs, settings, reclassify = False):
    # counters
    number_dates = len(outputs['merged_data'])
    start_time = time.time()
        
    # Create ref mask for generic mask setting
    if settings['generic_sl_region'] == True:
        im_classif = classify_single(settings['classifier_load'], 
                                        settings, 
                                        settings['ref_merge_im'], 
                                        no_mask = False, raw_mask = False)
  
    # Extract shorelines
    for n, dates in enumerate(outputs['merged_data']):
        
        # Print progress
        curr_time = int(time.time() - start_time)
        time_elap = str(int(curr_time/60)) + 'm ' + str(curr_time%60) + 's'
        print('\rClassifying images for date', n+1, 'of',number_dates,'(' + time_elap,' elapsed)', end = '')
        
        for sat in outputs['merged_data'][dates]:
            # extract data/structure
            image_dict = outputs['merged_data'][dates][sat]
            toa_path = image_dict['toa_filepath'][0]
            image_name = image_dict['toa_filename'][0][0:28]
            class_path = toa_path.replace('TOA.tif', 'class.tif')
            
            # Save im_classif if it doesnt exist already
            if (not os.path.isfile(class_path)) or (reclassify == True):
                # Create ref mask for individual mask setting
                if settings['generic_sl_region'] == False:
                    # Classify image 
                    im_classif = classify_single(settings['classifier_load'], 
                                                            settings, 
                                                            toa_path, 
                                                            no_mask = False, raw_mask = False)
                    
                # Copy geo info from TOA file
                with rasterio.open(toa_path, 'r') as src:
                    kwargs = src.meta
                # Update band info/type
                kwargs.update(
                    dtype=rasterio.uint8,
                    count = 1)
                # Save im_classif in TOA folder
                with rasterio.open(class_path, 'w', **kwargs) as dst:
                    dst.write_band(1, im_classif.astype(rasterio.uint8))
                    
            # Calculate water index
            calc_water_index(toa_path, settings, image_name)
                

def index_dict_update(outputs, settings):
    print('\nUpdating output dictionary...')
    # Extract shorelines
    for n, dates in enumerate(outputs['merged_data']):
        for sat in outputs['merged_data'][dates]:
            # extract data/structure
            image_dict = outputs['merged_data'][dates][sat]
            
            # Construct save name
            image_name = image_dict['toa_filename'][0][0:28]
            file_end = settings['water_index_list'][3]
            index_file = os.path.join(settings['index_tif_out'], image_name + file_end)
            
            # Save to dict
            image_dict['index_file'] = index_file


def batch_threshold_sl(outputs, settings):
    
    # Initialise count
    number_dates = len(outputs['merged_data'])
    start_time = time.time()
    plt.close("all")

    # Create save structure
    shorelines_out = {'shorelines':[],
                      'date':[],
                      'timestamp utc':[],
                      'time':[],
                      'name':[],
                      'cloud_cover':[],
                      'aoi_cover':[],
                      'ps_sat_name':[], 
                      'sensor': [],
                      'threshold':[]
                      }
    
    # Extract general data
    image_epsg = int(settings['output_epsg'].replace('EPSG:',''))

    #calculate a buffer around the reference shoreline
    with rasterio.open(settings['ref_merge_im'], 'r') as src:
        im_shape = src.read(1).shape
        image_epsg = int(str(src.crs).replace('EPSG:',''))
    data = gdal.Open(settings['ref_merge_im'], gdal.GA_ReadOnly)
    georef = np.array(data.GetGeoTransform())
    im_ref_buffer = create_shoreline_buffer(im_shape, georef, image_epsg,
                                            settings['pixel_size'], settings)

    # Transects to pic coords    
    transects = {}
    for ts in settings['transects_load']:
        transects[ts] = convert_world2pix(settings['transects_load'][ts], georef)

    # Find generic im_classif from ref_merge_im
    with rasterio.open(settings['ref_merge_im'].replace('TOA.tif', 'class.tif')) as src:
        im_classif = src.read(1)
    
    # Create generic masks
    mask_gen, sand_mask, water_mask = create_crop_mask(im_classif, im_ref_buffer, settings)
    
    # Initialise plot colours
    cmap = cm.get_cmap('tab20c')
    colorpalette = cmap(np.arange(0,13,1))
    colours = np.zeros((3,4))
    colours[0,:] = colorpalette[5] # sand
    colours[1,:] = np.array([150/255,1,1,1]) # ww
    colours[2,:] = np.array([0,91/255,1,1]) # water
    
    # Loop through images
    for n, dates in enumerate(outputs['merged_data']):
        for sat in outputs['merged_data'][dates]:
            # Print progress
            curr_time = int(time.time() - start_time)
            time_elap = str(int(curr_time/60)) + 'm ' + str(curr_time%60) + 's'
            print('\rExtracting shoreline for date', n+1, 'of',number_dates,'(' + time_elap,' elapsed)', end = '')

            # extract data/structure and settings
            image_dict = outputs['merged_data'][dates][sat]
            index_im = image_dict['index_file']
            georef = image_dict['georef']
            toa_filepath = image_dict['toa_filepath'][0]
            georef = image_dict['georef']
            im_name = image_dict['toa_filename'][0][0:28]
            class_path = toa_filepath.replace('TOA.tif', 'class.tif')

            # Import water index and mask
            with rasterio.open(index_im) as src:
                index = src.read(1)
                
            with rasterio.open(class_path) as src:
                im_classif = src.read(1)
            
            # Extract crop mask
            if settings['generic_sl_region'] == False:
                # Create crop mask
                mask, sand_mask, water_mask = create_crop_mask(im_classif, im_ref_buffer, settings)
            else:
                mask = mask_gen
            
            # Get nan/cloud mask and im_ms for plotting
            im_ms, comb_mask = get_ps_data(toa_filepath)

            # Apply classified mask and cloud/nan
            masked_im = np.copy(index)
            masked_im[mask == 1] = np.nan
            masked_im[comb_mask] = np.nan
                                    
            # Extract sl contours (check # sand pixels)
            if settings['generic_sl_region'] or (np.sum(~sand_mask) < 100):
                contours, vec, t_otsu = sl_extract_generic(masked_im, settings)
            else:   
                # Generic crop region for contouring only (avoids 'other' misclassification error)
                masked_im_gen = np.copy(index)
                masked_im_gen[mask_gen == 1] = np.nan
                masked_im_gen[comb_mask] = np.nan
                # Extract SL
                contours, vec, t_otsu = sl_extract(masked_im, sand_mask, water_mask, masked_im_gen, settings)
            
            # Process shorelines
            shoreline_single = process_shoreline(contours, comb_mask,
                                                georef, image_epsg, settings)
            
            # Process shoreline for plotting
            sl_pix = convert_world2pix(shoreline_single, georef) 
            im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], comb_mask, 99.9)

            # Plot shorelines and histogram
            if settings['generic_sl_region']:
                fig, ax1, ax2, ax3 = initialise_plot_gen(settings, im_name, index)
                rgb_plot(ax1, im_RGB, sl_pix, transects)
                index_plot(ax2, index, t_otsu, comb_mask, sl_pix, 
                           transects, fig, settings)
                histogram_plot(ax3, vec, t_otsu, settings)
            else:
                fig, ax1, ax2, ax3 = initialise_plot(settings, im_name, index)
                rgb_plot(ax1, im_RGB, sl_pix, transects)
                #ax1.imshow(mask)
                class_plot(ax2, im_RGB, im_classif, sl_pix, transects, settings, colours) 
                index_plot(ax3, index, t_otsu, comb_mask, sl_pix, 
                           transects, fig, settings)
                # histogram_plot_split(ax4, index, im_classif, im_ref_buffer, t_otsu, settings, colours)
            
            # Save plot
            plt.close('all')
            save_path = settings['index_png_out']
            save_file = os.path.join(save_path, im_name + ' shoreline plot.png')
            fig.savefig(save_file, dpi=250)#, bbox_inches='tight', pad_inches=0.7) 
            
            # update dictionary
            shorelines_out['shorelines'] += [shoreline_single]
            shorelines_out['date'] += [str(image_dict['timestamp'][0].date())]
            shorelines_out['time'] += [str(image_dict['timestamp'][0].time())]
            shorelines_out['timestamp utc'] += [image_dict['timestamp'][0]]
            shorelines_out['cloud_cover'] += [image_dict['cloud_cover'][0]]
            shorelines_out['aoi_cover'] += [image_dict['aoi_coverage'][0]]
            shorelines_out['ps_sat_name'] += [sat]
            shorelines_out['name'] += [im_name]
            shorelines_out['sensor'] += [image_dict['Sensor'][0]]
            shorelines_out['threshold'] += [t_otsu]

    # Sort dictionary chronologically
    idx_sorted = sorted(range(len(shorelines_out['timestamp utc'])), key=shorelines_out['timestamp utc'].__getitem__)
    for key in shorelines_out.keys():
        shorelines_out[key] = [shorelines_out[key][i] for i in idx_sorted]
    
    # saving outputs
    print('\n    Saving shorelines to pkl and geojson')
        
    # save outputput structure as output.pkl
    with open(settings['sl_pkl_file'], 'wb') as f:
        pickle.dump(shorelines_out, f)
    
    # comvert to geopandas
    gdf = output_to_gdf_PL(shorelines_out)
    
    # set projection
    gdf.crs = {'init':str(settings['output_epsg'])}
    
    # save as geojson
    gdf.to_file(settings['sl_geojson_file'], driver='GeoJSON', encoding='utf-8')
    
    return shorelines_out



#%%

def compute_intersection(shoreline_data, settings):
    """
    Computes the intersection between the 2D shorelines and the shore-normal.
    transects. It returns time-series of cross-shore distance along each transect.
    
    KV WRL 2018 

    Modified YD 2020      

    Arguments:
    -----------
    output: dict
        contains the extracted shorelines and corresponding metadata
    transects: dict
        contains the X and Y coordinates of each transect
    settings: dict with the following keys
        'along_dist': int
            alongshore distance considered caluclate the intersection
              
    Returns:    
    -----------
    cross_dist: dict
        time-series of cross-shore distance along each of the transects. 
        Not tidally corrected. 

    """    
    print('Calculating shoreline intersections...')
    
    # unpack data
    shorelines = shoreline_data['shorelines']
    transects = settings['transects_load']

    # loop through shorelines and compute the median intersection    
    intersections = np.zeros((len(shorelines),len(transects)))
    for i in range(len(shorelines)):

        sl = shorelines[i]
        
        for j,key in enumerate(list(transects.keys())): 
            
            # compute rotation matrix
            X0 = transects[key][0,0]
            Y0 = transects[key][0,1]
            temp = np.array(transects[key][-1,:]) - np.array(transects[key][0,:])
            phi = np.arctan2(temp[1], temp[0])
            Mrot = np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]])
    
            # calculate point to line distance between shoreline points and the transect
            p1 = np.array([X0,Y0])
            p2 = transects[key][-1,:]
            d_line = np.abs(np.cross(p2-p1,sl-p1)/np.linalg.norm(p2-p1))
            # calculate the distance between shoreline points and the origin of the transect
            d_origin = np.array([np.linalg.norm(sl[k,:] - p1) for k in range(len(sl))])
            # find the shoreline points that are close to the transects and to the origin
            # the distance to the origin is hard-coded here to 1 km 
            idx_dist = np.logical_and(d_line <= settings['along_dist'], d_origin <= 1000)
            # find the shoreline points that are in the direction of the transect (within 90 degrees)
            temp_sl = sl - np.array(transects[key][0,:])
            phi_sl = np.array([np.arctan2(temp_sl[k,1], temp_sl[k,0]) for k in range(len(temp_sl))])
            diff_angle = (phi - phi_sl)
            idx_angle = np.abs(diff_angle) < np.pi/2
            # combine the transects that are close in distance and close in orientation
            idx_close = np.where(np.logical_and(idx_dist,idx_angle))[0]     
            
            # in case there are no shoreline points close to the transect 
            if len(idx_close) == 0:
                intersections[i,j] = np.nan
            else:
                # change of base to shore-normal coordinate system
                xy_close = np.array([sl[idx_close,0],sl[idx_close,1]]) - np.tile(np.array([[X0],
                                   [Y0]]), (1,len(sl[idx_close])))
                xy_rot = np.matmul(Mrot, xy_close)
                               
                # remove points that are too far landwards relative to the transect origin (i.e., negative chainage)
                xy_rot[0, xy_rot[0,:] < settings['min_val']] = np.nan
               
                # compute std, median, max, min of the intersections
                std_intersect = np.nanstd(xy_rot[0,:])
                max_intersect = np.nanmax(xy_rot[0,:])
                min_intersect = np.nanmin(xy_rot[0,:])
                n_intersect = len(xy_rot[0,:])
               
                # quality control the intersections using dispersion metrics (std and range)
                condition1 = std_intersect <= settings['max_std']
                condition2 = (max_intersect - min_intersect) <= settings['max_range']
                condition3 = n_intersect > settings['min no. intercepts']
                if np.logical_and(np.logical_and(condition1, condition2), condition3):
                    # compute the median of the intersections along the transect
                    intersections[i,j] = np.nanmedian(xy_rot[0,:])
                else:
                    intersections[i,j] = np.nan
                
    # fill the results into a dictionnary
    cross_dist = dict([])
    for j,key in enumerate(list(transects.keys())): 
        cross_dist[key] = intersections[:,j]   
    
    shoreline_data['cross_distance'] = cross_dist
    
    # save a .csv file for Excel users
    # Initialise data columns
    col_list = ['timestamp utc', 'filter', 'ps_sat_name', 'sensor', 'cloud_cover', 'aoi_cover', 'threshold']
    col_names = ['Date', 'Filter', 'PS Satellite Key', 'Sensor', 'Cloud Cover %', 'AOI Coverage %', 'Index Threshold'] 

    # Create dataframe
    csv_out = pd.DataFrame()
    for i in range(len(col_list)):
        csv_out[col_names[i]] = shoreline_data[col_list[i]]
    
    # Add intersection data
    for ts in cross_dist:
        col = pd.DataFrame(cross_dist[ts])
        csv_out[ts] = col
    
    # Save file
    csv_out.to_csv(settings['sl_transect_csv']) 

    return csv_out


