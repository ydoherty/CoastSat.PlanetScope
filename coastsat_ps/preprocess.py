# Image & mask preparation and merged scene generation
# YD, Sep 2020

import os
import time
import datetime
import rasterio
import numpy as np
from osgeo import gdal
import copy
import sys
from shutil import copyfile
import shutil

from coastsat_ps.interactive import (ref_im_select, get_reference_sl, 
                                     get_transects, merge_im_select)
from coastsat_ps.preprocess_tools import (save_mask, TOA_conversion, merge_crop, 
                                          get_cloud_percentage_nan_cloud, 
                                          get_file_extent, get_epsg, 
                                          local_coreg, mask_coreg, global_coreg,
                                          get_raster_bounds, gdal_subprocess, 
                                          zero_to_nan, load_udm, create_land_mask)
from coastsat_ps.plotting import check_land_mask

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

#%% Suppress outputs from a function

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


#%% Overall Pre-Processing Functions

def data_extract(settings, outputs):
    
    ''' Function combining all data extraction steps '''
    if len(os.listdir(settings['raw_data']))<2:
        # Time run
        start_time = time.time()
        
        # convert images to TOA
        convert_to_TOA(outputs, settings)
        
        # extract cloud and nan masks
        extract_masks(outputs, settings)
        
        # Print run time for pre processing
        runtime = (time.time() - start_time)
        
        print('Total Data Extraction run time =',int(runtime), 'seconds', '('+str(round(runtime/60,2))+') minutes')
    else:
        print('Previous run data loaded')

def select_ref_image(settings, replace_ref_im):
    
    if settings['im_coreg'] != 'Coreg Off':
        # When no georectified image provided, user select suitable toa image
        if settings['georef_im'] == False:
            settings['georef_im_path'] = os.path.join(settings['run_input_folder'], 
                                                      settings['site_name'] + '_im_ref.tif')
            if os.path.isfile(settings['georef_im_path']):
                if replace_ref_im:
                    ref_im_select(settings)
                else:
                    print('Previous reference image loaded')
            else:
                # Select cloud free raw toa image covering entire aoi
                ref_im_select(settings)
        else:
            print('Reference image loaded\n')
        
        # Create land mask
        if (not os.path.isfile(settings['land_mask'])) or replace_ref_im:
            print('\nCreating land mask...')
            create_land_mask(settings, settings['georef_im_path'], settings['land_mask'], nan_path = False, save_class = True)
            check_land_mask(settings)
    else:
        print('No reference image required for image co-registration = False')


def add_ref_features(settings):
        
    if not os.path.isfile(settings['ref_merge_im_txt']):
        # Select cloud free raw toa image covering entire aoi
        merge_im_select(settings)
    
    else:
        print('Reference merge image loaded')
        with open(settings['ref_merge_im_txt'], "r") as text_file:
                settings['ref_merge_im'] = text_file.read()
        
    # Digitise reference shoreline
    get_reference_sl(settings)
    
    
    # Import transects
    if settings['transects'] == False:
        settings['geojson_file'] = os.path.join(settings['run_input_folder'], 
                                                  settings['site_name'] + '_transects.geojson')
        get_transects(settings)
    else:
        print('Transects loaded')



def pre_process(settings, outputs, del_files_int = True, del_no_pass = False, print_no_pass = False, rerun_preprocess = False):
    
    # Check if pre-processing run previously
    if len(os.listdir(settings['merge_out']))<1 or rerun_preprocess:
        # Time run
        start_time = time.time()
        
        # Map raw folder contents
        map_raw_folder(settings, outputs)
        
        # Co-register images
        coregistration(settings, outputs)
        
        # Merge overlapping images to create scenes
        image_merge(settings, outputs, del_no_pass = del_no_pass, 
                    print_no_pass = print_no_pass)
        
        # Print run time for pre processing
        runtime = (time.time() - start_time)
            
        print('\nTotal Pre-Processing run time =',int(runtime), 'seconds', '('+str(round(runtime/60,2))+') minutes\n')
    
    else:
        print('\nPre-process run previously, skipping coregistration and merge steps\n')

        # Map folder
        map_merge(settings, outputs)


    if del_files_int == True:
        if settings['im_coreg'] != 'Coreg Off':
            if os.path.isdir(settings['coreg_out']):
                shutil.rmtree(settings['coreg_out'])
            # else:
            #     os.rmdir(settings_dict[merge_folder])
                    


#%% Pre process steps

def coregistration(settings, outputs):
    
    ''' Function combining all pre-processing steps - Coregistration'''
    
    # Time run
    start_time = time.time()
    
    # co-register images
    if settings['im_coreg'] != 'Coreg Off':
        batch_coreg(settings, outputs)
        map_raw_folder(settings, outputs, coreg = True)
        
    else:
        print('Co-registration step skipped')
    
    # Print run time for pre processing
    runtime = (time.time() - start_time)
    print('\nTotal co-registration run time =',int(runtime), 'seconds', '('+str(round(runtime/60,2))+') minutes\n')


def image_merge(settings, outputs, print_merge_summary = False, 
                del_no_pass = False, print_no_pass = False, debug_zero_nan = False):
    
    ''' Function combining all merge steps '''
    
    # Time run
    start_time = time.time()
    
    # Merge TOA and UDM files based on date and satellite
    if settings['im_coreg'] != 'Coreg Off':
        merge_raw_files(settings, outputs, print_merge_summary, coreg = True)
    else:
        merge_raw_files(settings, outputs, print_merge_summary)
                
    # Filter by AOI/cloud cover and Map merge folder
    map_merge(settings, outputs, del_no_pass, print_no_pass)
    
    # Print run time for pre processing
    runtime = (time.time() - start_time)
    print('Total image merge run time =',int(runtime), 'seconds', '('+str(round(runtime/60,2))+') minutes')


#%% Pre Processing Components

def convert_to_TOA(output_dict, settings):  
    
    ''' Convert image DN values to TOA using Planet xml file conversion values '''
    
    # time the run
    start_time = time.time()
    
    # Find number of scenes to be converted and initialise count
    noImage = settings['input_image_count']
    count = 1
    
    # convert each MS tif file
    for folder in output_dict['downloads_map']:  
        for i in range(len(output_dict['downloads_map'][folder]['_AnalyticMS_clip.tif']['filepaths'])):
            # print progress
            if settings['arosics_reproject'] == False:
                print('\rConverting image ' + str(count) + ' of ' + str(noImage) + ' to TOA (' + str(int((count/noImage)*100)) + '%)', end = '')
            else:
                print('\rConverting image ' + str(count) + ' of ' + str(noImage) + ' to TOA (' + str(int((count/noImage)*100)) + '%) - arosics_reproject workaround applied', end = '')
            image_path = output_dict['downloads_map'][folder]['_AnalyticMS_clip.tif']['filepaths'][i]
            # Find corresponding xml file
            search_id = output_dict['downloads_map'][folder]['_AnalyticMS_clip.tif']['filenames'][i][9:20]
            for ii in range(len(output_dict['downloads_map'][folder]['_metadata_clip.xml']['filenames'])):            
                if output_dict['downloads_map'][folder]['_metadata_clip.xml']['filenames'][ii][9:20] == search_id:
                    xml_path = output_dict['downloads_map'][folder]['_metadata_clip.xml']['filepaths'][ii]
            
            save_name = output_dict['downloads_map'][folder]['_AnalyticMS_clip.tif']['filenames'][i].replace('.tif','')
            save_path = os.path.join(settings['raw_data'],save_name)
            
            count = count + 1
                        
            # convert to TOA
            TOA_conversion(settings, image_path, xml_path, save_path)
                                                
    # print run time
    print('\n    '+str(noImage)+' images converted to TOA in ' + str(round(time.time() - start_time)) + ' seconds\n')
   
    
#%%

def extract_masks(output_dict, settings):
    
    ''' Extract mask from UDM '''
    
    # time the run
    start_time = time.time()

    # Find number of scenes to be masked and initialise count
    img_count = 1
    img_all = settings['input_image_count']
    
    # find total number of images  
    noAllImages = 0
    for folder in output_dict['downloads_map']:   
        noAllImages = noAllImages + len(output_dict['downloads_map'][folder]['_AnalyticMS_clip.tif']['filepaths'])
    
    # initialise udm counts
    udm_count = 0
    udm2_count = 0
    
    # Mask each MS tif file (check if udm or udm2 mask)
    for folder in output_dict['downloads_map']:  
        no_images = len(output_dict['downloads_map'][folder]['_AnalyticMS_clip.tif']['filepaths'])
        
        for i in range(no_images):
            search_id = output_dict['downloads_map'][folder]['_AnalyticMS_clip.tif']['filenames'][i][9:20]
            
            # print progress
            print('\rExtracting masks for image ' + str(img_count) + ' of ' + str(img_all), end='')
            img_count = img_count + 1 
            
            # Find corresponding udm or udm2 file
            if len(output_dict['downloads_map'][folder]['_udm2_clip.tif']['filenames'])>0:
                udm2_count = udm2_count + 1
                print(' - Code for new UDM2 mask not yet supported. Original UDM form used instead', end = '')
                
                # Add udm2 workaround if no udm present - creates udm from udm2
                if len(output_dict['downloads_map'][folder]['_udm_clip.tif']['filenames']) == 0:
                    
                    for i_ in range(len(output_dict['downloads_map'][folder]['_udm2_clip.tif']['filenames'])):
                    
                        udm2_name = output_dict['downloads_map'][folder]['_udm2_clip.tif']['filenames'][i_]
                        udm2_path = output_dict['downloads_map'][folder]['_udm2_clip.tif']['filepaths'][i_]
    
                        with rasterio.open(udm2_path) as src:
                            udm = src.read(8)
                        
                        # Set spatial characteristics of the output object to mirror the input
                        kwargs = src.meta
                        kwargs.update(
                            dtype=rasterio.uint8,
                            count = 1)
                            
                        udm_path = udm2_path.replace('udm2_clip.tif', 'AnalyticMS_DN_udm_clip.tif')
                        udm_name = udm2_name.replace('udm2_clip.tif', 'AnalyticMS_DN_udm_clip.tif')

                        with rasterio.open(udm_path, 'w', **kwargs) as dst:
                                dst.write_band(1, udm.astype(rasterio.uint8))

                        output_dict['downloads_map'][folder]['_udm_clip.tif']['filenames'] += [udm_name]
                        output_dict['downloads_map'][folder]['_udm_clip.tif']['filepaths'] += [udm_path]
                
            if len(output_dict['downloads_map'][folder]['_udm_clip.tif']['filenames']) == 0:
                print('No mask found for file' + output_dict['downloads_map'][folder]['_AnalyticMS_clip.tif']['filenames'][i])
            else:
                for ii in range(len(output_dict['downloads_map'][folder]['_udm_clip.tif']['filenames'])):
                    if output_dict['downloads_map'][folder]['_udm_clip.tif']['filenames'][ii][9:20] == search_id:
                        udm_count = udm_count + 1
                        
                        mask_path = output_dict['downloads_map'][folder]['_udm_clip.tif']['filepaths'][ii]
                        
                        save_name = output_dict['downloads_map'][folder]['_udm_clip.tif']['filenames'][ii].replace('.tif','')+'_cloud_mask.tif'
                        save_path = os.path.join(settings['raw_data'],save_name) 
                        save_mask(settings, mask_path, save_path,'00000010', cloud_issue = True)
                        
                        save_name = output_dict['downloads_map'][folder]['_udm_clip.tif']['filenames'][ii].replace('.tif','')+'_NaN_mask.tif'
                        save_path = os.path.join(settings['raw_data'], save_name) 
                        save_mask(settings, mask_path, save_path,'1111100', nan_issue = True)
                                                                                                   
    # print run time
    print('\n\n    '+str(img_all)+' masks extracted in ' + str(round(time.time() - start_time)) + ' seconds')
    print('        ' + str(udm_count) + ' udm masks extracted' + '\n        ' + str(udm2_count) + ' udm2 masks converted to udm\n')
    return output_dict

    
#%%

def map_raw_folder(settings_dict, output_dict, coreg = False):    
    
    ''' Create dictionary of created raw TOA and mask file locations '''
    
    # unpack input dictionary
    if coreg:
        folder = settings_dict['coreg_out']
    else:
        folder = settings_dict['raw_data']
    
    # initialise output
    map_dict = {}
    count = 0
                    
    # Initialise dictionary structure using file dates
    for file in os.listdir(folder):
        if file.endswith('TOA.tif'):
            # get date of file
            year = file[0:4]
            month = file[4:6]
            day = file[6:8]
            date = year + '-' + month + '-' + day
            count = count + 1
                
            # create dictionary for each date
            if date not in map_dict.keys():
                map_dict[date] = {}
                for end in ['TOA.tif', 'NaN_mask.tif', 'cloud_mask.tif']:
                    map_dict[date][end] = {'filenames': [], 'filepaths':[], 'timestamp':[]}
                    if end == 'TOA.tif':
                        map_dict[date][end]['epsg'] = []
           
    # Fill in dictionary with filepath details
    for file in os.listdir(folder):
        for end in ['TOA.tif', 'NaN_mask.tif', 'cloud_mask.tif']:
            if file.endswith(end):
                # get date of file
                year = file[0:4]
                month = file[4:6]
                day = file[6:8]
                date = year + '-' + month + '-' + day
                
                # update dictionary
                map_dict[date][end]['filenames'] = map_dict[date][end]['filenames'] + [file]
                map_dict[date][end]['filepaths'] = map_dict[date][end]['filepaths'] + [os.path.join(folder, file)]
                
                # add time
                hour = file[9:11]
                minutes = file[11:13]
                seconds = file[13:15]
                time = date + ' '+str(hour) +':'+ str(minutes) +':'+str(seconds)
                time = datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S')
                
                map_dict[date][end]['timestamp'] = map_dict[date][end]['timestamp'] + [time]
        
                # Add epsg to dict   
                if end == 'TOA.tif':
                    if settings_dict['arosics_reproject'] == True:
                        # modify input epsg hen using arosics workaround
                        map_dict[date][end]['epsg'] += [settings_dict['output_epsg'].replace('EPSG:','')]
                    else:
                        map_dict[date][end]['epsg'] += [get_epsg(output_dict, date, file)]
        
    if len(map_dict) == 0:
        print('\n'+'No images found, check inputs and retry')  
    
    if coreg:
        output_dict['coreg_data'] = map_dict
    else:
        output_dict['raw_data'] = map_dict
        
    return output_dict
 
    
#%% 

def batch_coreg(settings, outputs):
    
    # Unpack data
    im_reference = settings['georef_im_path']
    out_folder = settings['coreg_out']
    
    # Time run
    start_time = time.time()
    img_all = settings['input_image_count']
    img_count = 1
    
    # Add outcome counter
    success = 0
    small_area = 0
    failure = 0

    # loop through every raw image
    for dates in outputs['raw_data']:
        for toa_id in range(len(outputs['raw_data'][dates]['TOA.tif']['filenames'])):
            
            #print('\nPerforming co-registration for image ' + str(img_count) + ' of ' + str(img_all))
            print('\rPerforming co-registration for image ' + str(img_count) + ' of ' + str(img_all) + ': ' + 
                  outputs['raw_data'][dates]['TOA.tif']['filenames'][toa_id] + '     ', end='')

            img_count += 1
            
            # Set TOA location
            im_target = outputs['raw_data'][dates]['TOA.tif']['filepaths'][toa_id]
            im_id = outputs['raw_data'][dates]['TOA.tif']['filenames'][toa_id][0:23]
            
            # Find corresponding masks
            for cloud_id in range(len(outputs['raw_data'][dates]['cloud_mask.tif']['filenames'])):
                if outputs['raw_data'][dates]['cloud_mask.tif']['filenames'][cloud_id][0:23] == im_id:
                    cloud_mask = outputs['raw_data'][dates]['cloud_mask.tif']['filepaths'][cloud_id]
                    cloud_id_match = copy.deepcopy(cloud_id)
            
            for nan_id in range(len(outputs['raw_data'][dates]['NaN_mask.tif']['filenames'])):
                if outputs['raw_data'][dates]['NaN_mask.tif']['filenames'][nan_id][0:23] == im_id:
                    nan_mask = outputs['raw_data'][dates]['NaN_mask.tif']['filepaths'][nan_id]
                    nan_id_match = copy.deepcopy(nan_id)
            
            # Set save locations
            im_out = os.path.join(out_folder, outputs['raw_data'][dates]['TOA.tif']['filenames'][toa_id])
            cloud_out = os.path.join(out_folder, outputs['raw_data'][dates]['cloud_mask.tif']['filenames'][cloud_id_match])
            nan_out = os.path.join(out_folder, outputs['raw_data'][dates]['NaN_mask.tif']['filenames'][nan_id_match])
                        
            # Perform co-registration
            outcome = single_mask_coreg(settings, 
                   im_reference, im_target, im_out, 
                   nan_mask, cloud_mask, nan_out, cloud_out)
            
            # Update outcome counters
            if outcome == 0:
                success += 1
            elif outcome == 1:
                small_area += 1
            elif outcome == 2:
                failure += 1
            
    # Print run time
    print('\n\n'+str(img_all)+' image co-registration attempts performed in ' + str(round(time.time() - start_time)) + ' seconds')
    print('    ' + str(success) + ' successful co-registrations')
    print('    ' + str(small_area) + ' co-registrations skipped due to insufficient overlap')
    print('    ' + str(failure) + ' un-successful co-registrations')

def single_mask_coreg(settings, 
                       im_reference, im_target, im_out, 
                       nan_mask_path, cloud_mask_path, nan_out, cloud_out):

    ''' Image co-registration based on a land mask '''    
    
    # Initalise outcome object
    outcome = 0
    
    # Temp mask save location
    tgt_mask_temp = settings['land_mask'].replace('.tif','_temp.tif')
    
    # Create combined mask with zero vals and cloud
    nan_mask = zero_to_nan(im_target, nan_mask_path, faulty_pixels = settings['faulty_pixels'], write = False)  
    cloud_mask = load_udm(cloud_mask_path)
    mask_comb = (nan_mask + cloud_mask) > 0
    
    if settings['generic_land_mask']:
        # Crop generic mask for images with a different footprint to ref im
        if get_raster_bounds(im_target) != get_raster_bounds(settings['land_mask']):
            # Create new cropped mask
            bounds = get_raster_bounds(im_target)
            gdal_subprocess(settings, 'gdal_translate', ['-projwin', str(bounds[0]), str(bounds[1]), str(bounds[2]), str(bounds[3]), 
                                               '-of', 'GTiff', 
                                               settings['land_mask'], tgt_mask_temp])
            land_load = tgt_mask_temp
        else:
            land_load = settings['land_mask']
    else:
        create_land_mask(settings, im_target, tgt_mask_temp, 
                         nan_path = nan_mask_path, raw_mask = mask_comb)
        land_load = tgt_mask_temp

    # Import masks
    with rasterio.open(land_load) as src:
        land_mask = src.read(1)
        kwargs = src.meta    
    
    # Combine masks
    mask = (mask_comb + land_mask) > 0

    # Save combined mask to temp location
    with rasterio.open(tgt_mask_temp, 'w', **kwargs) as dst:
        dst.write_band(1, mask.astype(rasterio.uint8))
        
    # Perform coregistration where valid overlap and save file to new location
    if np.sum(~mask) < 10*settings['grid_size']**2:
        copyfile(im_target, im_out)
        copyfile(cloud_mask_path, cloud_out)
        copyfile(nan_mask_path, nan_out)
        outcome = 1
        #print('Not enough overlap for co-registration, raw image copied to coreg folder')
    else:
        # Mute output
        with HiddenPrints():
            # Development hard coded setting
            if settings['local_avg']:
                # High # min points ensures a pseudo global x/y shift from tie points avg
                min_points = 100
            else:
                min_points = 5
        
            if settings['im_coreg'] == 'Local Coreg':
                # Calculate local co-registration parameters and apply to TOA image
                cr_param, coreg_success = local_coreg(im_reference, im_target, im_out, 
                        land_mask_ref = settings['land_mask'], land_mask_tgt = tgt_mask_temp,
                        grid_res = settings['grid_size'], window_size = settings['window_size'], 
                        min_points = min_points,
                        q = True, progress = False, ignore_errors = True, 
                        #q = False, progress = True, ignore_errors = False, 
                        filter_level = settings['filter_level'])
                
            elif settings['im_coreg'] == 'Global Coreg':
                # Calculate overlap coreg match size
                #ws = (mask.shape[1], mask.shape[0])
                ws = (256,256)
                
                # Calculate global co-registration parameters and apply to TOA image
                cr_param, coreg_success = global_coreg(im_reference, im_target, im_out, 
                        land_mask_ref = settings['land_mask'], land_mask_tgt = tgt_mask_temp,
                        ws = ws, 
                        q = True, progress = False, ignore_errors = True)
            
            # Apply co-registration parameters to NaN and Cloud masks
            mask_coreg(settings, cloud_mask_path, cr_param, cloud_out, 
                       min_points = min_points, coreg_success = coreg_success, q = True, progress = False)
            mask_coreg(settings, nan_mask_path, cr_param, nan_out, 
                       min_points = min_points, coreg_success = coreg_success, q = True, progress = False)
        
            # update outcome
            if coreg_success == False:
                outcome = 2
        
    # Remove created mask
    os.remove(tgt_mask_temp)
    
    return outcome

    
#%%

def merge_raw_files(settings_dict, output_dict, print_merge_summary = False, coreg = False):  
    
    ''' Merge overlapping TOA and mask images (from same day and satellite) '''
    
    if coreg:
        merge_folder = 'coreg_data'
    else:
        merge_folder = 'raw_data'
    
    # time the run
    start_time = time.time()

    # Find number of scenes to be merged/cropped and initialise count
    noScene = len(output_dict['downloads_map'])
            
    # initialise counters and data holders
    mergeCount = 0
    cropCount = 0
    count = 0
    imageAll = 0
    max_diff = 0
    print_list = ''
    name_list = []
    error_message = []
    
    # merge/crop each tif file and udm file (NaN and Cloud)
    for folder in output_dict[merge_folder]:  
        # create temp dictionary for each date folder
        files_dict = {}
        for data_type in ['TOA.tif', 'NaN_mask.tif', 'cloud_mask.tif']:
            files_dict[data_type] = output_dict[merge_folder][folder][data_type]
        imageAll = imageAll + len(output_dict[merge_folder][folder]['TOA.tif']['filenames'])
        
        # merge/crop based on number of files in folder
        if len(files_dict['TOA.tif']['filenames'])>1:
            # Merge these images
            sat_id_list = {}
            # find unique satellite ids for images on each date
            for i in range(len(files_dict['TOA.tif']['filenames'])):
                sat_id = files_dict['TOA.tif']['filenames'][i][16:20]
                toa_id = files_dict['TOA.tif']['filenames'][i][0:20]
                if '_' in sat_id:
                    #sat_id = str(files_dict['TOA.tif']['filenames'][i][19:23])# + '_' + str(files_dict['TOA.tif']['filenames'][0][16:18])
                    sat_id = str(files_dict['TOA.tif']['filenames'][i][-35:-31])
                if sat_id not in sat_id_list:
                    # create temp data dict of satellite ids for date
                    sat_id_list[sat_id] = {} 
                    for data_type in ['TOA.tif', 'NaN_mask.tif', 'cloud_mask.tif']:
                        sat_id_list[sat_id][data_type] = {}
                    # update dict
                    for data_type in sat_id_list[sat_id]:  
                        # find index corresponding to toa file
                        for ii in range(len(files_dict[data_type]['filenames'])):
                            search_id = files_dict[data_type]['filenames'][ii][0:20]
                            if search_id == toa_id:
                                idx = ii
                        
                        sat_id_list[sat_id][data_type]['filepath'] = [files_dict[data_type]['filepaths'][idx]]
                        sat_id_list[sat_id][data_type]['timestamp'] = [files_dict[data_type]['timestamp'][idx]]
                else:
                    # update dict
                    for data_type in sat_id_list[sat_id]: 
                        # find index corresponding to toa file
                        for ii in range(len(files_dict[data_type]['filenames'])):
                            search_id = files_dict[data_type]['filenames'][ii][0:20]
                            if search_id == toa_id:
                                idx = ii
                        
                        sat_id_list[sat_id][data_type]['filepath'] = sat_id_list[sat_id][data_type]['filepath'] + [files_dict[data_type]['filepaths'][idx]]
                        sat_id_list[sat_id][data_type]['timestamp'] = sat_id_list[sat_id][data_type]['timestamp'] + [files_dict[data_type]['timestamp'][idx]]
            
            for ids in sat_id_list:
                # merge if >1 file in folder
                if len(sat_id_list[ids]['TOA.tif']['filepath'])>1:
                    # extract/analyse time data for files from same satellite on same day
                    time_list = []
                    for times in sat_id_list[ids]['TOA.tif']['timestamp']:
                        time_list = time_list + [datetime.datetime.timestamp(times)]
                    time_max = max(time_list)
                    time_min = min(time_list)
                    time_dif = time_max-time_min
                    av_time = datetime.datetime.fromtimestamp(time_min + int(time_dif/len(time_list)))
                    print_list = print_list + str(len(time_list)) + ' images with time range of ' + str(int(time_dif)) + ' seconds (' + str(av_time) + ')'+ '\n'
                    # update max time difference value
                    if time_dif>max_diff:
                        max_diff = time_dif
                    # merge images
                    for data_type in sat_id_list[sat_id]: 
                        # create name from timestamps
                        name = str(av_time)[0:10]+ ' '+str(av_time)[11:13] + '_'+str(av_time)[14:16] + '_'+str(av_time)[17:19] + ' ' +ids + ' '
                        
                        # update sat type to name
                        if data_type == 'TOA.tif':
                            name += sat_id_list[ids][data_type]['filepath'][0][-11:-8] + ' merged ' + data_type
                        else:
                            name += 'merged ' + data_type
                            
                        # different mask for nan masks
                        if data_type == 'NaN_mask.tif': 
                            nan_mask_bool = True
                        else:
                            nan_mask_bool = False
                            
                        # Retireve and compare image epsg's
                        epsg_list = []
                        for merge_file in sat_id_list[ids][data_type]['filepath']:
                            for index_match in range(len(output_dict[merge_folder][folder][data_type]['filepaths'])):
                                if merge_file == output_dict[merge_folder][folder][data_type]['filepaths'][index_match]:
                                    epsg_list += [output_dict[merge_folder][folder]['TOA.tif']['epsg'][index_match]]
                        if len(np.unique(epsg_list)) != 1:
                            print('Error: epsg codes do not match, cannot merge without conversion first. Try setting arosics_reproject setting to True. ')
                        else:                    
                            epsg_in = 'EPSG:' + str(np.unique(epsg_list)[0])
                                                  
                        merge_crop(settings_dict, sat_id_list[ids][data_type]['filepath'], name, nan_mask = nan_mask_bool, epsg_in = epsg_in)
                        epsg_in = 0
                                                
                        if name in name_list:
                            error_message = error_message + [name]
                        name_list = name_list + [name] 
                    
                    # update count
                    mergeCount = mergeCount + 1
                
                else:
                    # time as name for crop
                    for data_type in sat_id_list[sat_id]: 
                        timestamp = sat_id_list[ids][data_type]['timestamp'][0]
                        name = str(timestamp)[0:10]+ ' '+str(timestamp)[11:13] + '_'+str(timestamp)[14:16] + '_'+str(timestamp)[17:19] + ' ' +ids + ' '
                        # add sat sensor type
                        if data_type == 'TOA.tif':
                            name += sat_id_list[ids][data_type]['filepath'][0][-11:-8] + ' cropped ' + data_type
                        else:
                            name += 'cropped ' + data_type
                        
                        if data_type == 'NaN_mask.tif': 
                            nan_mask_bool = True
                        else:
                            nan_mask_bool = False
                            
                        # Retireve image epsg
                        for index_match in range(len(output_dict[merge_folder][folder][data_type]['filepaths'])):
                            if sat_id_list[ids][data_type]['filepath'][0] == output_dict[merge_folder][folder][data_type]['filepaths'][index_match]:
                                epsg_in = 'EPSG:' + str(output_dict[merge_folder][folder]['TOA.tif']['epsg'][index_match])
                            
                        merge_crop(settings_dict, sat_id_list[ids][data_type]['filepath'], name, nan_mask = nan_mask_bool, epsg_in = epsg_in)
                        epsg_in = 0

                        if name in name_list:
                            error_message = error_message + [name]
                        name_list = name_list + [name] 
                    
                    # update count
                    cropCount = cropCount + 1
                
            count = count + 1
        
        # Crop these images
        elif len(files_dict['TOA.tif']['filenames'])==1:  
            sat_id = files_dict['TOA.tif']['filenames'][0][16:20]   
            if '_' in sat_id:
                sat_id = str(files_dict['TOA.tif']['filenames'][0][-35:-31])
            # time criteria to merge
            for data_type in files_dict:
                timestamp = files_dict[data_type]['timestamp'][0]
                name = str(timestamp)[0:10]+ ' '+str(timestamp)[11:13] + '_'+str(timestamp)[14:16] + '_'+str(timestamp)[17:19] + ' ' +sat_id + ' '
                
                # add sat sensor type
                if data_type == 'TOA.tif':
                    name += files_dict[data_type]['filenames'][0][-11:-8] + ' cropped ' + data_type
                else:
                    name += 'cropped ' + data_type
                
                if data_type == 'NaN_mask.tif': 
                    nan_mask_bool = True
                else:
                    nan_mask_bool = False
                
                # Retireve image epsg
                for index_match in range(len(output_dict[merge_folder][folder][data_type]['filepaths'])):
                    if files_dict[data_type]['filepaths'][0] == output_dict[merge_folder][folder][data_type]['filepaths'][index_match]:
                        epsg_in = 'EPSG:' + str(output_dict[merge_folder][folder]['TOA.tif']['epsg'][index_match])

                merge_crop(settings_dict, files_dict[data_type]['filepaths'], name, nan_mask = nan_mask_bool, epsg_in = epsg_in)
                epsg_in = 0
                
                if name in name_list:
                    error_message = error_message + [name]
                name_list = name_list + [name] 
            
            # update count
            cropCount = cropCount + 1
            count = count + 1
        
        print('\r'+str(count) + ' of ' + str(noScene) + ' Dates Processed (' + 
            str(mergeCount) + ' images merged, ' +  
            str(cropCount) + ' images cropped)', end = '')

    # print run time
    if print_merge_summary == True:
        print('\n\nMerge Summary:' + '\n' + print_list)
        
    print('\n    '+str(round(time.time() - start_time)) + ' seconds to process ' + str(imageAll) + ' images')
    print('    '+str(cropCount+mergeCount) + ' scenes output from ' + str(imageAll) + ' images')
    print('    max image merge time difference =',max_diff,'seconds\n')
    
    # print error message
    if error_message != []:
        print('files with non-standard naming:\n',error_message)
    coreg

    
#%%

def filter_aoi_cloud(settings, del_no_pass = False, print_no_pass = False):
    
    ''' 
    Creates a dictionary of merged file locations that pass input setting 
    threshold criteria (ie. max cloud, min NaN)
    
    Additionally updates nan mask with zero vals from MS image
    '''
    
    # unpack input dictionary
    folder = settings['merge_out']
    cloud_thresh = settings['cloud_threshold']
    extent_thresh = settings['extent_thresh']
    
    # initialise output
    file_count = 0
    cloud_fail = 0
    nan_fail = 0
    count = 0
    no_image = 0
    
    # Count # images
    for file in os.listdir(folder):
        if file.endswith('TOA.tif'):
            no_image += 1
        
    # Create list of file ids passing cloud criteria (excludes other files in mapping and later use)
    cloud_pass_id = []
    for file in os.listdir(folder):
        if file.endswith('TOA.tif'):
            print('\rFiltering image ' + str(count+1) + ' of ' + str(no_image), end = '')
            count += 1
            
            toa_path = os.path.join(folder, file)
            image_id = file[0:24]
            file_count = file_count +1
            
            # find corresponding nan and cloud file
            cloud_path = 0
            nan_path = 0
            for all_file in os.listdir(folder):
                if all_file.startswith(image_id):
                    if all_file.endswith('cloud_mask.tif'):
                        cloud_path = os.path.join(folder,all_file)
                    elif all_file.endswith('NaN_mask.tif'):
                        nan_path = os.path.join(folder,all_file)
            if nan_path == 0:
                print('No NaN mask found for file',file)
            if cloud_path == 0:
                print('No cloud mask found for file',file)

            # Update nan mask with zero vals from MS image
            zero_to_nan(toa_path, nan_path, faulty_pixels = settings['faulty_pixels'], write = True)
                
            # Calculate cloud percentage
            cloud_perc = get_cloud_percentage_nan_cloud(nan_path, cloud_path)
                        
            # calculate area extent
            image_extent = get_file_extent(nan_path)
              
            # compare againts threshold
            if cloud_perc > cloud_thresh or image_extent < extent_thresh:
                if cloud_perc > cloud_thresh and image_extent < extent_thresh:
                    if print_no_pass == True:
                        print('File (' + str(image_id) + ') did not pass criteria\n'+
                              '    Cloud percentage', cloud_perc, '> maximum', cloud_thresh,'\n',
                              '    AOI coverage', round(image_extent,2), '< min', extent_thresh,'%\n')
                    cloud_fail = cloud_fail + 1
                    nan_fail = nan_fail + 1
                elif cloud_perc > cloud_thresh:
                    if print_no_pass == True:
                        print('File ('+ str(image_id) + ') did not pass criteria\n'+
                              '    Cloud percentage', cloud_perc, '> maximum', cloud_thresh, '%\n')
                    cloud_fail = cloud_fail + 1
                elif image_extent < extent_thresh:
                    if print_no_pass == True:
                        print('File ('+ str(image_id) + ') did not pass criteria\n'+
                              '    AOI coverage', round(image_extent,2), '< min', extent_thresh,'%\n')
                    nan_fail = nan_fail + 1
                if del_no_pass == True:
                    os.remove(nan_path)
                    os.remove(cloud_path)
                    os.remove(os.path.join(folder,file))
                    print('TOA, Cloud and NaN data deleted for',image_id)
            if cloud_perc <= cloud_thresh and image_extent >= extent_thresh:
                cloud_pass_id = cloud_pass_id + [image_id]
    
    print('\n\n' + str(len(cloud_pass_id)) + ' of '+ str(file_count)+ ' scenes passed cloud and area criteria')
    print('   ', nan_fail,'files covered <', extent_thresh, '% of the AOI')
    print('   ',cloud_fail,'files had cloud coverage >', cloud_thresh, '%\n')
    
    return cloud_pass_id


def map_merge(settings, outputs, del_no_pass = False, print_no_pass = False):

    # initialise
    map_dict = {}
    folder = settings['merge_out']
    
    # Remove merged images where nan mask in cropped aoi is all true
    rem_list = []
    for file in os.listdir(folder):
        if file.endswith('TOA.tif'):
            for crop in ['merged', 'cropped'] :
                if crop in file:
                    cloud = file[:25] + crop + ' cloud_mask.tif'
                    nan = file[:25] + crop + ' NaN_mask.tif'

            with rasterio.open(os.path.join(folder, nan)) as src:
                im1 = src.read(1)
            if np.min(im1) == 1:
                rem_list += [file]
                
                # remove im/nan/cloud from merge out folder
                os.remove(os.path.join(settings['merge_out'], file))
                os.remove(os.path.join(settings['merge_out'], cloud))
                os.remove(os.path.join(settings['merge_out'], nan))
     
    if len(rem_list) != 0:
        print(len(rem_list), 'files removed as they contain no data after cropping to AOI from input .kml file')

    # Filter by AOI and cloud cover
    cloud_pass_id = filter_aoi_cloud(settings, del_no_pass, print_no_pass)
    
    # Initialise dictionary structure using file dates
    for file in os.listdir(folder):
        if file.endswith('TOA.tif'):
            file_id = file[0:24]
            if file_id in cloud_pass_id:
                # get date of file and sat ID
                date = file[0:10]
                # if file[24] == '_':
                #     sat_id = file[20:27]
                # else:
                #     sat_id = file[20:24]
                sat_id = file[20:24]
                    
                # create dictionary for each date
                if date not in map_dict.keys():
                    map_dict[date] = {}
                    map_dict[date][sat_id] = {
                            'toa_filename': [],
                            'toa_filepath':[],
                            'nan_filepath':[],
                            'cloud_filepath':[],                            
                            'timestamp':[],
                            'cloud_cover':[], 
                            'aoi_coverage':[],
                            'Sensor':[]
                             }
                elif sat_id not in map_dict[date].keys():
                    map_dict[date][sat_id] = {
                            'toa_filename': [],
                            'toa_filepath':[],
                            'nan_filepath':[],
                            'cloud_filepath':[],                            
                            'timestamp':[],
                            'cloud_cover':[],
                            'aoi_coverage':[],
                            'Sensor':[]
                             }
                else:
                    print('Somethings gone wrong uh oh')
                    
    # Fill in dictionary with filepath details
    for file in os.listdir(folder):
        if file.endswith('TOA.tif'):
            file_id = file[0:24]
            if file_id in cloud_pass_id:
                # get details from file
                date = file[0:10]
                time = file[11:19]
                timestamp = datetime.datetime.strptime(date + ' ' + time, '%Y-%m-%d %H_%M_%S')
                # if file[24] == '_':
                #     sat_id = file[20:27]
                #     sensor = file[28:30]
                # else:
                #     sat_id = file[20:24]
                #     sensor = file[25:27]
                sat_id = file[20:24]
                sensor = file[25:28]
                
                # update dictionary
                map_dict[date][sat_id]['toa_filename'] = map_dict[date][sat_id]['toa_filename'] + [file]
                map_dict[date][sat_id]['toa_filepath'] = map_dict[date][sat_id]['toa_filepath'] + [os.path.join(folder, file)]
                map_dict[date][sat_id]['timestamp'] = map_dict[date][sat_id]['timestamp'] + [timestamp]
                map_dict[date][sat_id]['Sensor'] += [sensor]
                                
                # find corresponding nan and cloud file
                for all_file in os.listdir(folder):
                    if all_file.startswith(file_id):
                        if all_file.endswith('cloud_mask.tif'):
                            cloud_path = os.path.join(folder,all_file)
                            map_dict[date][sat_id]['cloud_filepath'] = map_dict[date][sat_id]['cloud_filepath'] + [cloud_path]
                        elif all_file.endswith('NaN_mask.tif'):
                            nan_path = os.path.join(folder,all_file)
                            map_dict[date][sat_id]['nan_filepath'] = map_dict[date][sat_id]['nan_filepath'] + [nan_path]
                            
                # Calculate cloud percentage
                cloud_perc = get_cloud_percentage_nan_cloud(nan_path, cloud_path)
                map_dict[date][sat_id]['cloud_cover'] = map_dict[date][sat_id]['cloud_cover'] + [cloud_perc]
                
                # calculate area extent
                image_extent = get_file_extent(nan_path)
                map_dict[date][sat_id]['aoi_coverage'] = map_dict[date][sat_id]['aoi_coverage'] + [image_extent]
                
                if len(map_dict[date][sat_id]['toa_filepath'])>1:
                    print('Somethin wrong')
                            
    print(len(map_dict),'dates remain from',len(outputs['downloads_map']), '\n')
    
    outputs['merged_data'] = map_dict
    
    # Add georef values to folder
    for date in outputs['merged_data']:
        for sat in outputs['merged_data'][date]:
            im_path = outputs['merged_data'][date][sat]['toa_filepath'][0]
            data = gdal.Open(im_path, gdal.GA_ReadOnly)
            georef = np.array(data.GetGeoTransform())
            outputs['merged_data'][date][sat]['georef'] = georef
            



