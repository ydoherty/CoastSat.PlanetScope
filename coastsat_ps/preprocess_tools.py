# Tools for pre-processing functions
# YD, Sep 2020

import pathlib
import rasterio
import os
import subprocess
import numpy as np
import skimage.morphology as morphology
from shutil import copyfile

from coastsat_ps.shoreline_tools import classify_single

#from shapely.geometry import box
from arosics import COREG_LOCAL, DESHIFTER, COREG
from xml.dom import minidom

#%% General Functions

def create_folder(filepath):
    ''' Creates a filepath if it doesn't already exist
    Will not overwrite files that exist 
    Assign filepath string to a variable
    '''
    pathlib.Path(filepath).mkdir(exist_ok=True) 
    return filepath


#%% UDM load functions

def load_udm(udm_filename):
    '''Load single-band bit-encoded UDM as a 2D array
    
    Source: 
        https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/udm/udm.ipynb
        
    '''
    with rasterio.open(udm_filename, 'r') as src:
        udm = src.read()[0,...]
    return udm


def udm_to_mask(udm_array, bit_string):
    '''Create a mask from the udm, masking only pixels with RGB quality concerns
    
    Source: 
        https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/udm/udm.ipynb
        
    ''' 
    test_bits = int(bit_string,2)
    bit_matches = udm_array & test_bits # bit-wise logical AND operator (&)
    return bit_matches != 0 # mask any pixels that match test bits


#%% Mask manipulation functions

def save_mask(udm_filepath, save_path, bit_string, cloud_issue = False, nan_issue = False): 
    
    ''' Save extracted mask '''
    
    # open udm file
    with rasterio.open(udm_filepath, 'r') as src:
        udm = src.read()[0,...]
        kwargs = src.meta
    
    # Create mask
    mask = udm_to_mask(udm, bit_string)
    
    # Remove small elements from cloud mask (misdetections)
    if cloud_issue:
        # Remove cloud pixels that form  thin features. These are beach, building or swash pixels that are
        # erroneously identified as clouds by UDM cloud detection algorithm
        if sum(sum(mask)) > 0 and sum(sum(~mask)) > 0:
            # Remove long/thin mask elements (often WW)
            elem = morphology.square(25) # use a square of width 25 pixels (75m)
            mask = morphology.binary_opening(mask,elem) # perform image opening
            
            # remove objects with less than 25*75 connected pixels (~75m*225m = 16,875m^2)
            morphology.remove_small_objects(mask, min_size=25*75, connectivity=1, in_place=True)
      
    if nan_issue:
        # Remove nan pixels that form  thin features
        if sum(sum(mask)) > 0 and sum(sum(~mask)) > 0:
            # Remove long/thin mask elements
            elem = morphology.square(5) # use a square of width 5 pixels (15m)
            mask = morphology.binary_opening(mask,elem) # perform image opening
            
            # remove small objects 
            morphology.remove_small_objects(mask, min_size=500, connectivity=2, in_place=True)
            # remove 

    # save mask
    with rasterio.open(save_path, 'w', **kwargs) as dst:
            dst.write_band(1, mask.astype(rasterio.uint8))
     
        
def get_cloud_percentage_nan_cloud(nan_path, cloud_mask_path):
    # import nan and cloud masks
    nan_binary = load_udm(nan_path)
    cloud_binary = load_udm(cloud_mask_path)
    # calculate cover percentage
    cloud_perc = round(
                    np.sum(cloud_binary) /
                    ( cloud_binary.shape[1]*cloud_binary.shape[0] - np.sum(nan_binary) )
                    *100,2)
    
    return cloud_perc            
      

def get_file_extent(nan_path):
    nan_mask = load_udm(nan_path)
    aoi_area = nan_mask.shape[1]*nan_mask.shape[0]
    nan_area = np.sum(nan_mask)
    image_extent = 100-(nan_area/aoi_area)*100
    
    return image_extent


def zero_to_nan(im_path, nan_path, faulty_pixels = True, write = True):
    
    ''' Add zero values in image bands to nan mask '''
    
    # Open mask
    with rasterio.open(nan_path, 'r') as src:
        nan_mask = src.read()[0,...]
        kwargs = src.meta
    
    # Initialise update mask
    mask_out = np.zeros((nan_mask.shape[0], nan_mask.shape[1]), dtype = int)
    
    # for each band, find zero values and add to nan_mask
    for i in [1,2,3,4]:
        # open band
        with rasterio.open(im_path, 'r') as src1:
            im_band = src1.read(i)
            
        # boolean of band pixels with zero value
        im_zero = im_band == 0
        
        # update mask_out
        mask_out += im_zero
            
    # Update with faulty pixels if desierd
    if faulty_pixels:
        mask_out += nan_mask
        
    # Convert mask_out to boolean
    mask_out = mask_out>0
    
    # Overwrite original nan_mask with new nan_mask
    if write:
        with rasterio.open(nan_path, 'w', **kwargs) as dst:
            dst.write_band(1, mask_out.astype(rasterio.uint8))
    else:
        return mask_out


#%% GDAL subprocesses

def gdal_subprocess(settings, gdal_command_in, command_list):
    ''' Python access to all GDAL commands
        
    Inputs:
    
    settings - CoastSat.PlanetScope settings dictionary

    gdal_command_in -   name of command (may be an executable or .py file)
                            .py extension must be provided if present
                            
    command_list -      string list of commands (after GDAL function call)
    
    print_output -      write any text to print GDAL output information captured
                            by subprocess.check_output
       
        
    Unlike in terminal, full filepath of gdal command must be provided for subcommand
        ie. 'gdalsrsinfo' needs filepath.../gdalsrsinfo
    Filepath is hard coded in and may need to be changed for user
        
    Subprocess commands need to be given as a list of strings
           
    Function example for " gdalsrsinfo -o xml in_DEM.tif "
        gdal_subprocess('gdalsrsinfo', ["-o", "xml", "in_dem.tif"], 'yes')
    
    '''
    
    #gdal_loc = [os.path.join('/anaconda2/envs/coastsat/bin/', gdal_command_in)] #removed 30/5/2021
    gdal_loc = [os.path.join(settings['GDAL_location'], gdal_command_in)]
    
    gdal_command = gdal_loc + command_list
    gdal_output = subprocess.check_call(gdal_command)
    if gdal_output != 0:
        gdal_output = subprocess.check_output(gdal_command)
        print(gdal_output)


def merge_crop(settings, files_list, file_out_name, epsg_in = False, nan_mask = False):
    ''' Merge and crop list of images with gdalwarp
    Note - second file is output in areas of overlap'''
    
    if epsg_in == False:
        epsg_in = settings['output_epsg']
    
    if nan_mask == True:
        no_data_out_val = "1"
        no_data_in_val = "1"
    else:
        no_data_out_val = "0"
        no_data_in_val = "0"
    
    filepath_out = [os.path.join(settings['merge_out'], file_out_name)]
    
    command_list = ["-s_srs", epsg_in,
                "-t_srs", settings['output_epsg'],
                "-of", "GTiff",
                "-cutline",settings['aoi_geojson'],
                "-srcnodata", no_data_in_val, 
                "-dstnodata", no_data_out_val,
                "-crop_to_cutline",
                "-overwrite"]
    
    command_list = command_list + files_list + filepath_out

    # run proces (seconds)
    gdal_subprocess(settings, 'gdalwarp', command_list)
    
    
#%% Georectification functions using AROSICS

def get_raster_bounds(file):
    
    ''' Find raster bounding parameters '''
    
    dataset = rasterio.open(file)
    
    bounds = [
        dataset.bounds.left,
        dataset.bounds.top,
        dataset.bounds.right,
        dataset.bounds.bottom
        ]
    
    return bounds

# def get_raster_corners(file):
    
#     bound_im = rasterio.open(file)
#     bounds  = bound_im.bounds
#     geom = box(*bounds)
#     corners = str(geom.wkt)
    
#     return corners


def global_coreg(im_reference, im_target, im_out_global, 
                 land_mask_ref = None, land_mask_tgt = None, 
                 ws = (400,400),
                 q = True, progress = False, ignore_errors = True):
    
    # Global coregistration
    CR = COREG(im_reference, im_target, 
                path_out = im_out_global, fmt_out = 'GTiff',
                ws = ws,
                
                # Data mask settings
                mask_baddata_ref = land_mask_ref, 
                mask_baddata_tgt = land_mask_tgt,

                # Console output settings
                q = q, progress = progress, ignore_errors = ignore_errors, 

                # Hard coded settings
                nodata = (0,0),
                #align_grids = True,
                )
    
    #CR.calculate_spatial_shifts()
    #CR.show_matchWin()

    CR.correct_shifts()

    # Determine success
    coreg_success = CR.coreg_info['success']
            
    if (coreg_success == False): # or (CR.ssim_improved == False):
        copyfile(im_target, im_out_global)
        print('Coregistration failed, raw image copied instead')
        #coreg_success = False

    elif coreg_success:
        # Calculate image shift
        shift_m = np.sqrt(CR.coreg_info['corrected_shifts_map']['y']**2 + 
            CR.coreg_info['corrected_shifts_map']['x']**2)
        print('Shift of ' + str(round(shift_m,2)) + 'm')

    return CR.coreg_info, coreg_success


def local_coreg(im_reference, im_target, im_out_local, 
                land_mask_ref = None, land_mask_tgt = None, 
                grid_res = 100, window_size = (256,256), 
                min_points = 5,
                #footprint_poly_ref=None, footprint_poly_tgt=None,
                q = True, progress = False, ignore_errors = True,
                filter_level = 2):
        
    # Global coregistration
    CRL = COREG_LOCAL(im_reference,im_target,
                      path_out = im_out_local, fmt_out = 'GTiff',
                      grid_res = grid_res, window_size = window_size,
                      tieP_filter_level = filter_level, 

                      # Data mask settings
                      mask_baddata_ref = land_mask_ref, 
                      mask_baddata_tgt = land_mask_tgt,
                      
                      # Console output settings
                      q = q, progress = progress, ignore_errors = ignore_errors, 
                      
                      # Hard coded settings
                      nodata = (0,0),
                      min_reliability = 50,
                      #rs_max_outlier = 10,
                      #r_b4match = 1,
                      #s_b4match = 1,
                      #align_grids = True,
                      #footprint_poly_ref = footprint_poly_ref,
                      #footprint_poly_tgt = footprint_poly_tgt,
                      )

    #CRL.view_CoRegPoints()
    #CRL.tiepoint_grid.to_PointShapefile(path_out=im_out_local.replace('.tif','.shp'))
    #CRL.view_CoRegPoints_folium().save(im_out_local.replace('.tif','.html'))
    
    # Correct image
        # High min # points creates shift based on average of x/y shifts only
    CRL.correct_shifts(min_points_local_corr = min_points)
    
    # Determine success
    coreg_success = CRL.coreg_info['success']
    
    if coreg_success == False:
        copyfile(im_target, im_out_local)
        print('Coregistration failed, raw image copied instead')

    coreg_info_out = CRL.coreg_info
    
    return coreg_info_out, coreg_success


def mask_coreg(settings, im_target_mask, cr_param, mask_out_path, 
               min_points = 5, coreg_success = False, q = True, progress = False):
    
    if coreg_success == False:
        copyfile(im_target_mask, mask_out_path)
        
    else:
        if settings['im_coreg'] == 'Global Coreg':
            # Apply cr shift to mask 
            DESHIFTER(im_target_mask, cr_param, 
                      path_out =  mask_out_path, fmt_out = 'GTiff',
                      nodata = 255, # if 0 or 1 doesn't shoft properly
                      q = q, progress = progress
                      #align_grids = True,
                      ).correct_shifts()
        
        elif settings['im_coreg'] == 'Local Coreg':
            # Apply cr shift to mask 
            DESHIFTER(im_target_mask, cr_param, 
                      path_out =  mask_out_path, fmt_out = 'GTiff',
                      nodata = 255, # if 0 or 1 doesn't shoft properly
                      q = q, progress = progress,
                      min_points_local_corr = min_points
                      #align_grids = True,
                      ).correct_shifts()
            
        # Set no data vals as zero again
        with rasterio.open(mask_out_path, 'r') as src:
            mask = src.read()[0,...]
            kwargs = src.meta
        
        # Boolean of mask vals = 1
        mask = mask == 1
        
        # Overwrite original nan_mask with new nan_mask
        with rasterio.open(mask_out_path, 'w', **kwargs) as dst:
            dst.write_band(1, mask.astype(rasterio.uint8))
                
    
def create_land_mask(settings, toa_path, save_loc, nan_path = False, raw_mask = False):
    
    # Classify image
    if nan_path == False:
        im_classif = classify_single(settings['land_classifier_load'], 
                                                settings, toa_path, no_mask = True)
    else:
        if type(raw_mask) is bool:
            if raw_mask == False:
                im_classif = classify_single(settings['land_classifier_load'], 
                                                        settings, toa_path, no_mask = False)
            else:
                print('mask_comb needs to be an array or False')
        else:
            im_classif = classify_single(settings['land_classifier_load'], 
                                                    settings, toa_path, no_mask = False, 
                                                    raw_mask = raw_mask)
    
    # Extract mask of non-other pixels 
    other_mask = im_classif == 0
    
    # Smooth and remove small features
    elem = morphology.square(15) # use a square of width 10 pixels (30m)
    other_mask = morphology.binary_opening(other_mask,elem) # perform image opening
    other_mask = morphology.remove_small_objects(other_mask, 
                                        min_size=settings['min_beach_area_pixels'], 
                                        connectivity=1)
    
    # invert boolean and smooth again
    other_mask = other_mask == 0
    elem = morphology.square(10) # use a square of width 15 pixels (45m)
    other_mask = morphology.binary_opening(other_mask,elem) # perform image opening
    other_mask = morphology.remove_small_objects(other_mask, 
                                        min_size=settings['min_beach_area_pixels'], 
                                        connectivity=1)
    
    # Find geo kwargs
    if nan_path == False:
        with rasterio.open(toa_path, 'r') as src:
            kwargs = src.meta
        kwargs.update(
            dtype=rasterio.uint8,
            count = 1)
    else:
        with rasterio.open(nan_path) as src:
            kwargs = src.meta

    # save mask
    with rasterio.open(save_loc, 'w', **kwargs) as dst:
        dst.write_band(1, other_mask.astype(rasterio.uint8))


#%% XML file functions

def get_epsg(output_dict, date, raw_toa_filename):
    
    ''' Gets image epsg from xml file '''
    
    # Find corresponding xml file
    search_id = raw_toa_filename[9:20]
    for ii in range(len(output_dict['downloads_map'][date]['_metadata_clip.xml']['filenames'])):            
        if output_dict['downloads_map'][date]['_metadata_clip.xml']['filenames'][ii][9:20] == search_id:
            xml_path = output_dict['downloads_map'][date]['_metadata_clip.xml']['filepaths'][ii]

    # open xml file
    xmldoc = minidom.parse(xml_path)
    
    # find epsg
    epsg = xmldoc.getElementsByTagName("ps:epsgCode")[0].firstChild.data
    
    return epsg

def TOA_conversion(image_path, xml_path, save_path):
    
    ''' 
    1) Convert DN values to Top of Atmosphere (TOA)
    2) Add sensor type (PS2, PS2.SD, PSB.SD) to save filename
    
    Function modified from:
        https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/toar/toar_planetscope.ipynb 

    '''
    
    # Load image bands - note all PlanetScope 4-band images have band order BGRN
    with rasterio.open(image_path) as src:
        band_blue_radiance = src.read(1)
        
    with rasterio.open(image_path) as src:
        band_green_radiance = src.read(2)
    
    with rasterio.open(image_path) as src:
        band_red_radiance = src.read(3)
    
    with rasterio.open(image_path) as src:
        band_nir_radiance = src.read(4)
    
    
    ### Get TOA Factor ###
    xmldoc = minidom.parse(xml_path)
    nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")
    
    # XML parser refers to bands by numbers 1-4
    coeffs = {}
    for node in nodes:
        bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
        if bn in ['1', '2', '3', '4']:
            i = int(bn)
            value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
            coeffs[i] = float(value)
    
    #print("Conversion coefficients: {}".format(coeffs))  
    
    
    ### Convert to TOA ###
    
    # Multiply the Digital Number (DN) values in each band by the TOA reflectance coefficients
    band_blue_reflectance = band_blue_radiance * coeffs[1]
    band_green_reflectance = band_green_radiance * coeffs[2]
    band_red_reflectance = band_red_radiance * coeffs[3]
    band_nir_reflectance = band_nir_radiance * coeffs[4]
    
    #print("Red band radiance is from {} to {}".format(np.amin(band_red_radiance), np.amax(band_red_radiance)))
    #print("Red band reflectance is from {} to {}".format(np.amin(band_red_reflectance), np.amax(band_red_reflectance)))
    

    # find sensor name
    node = xmldoc.getElementsByTagName("eop:Instrument")
    sensor = node[0].getElementsByTagName("eop:shortName")[0].firstChild.data
    
    if sensor == 'PS2':
        save_path += '_PS2_TOA.tif'
    elif sensor == 'PS2.SD':
        save_path += '_2SD_TOA.tif'
    elif sensor == 'PSB.SD':
        save_path += '_BSD_TOA.tif'
    else:
        print('Error in detecting sensor name')
    
    ### Save output images ###
    
    # Set spatial characteristics of the output object to mirror the input
    kwargs = src.meta
    kwargs.update(
        dtype=rasterio.uint16,
        count = 4)
    
    #print("Before Scaling, red band reflectance is from {} to {}".format(np.amin(band_red_reflectance), np.amax(band_red_reflectance)))
    # Here we include a fixed scaling factor. This is common practice.
    scale = 10000
    blue_ref_scaled = scale * band_blue_reflectance
    green_ref_scaled = scale * band_green_reflectance
    red_ref_scaled = scale * band_red_reflectance
    nir_ref_scaled = scale * band_nir_reflectance
    
    #print("After Scaling, red band reflectance is from {} to {}".format(np.amin(red_ref_scaled), np.amax(red_ref_scaled)))
        
    # Write band calculations to a new raster file
    with rasterio.open(save_path, 'w', **kwargs) as dst:
            dst.write_band(1, blue_ref_scaled.astype(rasterio.uint16))
            dst.write_band(2, green_ref_scaled.astype(rasterio.uint16))
            dst.write_band(3, red_ref_scaled.astype(rasterio.uint16))
            dst.write_band(4, nir_ref_scaled.astype(rasterio.uint16))
    
