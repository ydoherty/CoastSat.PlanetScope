# Shoreline extraction function tools

import rasterio
import os
import numpy as np
import pandas as pd
import skimage.filters as filters
import skimage.morphology as morphology
import skimage.measure as measure
import geopandas as gpd

from astropy.convolution import convolve
from shapely.geometry import LineString
from shapely import geometry


from coastsat_ps.interactive import (convert_epsg, convert_world2pix, convert_pix2world, 
                                    get_ps_no_mask, get_ps_data, get_im_ms)


#%% Index calculations

def calc_water_index(toa_path, settings, save_name):
    
    # Extract water index vals
    band1_no = settings['water_index_list'][0]
    band2_no = settings['water_index_list'][1]    
    norm_bool = settings['water_index_list'][2]
    file_end = settings['water_index_list'][3]
    
    # import bands
    with rasterio.open(toa_path) as src:
        band_1_reflectance = src.read(band1_no)
        band_1 = band_1_reflectance/10000
        
    with rasterio.open(toa_path) as src:
        band_2_reflectance = src.read(band2_no)
        band_2 = band_2_reflectance/10000
    
    band_1[band_1 == 0] = np.nan
    band_2[band_2 == 0] = np.nan

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
        
    # perform calculation
    if norm_bool:
        water_index = (band_1.astype(float) - band_2.astype(float)) / (band_1 + band_2)
    else:
        water_index = (band_1.astype(float) - band_2.astype(float))

    # Set spatial characteristics of the output object to mirror the input
    kwargs = src.meta
    kwargs.update(
        dtype=rasterio.float32,
        count = 1)
    
    # Save image to new file
    save_file = os.path.join(settings['index_tif_out'], save_name + file_end)
    
    with rasterio.open(save_file, 'w', **kwargs) as dst:
            dst.write_band(1, water_index.astype(rasterio.float32))


#%% Thresholding

def peak_fraction_thresh(vec_sand, vec_water, thresh_fraction, nBins):
    
    # Find peak water val
    water_vals = pd.DataFrame()
    water_vals['count'] = np.histogram(vec_water, bins = nBins)[0]
    water_vals['ind_vals'] = np.histogram(vec_water, bins = nBins)[1][0:-1]
    water_max = water_vals.loc[water_vals['count'].idxmax(), 'ind_vals']
    
    # Find peak sand val
    sand_vals = pd.DataFrame()
    sand_vals['count'] = np.histogram(vec_sand, bins = nBins)[0]
    sand_vals['ind_vals'] = np.histogram(vec_sand, bins = nBins)[1][0:-1]
    sand_max = sand_vals.loc[sand_vals['count'].idxmax(), 'ind_vals']
    
    # Create threshold
    threshold = water_max + thresh_fraction*(sand_max-water_max)

    return threshold


def peak_fraction_generic(vec, thresh_fraction, nBins, multi_otsu = True):
    
    histogram_vals = pd.DataFrame()
    histogram_vals['count'] = np.histogram(vec, bins = nBins)[0]
    histogram_vals['ind_vals'] = np.histogram(vec, bins = nBins)[1][0:-1]
    
    if multi_otsu:
        t_otsu_all = filters.threshold_multiotsu(vec, nbins = nBins)
        mid_val = t_otsu_all[1]
    else:
        mid_val = filters.threshold_otsu(vec)
    
    vec_sand_half = histogram_vals[histogram_vals['ind_vals']>mid_val]
    sand_max = histogram_vals.loc[vec_sand_half['count'].idxmax(), 'ind_vals']
    
    vec_water_half = histogram_vals[histogram_vals['ind_vals']<mid_val]
    water_max = histogram_vals.loc[vec_water_half['count'].idxmax(), 'ind_vals']
    
    threshold = water_max + thresh_fraction*(sand_max-water_max)

    return threshold



#%% CoastSat data match functions for inputs

def classify_single(classifier, settings, toa_path, no_mask = True, raw_mask = False):

    # Extract image data and cloud mask
    if no_mask:
        im_ms, im_mask = get_ps_no_mask(toa_path)
    else:
        if type(raw_mask) is bool:
            if raw_mask == False:
                im_ms, im_mask = get_ps_data(toa_path)
            else:
                print('mask_comb needs to be an array or False')
        else:    
            im_ms = get_im_ms(toa_path)
            im_mask = raw_mask
    
    # classify image in 4 classes (sand, whitewater, water, other) with NN classifier
    im_classif = classify_image_NN(im_ms, im_mask, classifier)
    
    return im_classif


#%% Modified CoastSat Functions - SDS shoreline

def calculate_features(im_ms, cloud_mask, im_bool):
    """
    Calculates features on the image that are used for the supervised classification. 
    The features include spectral normalized-difference indices and standard 
    deviation of the image for all the bands and indices.

    KV WRL 2018
    
    Modified for PS data by YD 2020

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_bool: np.array
        2D array of boolean indicating where on the image to calculate the features

    Returns:    
    -----------
    features: np.array
        matrix containing each feature (columns) calculated for all
        the pixels (rows) indicated in im_bool
        
    """

    # add all the multispectral bands
    features = np.expand_dims(im_ms[im_bool,0],axis=1)
    for k in range(1,im_ms.shape[2]):
        feature = np.expand_dims(im_ms[im_bool,k],axis=1)
        features = np.append(features, feature, axis=-1)
        
    # NIR-G
    im_NIRG = nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRG[im_bool],axis=1), axis=-1)
    
    # NIR-B
    im_NIRB = nd_index(im_ms[:,:,3], im_ms[:,:,0], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRB[im_bool],axis=1), axis=-1)
    
    # NIR-R
    im_NIRR = nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRR[im_bool],axis=1), axis=-1)
        
    # B-R
    im_BR = nd_index(im_ms[:,:,0], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_BR[im_bool],axis=1), axis=-1)
    
    # calculate standard deviation of individual bands
    for k in range(im_ms.shape[2]):
        im_std =  image_std(im_ms[:,:,k], 2)
        features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
        
    # calculate standard deviation of the spectral indices
    im_std = image_std(im_NIRG, 2)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = image_std(im_NIRB, 2)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = image_std(im_NIRR, 2)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = image_std(im_BR, 2)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)

    return features


def classify_image_NN(im_ms, cloud_mask, clf):
    """
    Classifies every pixel in the image in one of 4 classes:
        - sand                                          --> label = 1
        - whitewater (breaking waves and swash)         --> label = 2
        - water                                         --> label = 3
        - other (vegetation, buildings, rocks...)       --> label = 0

    The classifier is a Neural Network that is already trained.

    KV WRL 2018
    
    Modified YD 2020

    Arguments:
    -----------
    im_ms: np.array
        Pansharpened RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    clf: joblib object
        pre-trained classifier

    Returns:    
    -----------
    im_classif: np.array
        2D image containing labels
            Pixel Values:
                0) Other, 1) Sand, 2) Whitewater, 3) Water
    
    """

    # calculate features
    vec_features = calculate_features(im_ms, cloud_mask, np.ones(cloud_mask.shape).astype(bool))
    vec_features[np.isnan(vec_features)] = 1e-9 # NaN values are create when std is too close to 0

    # remove NaNs and cloudy pixels
    vec_cloud = cloud_mask.reshape(cloud_mask.shape[0]*cloud_mask.shape[1])
    vec_nan = np.any(np.isnan(vec_features), axis=1)
    vec_mask = np.logical_or(vec_cloud, vec_nan)
    vec_features = vec_features[~vec_mask, :]

    # classify pixels
    labels = clf.predict(vec_features)

    # recompose image
    vec_classif = np.nan*np.ones((cloud_mask.shape[0]*cloud_mask.shape[1]))
    vec_classif[~vec_mask] = labels
    im_classif = vec_classif.reshape((cloud_mask.shape[0], cloud_mask.shape[1]))

    return im_classif


def create_shoreline_buffer(im_shape, georef, image_epsg, pixel_size, settings):
    """
    Creates a buffer around the reference shoreline. The size of the buffer is 
    given by settings['max_dist_ref'].

    KV WRL 2018
    
    Modified for PS by YD 2020

    Arguments:
    -----------
    im_shape: np.array
        size of the image (rows,columns)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    pixel_size: int
        size of the pixel in metres (15 for Landsat, 10 for Sentinel-2)
    settings: dict with the following keys
        'output_epsg': int
            output spatial reference system
        'reference_shoreline': np.array
            coordinates of the reference shoreline
        'max_dist_ref': int
            maximum distance from the reference shoreline in metres

    Returns:    
    -----------
    im_buffer: np.array
        binary image, True where the buffer is, False otherwise

    """
    # initialise the image buffer
    im_buffer = np.ones(im_shape).astype(bool)

    # convert reference shoreline to pixel coordinates
    ref_sl = settings['reference_shoreline']
    out_epsg_int = int(settings['output_epsg'].replace('EPSG:',''))
    ref_sl_conv = convert_epsg(ref_sl, out_epsg_int, image_epsg)[:,:-1]
    ref_sl_pix = convert_world2pix(ref_sl_conv, georef)
    ref_sl_pix_rounded = np.round(ref_sl_pix).astype(int)

    # make sure that the pixel coordinates of the reference shoreline are inside the image
    idx_row = np.logical_and(ref_sl_pix_rounded[:,0] > 0, ref_sl_pix_rounded[:,0] < im_shape[1])
    idx_col = np.logical_and(ref_sl_pix_rounded[:,1] > 0, ref_sl_pix_rounded[:,1] < im_shape[0])
    idx_inside = np.logical_and(idx_row, idx_col)
    ref_sl_pix_rounded = ref_sl_pix_rounded[idx_inside,:]

    # create binary image of the reference shoreline (1 where the shoreline is 0 otherwise)
    im_binary = np.zeros(im_shape)
    for j in range(len(ref_sl_pix_rounded)):
        im_binary[ref_sl_pix_rounded[j,1], ref_sl_pix_rounded[j,0]] = 1
    im_binary = im_binary.astype(bool)

    # dilate the binary image to create a buffer around the reference shoreline
    max_dist_ref_pixels = np.ceil(settings['max_dist_ref']/pixel_size)
    se = morphology.disk(max_dist_ref_pixels)
    im_buffer = morphology.binary_dilation(im_binary, se)

    # Invert boolean (True is outside region)
    im_buffer = im_buffer == False

    return im_buffer


#%% Contour extraction functions

def sl_extract(masked_im, sand_mask, water_mask, masked_im_gen, settings):
    
    # Extract data
    thresh_type = settings['thresholding']
    n_bins = settings['otsu_hist_bins']
    thresh_fraction = settings['peak_fraction']
        
    # Create sand vec 
    sand_im = np.copy(masked_im)
    sand_im[sand_mask] = np.nan

    # Create water vec 
    water_im = np.copy(masked_im)
    water_im[water_mask] = np.nan
   
    # Reshape to 1D array and remove any nan values
    vec_sand = sand_im.reshape(sand_im.shape[0] * sand_im.shape[1])
    vec_water = water_im.reshape(water_im.shape[0] * water_im.shape[1]) 

    # Remove nans from vecs
    vec_sand = vec_sand[~np.isnan(vec_sand)]
    vec_water = vec_water[~np.isnan(vec_water)]

    # Combine raw vecs
    int_all_raw = np.append(vec_water,vec_sand, axis=0)

    # make sure both classes have the same number of pixels before thresholding
    if len(vec_water) > 0 and len(vec_sand) > 0:
        if np.argmin([vec_sand.shape[0],vec_water.shape[0]]) == 1:
            vec_sand = vec_sand[np.random.choice(vec_sand.shape[0],vec_water.shape[0], replace=False)]
        else:
            vec_water = vec_water[np.random.choice(vec_water.shape[0],vec_sand.shape[0], replace=False)]

    # Threshold image
    if thresh_type == 'Otsu':
        # Combine vecs
        int_all = np.append(vec_water,vec_sand, axis=0)
        # Threshold
        t_otsu = filters.threshold_otsu(int_all)
        
    elif thresh_type == 'Peak Fraction':
        t_otsu = peak_fraction_thresh(vec_sand, vec_water, thresh_fraction, n_bins)

    # Contour image
    contours = measure.find_contours(masked_im_gen, t_otsu)
    
    # remove contours that contain NaNs
    contours_nonans = []
    for k in range(len(contours)):
        if np.any(np.isnan(contours[k])):
            index_nan = np.where(np.isnan(contours[k]))[0]
            contours_temp = np.delete(contours[k], index_nan, axis=0)
            if len(contours_temp) > 1:
                contours_nonans.append(contours_temp)
        else:
            contours_nonans.append(contours[k])
    contours = contours_nonans
    
    return contours, int_all_raw, t_otsu



def sl_extract_generic(masked_im, settings):
    
    # Extract data
    thresh_type = settings['thresholding']
    n_bins = settings['otsu_hist_bins']
    thresh_fraction = settings['peak_fraction']
        
    # Create vec 
    vec_im = np.copy(masked_im)
    vec = vec_im.reshape(vec_im.shape[0] * vec_im.shape[1])
    vec = vec[~np.isnan(vec)]

    # Threshold image
    if thresh_type == 'Otsu':
        # Threshold
        t_otsu = filters.threshold_otsu(vec)
        
    elif thresh_type == 'Peak Fraction':
        t_otsu = peak_fraction_generic(vec, thresh_fraction, n_bins)

    # Contour image
    contours = measure.find_contours(masked_im, t_otsu)
    
    # remove contours that contain NaNs
    contours_nonans = []
    for k in range(len(contours)):
        if np.any(np.isnan(contours[k])):
            index_nan = np.where(np.isnan(contours[k]))[0]
            contours_temp = np.delete(contours[k], index_nan, axis=0)
            if len(contours_temp) > 1:
                contours_nonans.append(contours_temp)
        else:
            contours_nonans.append(contours[k])
    contours = contours_nonans
    
    return contours, vec, t_otsu



# def cs_sl_extract(index_im, im_ref_buffer, comb_mask, im_classif, settings):
    
#     # Extract data
#     nrows = comb_mask.shape[0]
#     ncols = comb_mask.shape[1]
#     sand_im = im_classif == 1
#     water_im = im_classif == 3
    
#     # Nan index im
#     index = np.copy(index_im)
#     index[comb_mask] = np.nan

#     # Reshape to vec
#     vec_ind = index.reshape(nrows*ncols,1)
#     vec_sand = sand_im.reshape(ncols*nrows)
#     vec_water = water_im.reshape(ncols*nrows)    
    
#     # create a buffer around the sandy beach
#     se = morphology.disk(settings['buffer_size_pixels'])
#     im_buffer = morphology.binary_dilation(sand_im, se)
#     vec_buffer = im_buffer.reshape(nrows*ncols)

#     # select water/sand/swash pixels that are within the buffer
#     int_water = vec_ind[np.logical_and(vec_buffer,vec_water),:]
#     int_sand = vec_ind[np.logical_and(vec_buffer,vec_sand),:]

    # # make sure both classes have the same number of pixels before thresholding
    # if len(int_water) > 0 and len(int_sand) > 0:
    #     if np.argmin([int_sand.shape[0],int_water.shape[0]]) == 1:
    #         int_sand = int_sand[np.random.choice(int_sand.shape[0],int_water.shape[0], replace=False),:]
    #     else:
    #         int_water = int_water[np.random.choice(int_water.shape[0],int_sand.shape[0], replace=False),:]

    # # threshold the sand/water intensities
    # int_all = np.append(int_water,int_sand, axis=0)
    # t_wi = filters.threshold_otsu(int_all[:,0])

#     # find contour with MS algorithm
#     im_wi_buffer = np.copy(index)
#     im_wi_buffer[im_ref_buffer>0] = np.nan
    
#     contours = measure.find_contours(im_wi_buffer, t_wi)

#     # remove contour points that are NaNs (around clouds)
#     contours_nonans = []
#     for k in range(len(contours)):
#         if np.any(np.isnan(contours[k])):
#             index_nan = np.where(np.isnan(contours[k]))[0]
#             contours_temp = np.delete(contours[k], index_nan, axis=0)
#             if len(contours_temp) > 1:
#                 contours_nonans.append(contours_temp)
#         else:
#             contours_nonans.append(contours[k])
#     contours = contours_nonans

#     return contours



#%% Modified CoastSat Functions - SDS Tools

def nd_index(im1, im2, cloud_mask):
    """
    Computes normalised difference index on 2 images (2D), given a cloud mask (2D).

    KV WRL 2018

    Arguments:
    -----------
    im1: np.array
        first image (2D) with which to calculate the ND index
    im2: np.array
        second image (2D) with which to calculate the ND index
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are

    Returns:    
    -----------
    im_nd: np.array
        Image (2D) containing the ND index
        
    """

    # reshape the cloud mask
    vec_mask = cloud_mask.reshape(im1.shape[0] * im1.shape[1])
    # initialise with NaNs
    vec_nd = np.ones(len(vec_mask)) * np.nan
    # reshape the two images
    vec1 = im1.reshape(im1.shape[0] * im1.shape[1])
    vec2 = im2.reshape(im2.shape[0] * im2.shape[1])
    # compute the normalised difference index
    temp = np.divide(vec1[~vec_mask] - vec2[~vec_mask],
                     vec1[~vec_mask] + vec2[~vec_mask])
    vec_nd[~vec_mask] = temp
    # reshape into image
    im_nd = vec_nd.reshape(im1.shape[0], im1.shape[1])

    return im_nd


def image_std(image, radius):
    """
    Calculates the standard deviation of an image, using a moving window of 
    specified radius. Uses astropy's convolution library'
    
    Copied with permission from CoastSat (KV, 2020) 
        https://github.com/kvos/CoastSat
    
    Arguments:
    -----------
    image: np.array
        2D array containing the pixel intensities of a single-band image
    radius: int
        radius defining the moving window used to calculate the standard deviation. 
        For example, radius = 1 will produce a 3x3 moving window.
        
    Returns:    
    -----------
    win_std: np.array
        2D array containing the standard deviation of the image
        
    """  
    
    # convert to float
    image = image.astype(float)
    # first pad the image
    image_padded = np.pad(image, radius, 'reflect')
    # window size
    win_rows, win_cols = radius*2 + 1, radius*2 + 1
    # calculate std with uniform filters
    win_mean = convolve(image_padded, np.ones((win_rows,win_cols)), boundary='extend',
                        normalize_kernel=True, nan_treatment='interpolate', preserve_nan=True)
    win_sqr_mean = convolve(image_padded**2, np.ones((win_rows,win_cols)), boundary='extend',
                        normalize_kernel=True, nan_treatment='interpolate', preserve_nan=True)
    win_var = win_sqr_mean - win_mean**2
    win_std = np.sqrt(win_var)
    # remove padding
    win_std = win_std[radius:-radius, radius:-radius]

    return win_std



def process_shoreline(contours, cloud_mask, georef, image_epsg, settings):
    """
    Converts the contours from image coordinates to world coordinates. 
    This function also removes the contours that are too small to be a shoreline 
    (based on the parameter settings['min_length_sl'])

    KV WRL 2018
    
    Modified YD 2020

    Arguments:
    -----------
    contours: np.array or list of np.array
        image contours as detected by the function find_contours
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    settings: dict with the following keys
        'output_epsg': int
            output spatial reference system
        'min_length_sl': float
            minimum length of shoreline contour to be kept (in meters)

    Returns:
    -----------
    shoreline: np.array
        array of points with the X and Y coordinates of the shoreline

    """
    # convert epsg
    out_epsg = int(settings['output_epsg'].replace('EPSG:',''))
    # convert pixel coordinates to world coordinates
    contours_world = convert_pix2world(contours, georef)
    # convert world coordinates to desired spatial reference system
    contours_epsg = convert_epsg(contours_world, image_epsg, out_epsg)
    # remove contours that have a perimeter < min_length_sl (provided in settings dict)
    # this enables to remove the very small contours that do not correspond to the shoreline
    contours_long = []
    for l, wl in enumerate(contours_epsg):
        coords = [(wl[k,0], wl[k,1]) for k in range(len(wl))]
        a = LineString(coords) # shapely LineString structure
        if a.length >= settings['min_length_sl']:
            contours_long.append(wl)
    # format points into np.array
    x_points = np.array([])
    y_points = np.array([])
    for k in range(len(contours_long)):
        x_points = np.append(x_points,contours_long[k][:,0])
        y_points = np.append(y_points,contours_long[k][:,1])
    contours_array = np.transpose(np.array([x_points,y_points]))

    shoreline = contours_array

    # now remove any shoreline points that are attached to cloud pixels
    if sum(sum(cloud_mask)) > 0:
        # get the coordinates of the cloud pixels
        idx_cloud = np.where(cloud_mask)
        idx_cloud = np.array([(idx_cloud[0][k], idx_cloud[1][k]) for k in range(len(idx_cloud[0]))])
        # convert to world coordinates and same epsg as the shoreline points
        coords_cloud = convert_epsg(convert_pix2world(idx_cloud, georef),
                                               image_epsg, out_epsg)[:,:-1]
        # only keep the shoreline points that are at least _m from any cloud pixel
        idx_keep = np.ones(len(shoreline)).astype(bool)
        for k in range(len(shoreline)):
            if np.any(np.linalg.norm(shoreline[k,:] - coords_cloud, axis=1) < settings['cloud_buffer']):
                idx_keep[k] = False
        shoreline = shoreline[idx_keep]

    return shoreline


def output_to_gdf_PL(output):
    """
    Modified from KV WRL 2018 by YD
        Adapted to PL metadata info
    
    Saves the mapped shorelines as a gpd.GeoDataFrame    
    
    KV WRL 2018
    
    Modified YD 2020

    Arguments:
    -----------
    output: dict
        contains the coordinates of the mapped shorelines + attributes          
                
    Returns:    
    -----------
    gdf_all: gpd.GeoDataFrame
        contains the shorelines + attirbutes
  
    """    
     
    # loop through the mapped shorelines
    counter = 0
    for i in range(len(output['shorelines'])):
        # skip if there shoreline is empty 
        if len(output['shorelines'][i]) == 0:
            continue
        else:
            # save the geometry + attributes
            geom = geometry.LineString(output['shorelines'][i])
            gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
            gdf.index = [i]
            gdf.loc[i,'timestamp'] = output['timestamp utc'][i].strftime('%Y-%m-%d %H:%M:%S')
            # gdf.loc[i,'satname'] = output['satname'][i]
            #gdf.loc[i,'geoaccuracy'] = output['geoaccuracy'][i]
            gdf.loc[i,'cloud_cover'] = output['cloud_cover'][i]
            gdf.loc[i,'aoi_coverage'] = output['aoi_cover'][i]
            gdf.loc[i,'ps_sat_name'] = output['ps_sat_name'][i]
            gdf.loc[i,'name'] = output['name'][i]

            # store into geodataframe
            if counter == 0:
                gdf_all = gdf
            else:
                gdf_all = gdf_all.append(gdf)
            counter = counter + 1
            
    return gdf_all
