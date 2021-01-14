# interactive window functions

import os
import pickle
import rasterio

import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as exposure
import skimage.transform as transform
import geopandas as gpd
import matplotlib.image as mpimg
import pandas as pd

from pylab import ginput
from osgeo import gdal, osr
from shapely import geometry
from shutil import copyfile


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#%%

def get_im_ms(fn):
    # Generate im_ms
    data = gdal.Open(fn, gdal.GA_ReadOnly)
    bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
    im_ms = np.stack(bands, 2)
    return im_ms



def find_mask(fn, mask_name_end, raw_folder = False):
    
    if raw_folder:
        # Find cloud mask
        mask_path = fn.replace('_TOA.tif', '')
        # Remove sensor naming
        name_add = 'DN_udm_clip_' + mask_name_end
        mask_path = mask_path.replace('clip_PS2', name_add)
        mask_path = mask_path.replace('clip_2SD', name_add)
        mask_path = mask_path.replace('clip_BSD', name_add)
        
    else:
        # Find mask
        mask_path = fn.replace('TOA.tif', mask_name_end)
        # Remove sensor naming
        mask_path = mask_path.replace(' PS2', '')
        mask_path = mask_path.replace(' 2SD', '')
        mask_path = mask_path.replace(' BSD', '')
        
    with rasterio.open(mask_path) as src:
        mask_bool = src.read(1)
        
    return mask_bool


def create_comb_mask(fn):
    # Get cloud and nan masks
    cloud_mask = find_mask(fn, 'cloud_mask.tif')
    nan_mask = find_mask(fn, 'NaN_mask.tif')
            
    # merge masks
    mask_out = (cloud_mask + nan_mask) != 0
    
    return mask_out


def get_ps_data(fn):
    
    ''' Function to emulate CoastSat 'SDS_preprocess.preprocess_single' for PS
            Only return necessary variables for 'SDS_classify.label_images'
    
    fn is filepath to a single TOA image
    
    Returns:
    -----------
    im_ms: np.array
        3D array containing the TOA bands (0:B, 1:G, 2:R, 3:NIR)
        
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    
    '''
    
    # Generate im_ms
    im_ms = get_im_ms(fn)
    
    # Get masks and combine
    mask_out = create_comb_mask(fn)

    return im_ms, mask_out


def get_ps_no_mask(toa_path):
    
    # Generate im_ms
    im_ms = get_im_ms(toa_path)
    
    # Create fake im_mask
    with rasterio.open(toa_path, 'r') as src1:
        im_band = src1.read(1)
    im_mask = np.zeros((im_band.shape[0], im_band.shape[1]), dtype = int)
    
    # for each band, find zero values and add to nan_mask
    for i in [1,2,3,4]:
        # open band
        with rasterio.open(toa_path, 'r') as src1:
            im_band = src1.read(i)
            
        # boolean of band pixels with zero value
        im_zero = im_band == 0
        
        # update mask_out
        im_mask += im_zero
    
    # Convert mask_out to boolean
    im_mask = im_mask>0   
    
    return im_ms, im_mask


#%% TOA selection

def ref_im_select(settings):
    
    ''' Select suitable raw TOA image '''
    
    # Set file search directory
    filepath = settings['raw_data']
    im_selected = False
    
    # Extract filenames
    filenames = []
    for file in os.listdir(filepath):
        if file[-7:] == 'TOA.tif':
            filenames += [file]
    filenames.sort(reverse = True)
    
    # create figure
    fig, ax = plt.subplots(1,1, figsize=[18,9], tight_layout=True)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    
    nan_max = 0
    
    # loop through the images
    for i in range(len(filenames)):

        # read image
        fn = os.path.join(filepath, filenames[i])
        im_ms, zero_mask = get_ps_no_mask(fn)

        # Extract cloud mask
        cloud_mask = find_mask(fn, 'cloud_mask.tif', raw_folder = True)
        nan_mask = find_mask(fn, 'NaN_mask.tif', raw_folder = True)
        
        # Remove images with small aoi
            # Remembers the min nan so far as a limit
        if nan_max == 0:
            nan_max = np.sum(im_ms[:,:,0] == 0)
        if np.sum(im_ms[:,:,0] == 0) > nan_max or np.sum(nan_mask) > 0:
            continue
        elif np.sum(im_ms[:,:,0] == 0) < nan_max:
            nan_max = np.sum(im_ms[:,:,0] == 0)
            
        # calculate cloud cover
        cloud_cover = 100*np.divide(sum(sum(cloud_mask.astype(int))),
                                (cloud_mask.shape[0]*cloud_mask.shape[1]))

        # skip image if cloud cover is above threshold
        if cloud_cover > settings['cloud_threshold']:
            continue

        # Merge nan and cloud masks
        mask_comb = (zero_mask + nan_mask + cloud_mask) != 0
        
        # rescale image intensity for display purposes
        im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], mask_comb, 99.9)

        # plot the image RGB on a figure
        ax.axis('off')
        ax.imshow(im_RGB)

        # decide if the image if good enough for digitizing the shoreline
        ax.set_title('Press <right arrow> to accept scene as reference image\n' +
                  'If the image is cloudy press <left arrow> to get another image\n' + 
                  #'Image ' + str(i+1) + ' of ' + str(len(filenames)) + '\n' +
                  filenames[i],
                  fontsize=12)
        
        # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
        # this variable needs to be immuatable so we can access it after the keypress event
        skip_image = False
        key_event = {}
        def press(event):
            # store what key was pressed in the dictionary
            key_event['pressed'] = event.key
        # let the user press a key, right arrow to keep the image, left arrow to skip it
        # to break the loop the user can press 'escape'
        while True:
            btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            plt.draw()
            fig.canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress()
            # after button is pressed, remove the buttons
            btn_skip.remove()
            btn_keep.remove()
            btn_esc.remove()
            # keep/skip image according to the pressed key, 'escape' to break the loop
            if key_event.get('pressed') == 'right':
                skip_image = False
                break
            elif key_event.get('pressed') == 'left':
                skip_image = True
                break
            elif key_event.get('pressed') == 'escape':
                plt.close()
                raise StopIteration('User cancelled reference image selection')
            else:
                plt.waitforbuttonpress()
                    
        if skip_image:
            ax.clear()
            continue
        else:
            # copy selected image to user inputs folder
            file_out = fn
            copyfile(file_out, settings['georef_im_path'])
            print('Selected reference image copied to:\n' + settings['georef_im_path'])
            plt.close()
            im_selected = True
            break

    # check if an image sleected
    if im_selected == False:
        plt.close()
        raise Exception('No further images available for use as reference image')



def merge_im_select(settings):
    
    ''' Select suitable TOA image '''
    
    # Set file search directory
    filepath = settings['merge_out']
    im_selected = False
    
    # Extract filenames
    filenames = []
    for file in os.listdir(filepath):
        if file[-7:] == 'TOA.tif':
            filenames += [file]
    filenames.sort(reverse = True)
    
    # create figure
    fig, ax = plt.subplots(1,1, figsize=[18,9], tight_layout=True)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    
    # loop through the images
    for i in range(len(filenames)):

        # read image
        fn = os.path.join(filepath, filenames[i])
        im_ms, mask_comb = get_ps_data(fn)

        # Extract cloud mask
        cloud_mask = find_mask(fn, 'cloud_mask.tif', raw_folder = False)
        
        # calculate cloud cover
        cloud_cover = 100*np.divide(sum(sum(cloud_mask.astype(int))),
                                (cloud_mask.shape[0]*cloud_mask.shape[1]))

        # skip image if cloud cover is above threshold
        if cloud_cover > settings['cloud_threshold']:
            continue

        # rescale image intensity for display purposes
        im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], mask_comb, 99.9)

        # plot the image RGB on a figure
        ax.axis('off')
        ax.imshow(im_RGB)

        # decide if the image if good enough for digitizing the shoreline
        ax.set_title('Press <right arrow> to accept scene as reference image\n' +
                  'If the image is cloudy press <left arrow> to get another image\n' + 
                  #'Image ' + str(i+1) + ' of ' + str(len(filenames)) + '\n' +
                  filenames[i],
                  fontsize=12)
        
        # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
        # this variable needs to be immuatable so we can access it after the keypress event
        skip_image = False
        key_event = {}
        def press(event):
            # store what key was pressed in the dictionary
            key_event['pressed'] = event.key
        # let the user press a key, right arrow to keep the image, left arrow to skip it
        # to break the loop the user can press 'escape'
        while True:
            btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            plt.draw()
            fig.canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress()
            # after button is pressed, remove the buttons
            btn_skip.remove()
            btn_keep.remove()
            btn_esc.remove()
            # keep/skip image according to the pressed key, 'escape' to break the loop
            if key_event.get('pressed') == 'right':
                skip_image = False
                break
            elif key_event.get('pressed') == 'left':
                skip_image = True
                break
            elif key_event.get('pressed') == 'escape':
                plt.close()
                raise StopIteration('User cancelled reference image selection')
            else:
                plt.waitforbuttonpress()
                    
        if skip_image:
            ax.clear()
            continue
        else:
            # copy selected image to user inputs folder
            settings['ref_merge_im'] = fn
            print('Merged reference image selection saved:\n' + settings['ref_merge_im'])
            
            # Save a txt file of selection
            with open(settings['ref_merge_im_txt'], "w") as text_file:
                text_file.write(settings['ref_merge_im'])
                
            plt.close()
            
            im_selected = True
            
            break

    # check if an image selected
    if im_selected == False:
        plt.close()
        raise Exception('No further images available for use as reference image')




#%% Ref SL Digitisation

def get_reference_sl(settings):

    # unpack settings
    sitename = settings['site_name']
    pts_coords = []
    
    # check if reference shoreline already exists in the corresponding folder
    filename = sitename + '_reference_shoreline.pkl'
    # if it exist, load it and return it
    if filename in os.listdir(settings['run_input_folder']):
        print('Previous reference shoreline loaded')
        settings['reference_shoreline'] = filename
        with open(os.path.join(settings['run_input_folder'], filename), 'rb') as f:
            settings['reference_shoreline'] = pickle.load(f)
        return
    
    # otherwise get the user to manually digitise a shoreline
    else:
        # Digitise shoreline on im_ref
        fn = settings['ref_merge_im']
                            
        # create figure
        fig, ax = plt.subplots(1,1, figsize=[18,9], tight_layout=True)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

        # extract data and masks
        im_ms, nan_mask = get_ps_data(fn)
        
        # rescale image intensity for display purposes
        im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], nan_mask, 99.9)

        # plot the image RGB on a figure
        ax.axis('off')
        ax.imshow(im_RGB)


        # create buttons
        add_button = plt.text(0, 0.9, 'add', size=16, ha="left", va="top",
                               transform=plt.gca().transAxes,
                               bbox=dict(boxstyle="square", ec='k',fc='w'))
        end_button = plt.text(1, 0.9, 'end', size=16, ha="right", va="top",
                               transform=plt.gca().transAxes,
                               bbox=dict(boxstyle="square", ec='k',fc='w'))
        # add multiple reference shorelines (until user clicks on <end> button)
        pts_sl = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
        geoms = []
        while 1:
            add_button.set_visible(False)
            end_button.set_visible(False)
            # update title (instructions)
            ax.set_title('Click points along the shoreline (enough points to capture the beach curvature).\n' +
                      'Start at one end of the beach.\n' + 'When finished digitizing, click <ENTER>',
                      fontsize=14)
            plt.draw()

            # let user click on the shoreline
            pts = ginput(n=50000, timeout=-1, show_clicks=True)
            pts_pix = np.array(pts)
            # get georef val
            data = gdal.Open(fn, gdal.GA_ReadOnly)
            georef = np.array(data.GetGeoTransform())
            # convert pixel coordinates to world coordinates
            pts_world = convert_pix2world(pts_pix[:,[1,0]], georef)

            # interpolate between points clicked by the user (1m resolution)
            pts_world_interp = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
            for k in range(len(pts_world)-1):
                pt_dist = np.linalg.norm(pts_world[k,:]-pts_world[k+1,:])
                xvals = np.arange(0,pt_dist)
                yvals = np.zeros(len(xvals))
                pt_coords = np.zeros((len(xvals),2))
                pt_coords[:,0] = xvals
                pt_coords[:,1] = yvals
                phi = 0
                deltax = pts_world[k+1,0] - pts_world[k,0]
                deltay = pts_world[k+1,1] - pts_world[k,1]
                phi = np.pi/2 - np.math.atan2(deltax, deltay)
                tf = transform.EuclideanTransform(rotation=phi, translation=pts_world[k,:])
                pts_world_interp = np.append(pts_world_interp,tf(pt_coords), axis=0)
            pts_world_interp = np.delete(pts_world_interp,0,axis=0)

            # save as geometry (to create .geojson file later)
            geoms.append(geometry.LineString(pts_world_interp))

            # convert to pixel coordinates and plot
            pts_pix_interp = convert_world2pix(pts_world_interp, georef)
            pts_sl = np.append(pts_sl, pts_world_interp, axis=0)
            ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--')
            ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko')
            ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko')

            # update title and buttons
            add_button.set_visible(True)
            end_button.set_visible(True)
            ax.set_title('click on <add> to digitize another shoreline or on <end> to finish and save the shoreline(s)',
                      fontsize=14)
            plt.draw()

            # let the user click again (<add> another shoreline or <end>)
            pt_input = ginput(n=1, timeout=-1, show_clicks=False)
            pt_input = np.array(pt_input)

            # if user clicks on <end>, save the points and break the loop
            if pt_input[0][0] > im_ms.shape[1]/2:
                add_button.set_visible(False)
                end_button.set_visible(False)
                plt.title('Reference shoreline saved as ' + sitename + '_reference_shoreline.pkl and ' + sitename + '_reference_shoreline.geojson')
                plt.draw()
                ginput(n=1, timeout=3, show_clicks=False)
                plt.close()
                break

        pts_sl = np.delete(pts_sl,0,axis=0)
        # convert world image coordinates to user-defined coordinate system
        with rasterio.open(fn) as src:
            image_epsg = str(src.crs)
        # Fix epsg format
        image_epsg = int(image_epsg.replace('EPSG:',''))
        dest_epsg = int(settings['output_epsg'].replace('EPSG:',''))
        
        pts_coords = convert_epsg(pts_sl, image_epsg, dest_epsg)

        # save the reference shoreline as .pkl
        with open(os.path.join(settings['run_input_folder'], sitename + '_reference_shoreline.pkl'), 'wb') as f:
            pickle.dump(pts_coords, f)
        
        # Store in settings
        settings['reference_shoreline'] = pts_coords
        
        # # also store as .geojson in case user wants to drag-and-drop on GIS for verification
        # for k,line in enumerate(geoms):
        #     gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(line))
        #     gdf.index = [k]
        #     gdf.loc[k,'name'] = 'reference shoreline ' + str(k+1)
        #     # store into geodataframe
        #     if k == 0:
        #         gdf_all = gdf
        #     else:
        #         gdf_all = gdf_all.append(gdf)
        # gdf_all.crs = {'init':'epsg:'+str(image_epsg)}
        # # convert from image_epsg to user-defined coordinate system
        # gdf_all = gdf_all.to_crs({'init': 'epsg:'+str(dest_epsg)})
        # # save as geojson
        # gdf_all.to_file(os.path.join(settings['output_folder'], sitename + '_reference_shoreline.geojson'),
        #                 driver='GeoJSON', encoding='utf-8')

        print('\nReference shoreline has been saved in ' + settings['run_input_folder'] + '\n')
            
    # check if a shoreline was digitised
    if len(pts_coords) == 0:
        raise Exception('Shoreline digitisation failes') 



#%%

def get_transects(settings):

    if os.path.isfile(settings['geojson_file']):
        settings['transects_load'] = transects_from_geojson(settings['geojson_file'])
        return
    else:
        # otherwise get the user to manually digitise a shoreline
        # Digitise shoreline on im_ref
        fn = settings['ref_merge_im']
              
        # get georef val
        data = gdal.Open(fn, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
              
        # create figure
        fig, ax = plt.subplots(1,1, figsize=[18,9], tight_layout=True)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

        # extract data and masks
        im_ms, nan_mask = get_ps_data(fn)
        
        # rescale image intensity for display purposes
        im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], nan_mask, 99.9)

        # plot the image RGB on a figure
        ax.axis('off')
        ax.imshow(im_RGB)
        
        ax.set_title('Click two points to define each transect (first point is the ' +
                      'origin of the transect and is landwards, second point seawards).\n'+
                      'When all transects have been defined, click on <ENTER>', fontsize=10)

        # initialise transects dict
        transects = dict([])
        counter = 0
        # loop until user breaks it by click <enter>
        while 1:
            # let user click two points
            pts = ginput(n=2, timeout=1000000)
            if len(pts) > 0:
                origin = pts[0]
            # if user presses <enter>, no points are selected
            else:
                fig.gca().set_title('Transect locations', fontsize=16)
                plt.title('Transect coordinates saved as ' + settings['site_name'] + '_transects.geojson')
                plt.draw()
                # wait 2 seconds for user to visualise the transects that are saved
                ginput(n=1, timeout=3, show_clicks=True)
                plt.close(fig)
                # break the loop
                break
            
            # add selectect points to the transect dict
            counter = counter + 1
            transect = np.array([pts[0], pts[1]])
                        
            # plot the transects on the figure
            ax.plot(transect[:,0], transect[:,1], 'b-', lw=2.5)
            ax.plot(transect[0,0], transect[0,1], 'rx', markersize=10)
            ax.text(transect[-1,0], transect[-1,1], str(counter), size=16,
                     bbox=dict(boxstyle="square", ec='k',fc='w'))
            plt.draw()
            
            # Convert pix coord to real world coords
            transect_world = convert_pix2world(transect[:,[1,0]], georef)
            transects[str(counter)] = transect_world
        
        # save transects.geojson
        gdf = transects_to_gdf(transects)
        # set projection
        dest_epsg = int(settings['output_epsg'].replace('EPSG:',''))
        gdf.crs = {'init':'epsg:'+str(dest_epsg)}
        # save as geojson    
        gdf.to_file(settings['geojson_file'], driver='GeoJSON', encoding='utf-8')
        # print the location of the files
        print('Transect locations saved in ' + settings['geojson_file'] + '\n')

    settings['transects_load'] = transects_from_geojson(settings['geojson_file'])



#%%


def filter_shorelines(settings, manual_filter = False, load_csv = False):
    
    # Load shoreline data from .pkl file
    if os.path.isfile(settings['sl_pkl_file']):
        with open(settings['sl_pkl_file'], 'rb') as f:
            shoreline_data = pickle.load(f) 
    else:
        raise Exception('No shoreline data pkl file found. Re-run previous steps and try again')

    # Load/save location for filter csv
    csv_path = os.path.join(settings['sl_thresh_ind'], 'shoreline_filter.csv')


    # If want to import manually updated filter csv file
    if load_csv == True:
        # Update pkl file with csv values
        filter_csv = pd.read_csv(csv_path)
        if filter_csv['name'].tolist() != shoreline_data['name']:
            # remove dates from filter_csv not in shoreline data
            mask = []
            for im in filter_csv['name'].tolist():
                if im in shoreline_data['name']:
                    bool_add = 1
                else:
                    bool_add = 0
                mask += [bool_add]  
            filter_csv['remove'] = mask
            filter_csv = filter_csv[filter_csv['remove'] == 1]
            filter_csv.reset_index(inplace = True, drop = True)
            idx = len(filter_csv)

            # add dates not in           
            for im in shoreline_data['name']:
                if im not in filter_csv['name'].tolist():
                    filter_csv.loc[idx, 'name'] = im
                    filter_csv.loc[idx, 'filter'] = 1
                    idx += 1
            print('csv columns do not match sl data dict, filter updated')
        
        if filter_csv['name'].tolist().sort() == shoreline_data['name'].sort():
            # Change file names
            shoreline_data['filter'] = filter_csv['filter'].tolist()
            # Loop through and change file names
            for i, im_name in enumerate(shoreline_data['name']):
                # Find plot default name
                plot_keep = os.path.join(settings['index_png_out'], im_name + ' shoreline plot.png')
                plot_discard = os.path.join(settings['index_png_out'], 'discard ' + im_name + ' shoreline plot.png')
                plot_update = os.path.join(settings['index_png_out'], 'update ' + im_name + ' shoreline plot.png')
    
                # Find filter boolean
                filt_bool = shoreline_data['filter'][i]
                
                # Change names
                if os.path.isfile(plot_discard):
                    os.rename(plot_discard, plot_keep)
                elif os.path.isfile(plot_update):
                    os.rename(plot_update, plot_keep)
                
                if filt_bool == 0:
                    os.rename(plot_keep, plot_discard)
                elif filt_bool == 2:
                    os.rename(plot_keep, plot_update)
        else:
            raise Exception('csv columns do not match sl data dict')

    # If want to allow all to images to pass
    elif manual_filter == False:
        # Update accept dictionary
        shoreline_data['filter'] = [1]*len(shoreline_data['name'])
        
        # Loop through and change file names
        for i, im_name in enumerate(shoreline_data['name']):
            # Find plot default name
            plot_keep = os.path.join(settings['index_png_out'], im_name + ' shoreline plot.png')
            plot_discard = os.path.join(settings['index_png_out'], 'discard ' + im_name + ' shoreline plot.png')

            # Find filter boolean
            filt_bool = shoreline_data['filter'][i]
            
            # Change names
            if os.path.isfile(plot_discard):
                os.rename(plot_discard, plot_keep)


    # Otherwise loop through each individual image and manually select
    else:
        # Add bool to dict
        shoreline_data['filter'] = []
    
        # create figure
        fig, ax = plt.subplots(1,1, figsize=[18,9], tight_layout=True)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        
        # No images
        no_im = str(len(shoreline_data['name']))

        # Loop through and plot each shorieline to filter
        for i, im_name in enumerate(shoreline_data['name']):
            # Find saved image plot and load
            plot_keep = os.path.join(settings['index_png_out'], im_name + ' shoreline plot.png')
            plot_discard = os.path.join(settings['index_png_out'], 'discard ' + im_name + ' shoreline plot.png')
                                        
            # if previously filtered - modify filename back to normal
            if os.path.isfile(plot_discard):
                os.rename(plot_discard, plot_keep)
                
            # Load image
            img = mpimg.imread(plot_keep)
               
            # Show shoreline image
            ax.imshow(img)
            ax.axis('off')
    
            # Add title
            ax.set_title('Press <right arrow> to accept or <left arrow> to discard (' +
                      str(i+1) + '/' + no_im + ')',
                      fontsize=12)
            
            # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
            # this variable needs to be immuatable so we can access it after the keypress event
            skip_image = False
            key_event = {}
            def press(event):
                # store what key was pressed in the dictionary
                key_event['pressed'] = event.key
            # let the user press a key, right arrow to keep the image, left arrow to skip it
            # to break the loop the user can press 'escape'
            while True:
                btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                    transform=ax.transAxes,
                                    bbox=dict(boxstyle="square", ec='k',fc='w'))
                btn_skip = plt.text(-0.1, 0.9, '⇦ discard', size=12, ha="left", va="top",
                                    transform=ax.transAxes,
                                    bbox=dict(boxstyle="square", ec='k',fc='w'))
                btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                    transform=ax.transAxes,
                                    bbox=dict(boxstyle="square", ec='k',fc='w'))
                plt.draw()
                fig.canvas.mpl_connect('key_press_event', press)
                plt.waitforbuttonpress()
                # after button is pressed, remove the buttons
                btn_skip.remove()
                btn_keep.remove()
                btn_esc.remove()
                # keep/skip image according to the pressed key, 'escape' to break the loop
                if key_event.get('pressed') == 'right':
                    skip_image = False
                    break
                elif key_event.get('pressed') == 'left':
                    skip_image = True
                    break
                elif key_event.get('pressed') == 'escape':
                    plt.close()
                    raise StopIteration('User cancelled manual shoreline filtering')
                else:
                    plt.waitforbuttonpress()
                        
            if skip_image:
                shoreline_data['filter'] += [0]
                
                # Change image filename
                os.rename(plot_keep, plot_discard)
                
                ax.clear()
                continue
            else:
                shoreline_data['filter'] += [1]
                ax.clear()
                continue
    
        # Close figure window
        plt.close()

    if load_csv == False:
        # Save new filter csv 
        to_pd = pd.DataFrame()
        to_pd['name'] = shoreline_data['name']
        to_pd['filter'] = shoreline_data['filter']
        to_pd.to_csv(csv_path)

    print('Manual shoreline filtering complete\n' + 
          str(shoreline_data['filter'].count(1)) + ' images kept\n' + 
          str(shoreline_data['filter'].count(0)) + ' images discarded\n')

    if shoreline_data['filter'].count(2) > 0:
        print(str(shoreline_data['filter'].count(2)) + ' images require manual updating\n')

    # Save and overwrite previous .pkl file
    with open(settings['sl_pkl_file'], 'wb') as f:
        pickle.dump(shoreline_data, f)
        
    return shoreline_data
    

#%% Modified CoastSat Functions - SDS Preproceccing

def rescale_image_intensity(im, cloud_mask, prob_high):
    """
    Rescales the intensity of an image (multispectral or single band) by applying
    a cloud mask and clipping the prob_high upper percentile. This functions allows
    to stretch the contrast of an image, only for visualisation purposes.

    KV WRL 2018

    Arguments:
    -----------
    im: np.array
        Image to rescale, can be 3D (multispectral) or 2D (single band)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    prob_high: float
        probability of exceedence used to calculate the upper percentile

    Returns:
    -----------
    im_adj: np.array
        rescaled image
    """

    # lower percentile is set to 0
    prc_low = 0

    # reshape the 2D cloud mask into a 1D vector
    vec_mask = cloud_mask.reshape(im.shape[0] * im.shape[1])

    # if image contains several bands, stretch the contrast for each band
    if len(im.shape) > 2:
        # reshape into a vector
        vec =  im.reshape(im.shape[0] * im.shape[1], im.shape[2])
        # initiliase with NaN values
        vec_adj = np.ones((len(vec_mask), im.shape[2])) * np.nan
        # loop through the bands
        for i in range(im.shape[2]):
            # find the higher percentile (based on prob)
            prc_high = np.percentile(vec[~vec_mask, i], prob_high)
            
            # clip the image around the 2 percentiles and rescale the contrast
            vec_rescaled = exposure.rescale_intensity(vec[~vec_mask, i],
                                                      in_range=(prc_low, prc_high),
                                                      out_range = (0,1))  # YD
            vec_adj[~vec_mask,i] = vec_rescaled
        # reshape into image
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1], im.shape[2])

    # if image only has 1 bands (grayscale image)
    else:
        vec =  im.reshape(im.shape[0] * im.shape[1])
        vec_adj = np.ones(len(vec_mask)) * np.nan
        prc_high = np.percentile(vec[~vec_mask], prob_high)
        vec_rescaled = exposure.rescale_intensity(vec[~vec_mask], in_range=(prc_low, prc_high))
        vec_adj[~vec_mask] = vec_rescaled
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1])

    return im_adj


#%% Modified CoastSat Functions - SDS.tools

def convert_world2pix(points, georef):
    """
    Converts world projected coordinates (X,Y) to image coordinates 
    (pixel row and column) performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (X,Y)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates (pixel row and column)
    
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)
    
    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(tform.inverse(points))
            
    # if single array    
    elif type(points) is np.ndarray:
        points_converted = tform.inverse(points)
        
    else:
        print('invalid input type')
        raise
        
    return points_converted


def convert_pix2world(points, georef):
    """
    Converts pixel coordinates (pixel row and column) to world projected 
    coordinates performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (row first and column second)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates, first columns with X and second column with Y
        
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)

    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            tmp = arr[:,[1,0]]
            points_converted.append(tform(tmp))
          
    # if single array
    elif type(points) is np.ndarray:
        tmp = points[:,[1,0]]
        points_converted = tform(tmp)
        
    else:
        raise Exception('invalid input type')
        
    return points_converted


def convert_epsg(points, epsg_in, epsg_out):
    """
    Converts from one spatial reference to another using the epsg codes
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.ndarray
        array with 2 columns (rows first and columns second)
    epsg_in: int
        epsg code of the spatial reference in which the input is
    epsg_out: int
        epsg code of the spatial reference in which the output will be            
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates from epsg_in to epsg_out
        
    """
    
    # define input and output spatial references
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(epsg_in)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(epsg_out)
    # create a coordinates transform
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(np.array(coordTransform.TransformPoints(arr)))
    # if single array
    elif type(points) is np.ndarray:
        points_converted = np.array(coordTransform.TransformPoints(points))  
    else:
        raise Exception('invalid input type')

    return points_converted



#%% Modified CoastSat Functions - SDS transects

def transects_to_gdf(transects):
    """
    Saves the shore-normal transects as a gpd.GeoDataFrame    
    
    KV WRL 2018

    Arguments:
    -----------
    transects: dict
        contains the coordinates of the transects          
                
    Returns:    
    -----------
    gdf_all: gpd.GeoDataFrame

        
    """  
       
    # loop through the mapped shorelines
    for i,key in enumerate(list(transects.keys())):
        # save the geometry + attributes
        geom = geometry.LineString(transects[key])
        gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
        gdf.index = [i]
        gdf.loc[i,'name'] = key
        # store into geodataframe
        if i == 0:
            gdf_all = gdf
        else:
            gdf_all = gdf_all.append(gdf)
            
    return gdf_all



def transects_from_geojson(filename):
    """
    Reads transect coordinates from a .geojson file.
    
    Arguments:
    -----------
    filename: str
        contains the path and filename of the geojson file to be loaded
        
    Returns:    
    -----------
    transects: dict
        contains the X and Y coordinates of each transect
        
    Source:
        https://github.com/kvos/CoastSat
        
    """  
    
    gdf = gpd.read_file(filename)
    transects = dict([])
    for i in gdf.index:
        transects[gdf.loc[i,'name']] = np.array(gdf.loc[i,'geometry'].coords)
        
    print('%d transects have been loaded' % len(transects.keys()))

    return transects