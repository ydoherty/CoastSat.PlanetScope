# Plotting

import numpy as np
import rasterio

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import skimage.morphology as morphology


#%%

class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    Credit: Joe Kington, http://chris35wills.github.io/matplotlib_diverging_colorbar/
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


#%% 

def initialise_plot(settings, im_name, index):
    
    plt.ioff()
    
    # create figure
    fig = plt.figure()
    fig.tight_layout()
    fig.set_size_inches([8, 8])
    #mng = plt.get_current_fig_manager()
    #mng.window.showMaximized()

    # according to the image shape, decide whether it is better to have the images
    # in vertical subplots or horizontal subplots
    if index.shape[0] > 0.75*index.shape[1]:
        # vertical subplots
        gs = gridspec.GridSpec(nrows = 10, ncols = 30,
                        wspace = 0, hspace = 0.15,
                        bottom=0.07, top=0.89, 
                        left=0.1, right=0.9)
        
        ax1 = fig.add_subplot(gs[0:7,0:10])
        ax2 = fig.add_subplot(gs[0:7,10:20], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(gs[0:7,20:30], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(gs[8:,1:29])
        
    else:
        # horizontal subplots
        print(('\n    Code to format non-vertical images for plots is not properly developed - this will not impact shoreline extraction.') + 
              ('Manually edit initialise_plot function in coastsat_ps > plotting.py file as required.\n'))
        gs = gridspec.GridSpec(nrows = 8, ncols = 30,
                        wspace = 0, hspace = 0.15,
                        bottom=0.07, top=0.89, 
                        left=0.1, right=0.9)

        ax1 = fig.add_subplot(gs[0:2,:])
        ax2 = fig.add_subplot(gs[2:4,:], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(gs[4:6,:], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(gs[6:,1:29]) # histogram

    # Set title from im_name
    fig.suptitle(settings['water_index'] + ' Water Index with ' +  
                 settings['thresholding'] + ' Thresholding\n' + 
                 im_name, 
                 fontsize = 12)

    return fig, ax1, ax2, ax3, ax4


def initialise_plot_gen(settings, im_name, index):
    
    plt.ioff()
    
    # create figure
    fig = plt.figure()
    fig.tight_layout()
    fig.set_size_inches([8, 8])
    #mng = plt.get_current_fig_manager()
    #mng.window.showMaximized()

    # according to the image shape, decide whether it is better to have the images
    # in vertical subplots or horizontal subplots
    if index.shape[0] > 0.75*index.shape[1]:
        # vertical subplots
        gs = gridspec.GridSpec(nrows = 10, ncols = 30,
                        wspace = 0, hspace = 0.15,
                        bottom=0.07, top=0.89, 
                        left=0.1, right=0.9)
        
        ax1 = fig.add_subplot(gs[0:7,0:15])
        ax2 = fig.add_subplot(gs[0:7,15:30], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(gs[8:,1:29])
        
    else:
        # horizontal subplots
        print(('\n    Code to format non-vertical images for plots is not properly developed - this will not impact shoreline extraction.') + 
              ('Manually edit initialise_plot_gen function in coastsat_ps > plotting.py file as required.\n'))
        gs = gridspec.GridSpec(nrows = 8, ncols = 30,
                        wspace = 0, hspace = 0.15,
                        bottom=0.07, top=0.89, 
                        left=0.1, right=0.9)
        
        ax1 = fig.add_subplot(gs[0:3,:])
        ax2 = fig.add_subplot(gs[3:6,:], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(gs[6:,1:29]) # histogram

    # Set title from im_name
    fig.suptitle(settings['water_index'] + ' Water Index with ' +  
                 settings['thresholding'] + ' Thresholding\n' + 
                 im_name, 
                 fontsize = 12)

    return fig, ax1, ax2, ax3

#%%


def rgb_plot(ax, im_RGB, sl_pix, transects):
    
    # Set nan colour
    im_RGB = np.where(np.isnan(im_RGB), 0.3, im_RGB)
    
    # Plot background RGB im
    ax.imshow(im_RGB)
    
    # Overlay shoreline
    ax.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize = 0.3)     

    # Plot transects
    for pf in transects.keys():
        points = transects[pf]
        ax.plot(points[:,0], points[:,1], color = 'k', linestyle = ':')
        
        # Decide text layout
        if points[0,0] > points[1,0]:
            ha = 'right'
            text = pf + ' '
        else:
            ha = 'left'
            text = ' ' + pf

        ax.text(points[1,0], points[1,1], text, fontsize = 8, color = 'white', 
                ha = ha, va = 'center')

    # Figure settings
    ax.axis('off')
    ax.set_title('RGB', fontsize=10)        


#%%

def class_plot(ax, im_RGB, im_classif, sl_pix, transects, settings, colours):
    
    # compute classified image
    im_class = np.copy(im_RGB)
    
    # Create coastsat class format
    im_sand = im_classif == 1
    im_swash = im_classif == 2
    im_water = im_classif == 3
    
    # remove small patches of sand or water that could be around the image (usually noise)
    im_sand = morphology.remove_small_objects(im_sand, min_size=settings['min_beach_area_pixels'], connectivity=2)
    im_water = morphology.remove_small_objects(im_water, min_size=settings['min_beach_area_pixels'], connectivity=2)
    im_labels = np.stack((im_sand,im_swash,im_water), axis=-1)
    
    # Apply colours
    for k in range(0,im_labels.shape[2]):
        im_class[im_labels[:,:,k],0] = colours[k,0]
        im_class[im_labels[:,:,k],1] = colours[k,1]
        im_class[im_labels[:,:,k],2] = colours[k,2]
    
    # Set nan colour
    im_class = np.where(np.isnan(im_class), 0.3, im_class)

    # Plot classes over RGB
    ax.imshow(im_class)
    
    # Plot shoreline
    ax.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize = 0.3)     
    
    # Plot transects
    for pf in transects.values():
        ax.plot(pf[:,0], pf[:,1], color = 'k', linestyle = ':')  
        
    # Plot colours
    orange_patch = mpatches.Patch(color=colours[0,:], label='sand')
    white_patch = mpatches.Patch(color=colours[1,:], label='whitewater')
    blue_patch = mpatches.Patch(color=colours[2,:], label='water')
    black_patch = mpatches.Patch(color='0.3', label='nan/cloud')
    black_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
    red_line = mlines.Line2D([],[],color='k',linestyle=':', label='transects')
    
    # Add legend
    ax.legend(handles=[orange_patch, white_patch, blue_patch, black_patch, 
                       black_line, red_line],
                bbox_to_anchor=(0.5, 0), loc='upper center', fontsize=9,
                ncol = 6)
    
    # General settings
    ax.axis('off')    
    ax.set_title('Classified Image', fontsize=10)
    

    
#%%


def index_plot(ax, index_in, t_otsu, comb_mask, sl_pix, transects, fig, settings):
    
    # Mask index
    index = np.copy(index_in)
    index[comb_mask] = np.nan

    # Find index limits
    min = np.nanmin(index)
    max = np.nanmax(index)
    
    # Plot colourised index
    cmap = plt.cm.coolwarm  # red to blue
    cmap.set_bad(color='0.3')
    cax = ax.imshow(index, 
                     cmap=cmap, 
                     clim=(min, max), 
                     norm=MidpointNormalize(midpoint = t_otsu,
                                            vmin=min, vmax=max))
    
    # Overlay shoreline
    ax.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize = 0.3)     

    # Plot transects
    for pf in transects.values():
        ax.plot(pf[:,0], pf[:,1], color = 'k', linestyle = ':')
                
    # Add colourbar
    cbar = fig.colorbar(cax, ax = ax, orientation='vertical', shrink=0.65)
    cbar.set_label(settings['water_index'] + ' Pixel Value', rotation=270, labelpad=10)

    # Figure settings
    ax.axis('off')
    ax.set_title(settings['water_index'], fontsize=10)        
    

#%%

def histogram_plot(ax, vec, t_otsu, settings):

    # Set labels
    ax.set_title(settings['water_index'] + ' Pixel Value Histogram Thresholding',
                  fontsize = 10)
    ax.set_xlabel(settings['water_index'] + ' Pixel Value', fontsize = 10)
    #ax.set_ylabel("Pixel Count", fontsize= 10)
    ax.set_ylabel("Pixel Class PDF", fontsize= 10)
    ax.axes.yaxis.set_ticks([])
    
    # Plot threshold value(s)
    ax.axvline(x = t_otsu, color = 'k', linestyle = '--', label = 'Threshold Value')
    
    # Add legend
    ax.legend(bbox_to_anchor = (1,1), loc='lower right', framealpha = 1,
              fontsize = 8) #, fontsize = 'xx-small')
    
    # Plot histogram
    ax.hist(vec, settings['otsu_hist_bins'], color='blue', alpha=0.8, density=True)
     


def histogram_plot_split(ax, index, im_classif, im_ref_buffer, t_otsu, settings, colours):

    # Set labels
    ax.set_title(settings['water_index'] + ' Pixel Value Histogram Thresholding',
                  fontsize = 10)
    ax.set_xlabel(settings['water_index'] + ' Pixel Value', fontsize = 10)
    ax.set_ylabel("Pixel Class PDF", fontsize= 10)
    ax.axes.yaxis.set_ticks([])
    # Plot threshold value(s)
    l1 = ax.axvline(x = t_otsu, color = 'k', linestyle = '--', label = 'threshold')
    
    # Add legend
    grey_patch = mpatches.Patch(color='0.5', label='other')
    ax.legend(handles=[grey_patch, l1], bbox_to_anchor = (1,1), loc='upper right', framealpha = 1,
              fontsize = 9) #, fontsize = 'xx-small')
    
    # Organise colours
    col_list = ['0.5', # Other
                colours[0,:], # Sand
                colours[1,:], # WW
                colours[2,:]] # Water
    
    # Plot histograms
    for i in [0,1,2,3]:
        # Extract mask
        class_im = im_classif == i
        mask_class = ~class_im
        mask_all = (mask_class + im_ref_buffer) >0
        
        # Copy index
        idx_copy = np.copy(index)
        
        # Remove nan
        if i == 0:
            idx_copy[mask_class] = np.nan
        else:
            idx_copy[mask_all] = np.nan

        # Create vec
        vec = idx_copy.reshape(idx_copy.shape[0] * idx_copy.shape[1])
        vec = vec[~np.isnan(vec)]
    
        # Plot histogram
        ax.hist(vec, settings['otsu_hist_bins'], color=col_list[i], alpha=0.8,
                density=True)


#%%

def check_land_mask(settings):

    # Load the RGB image
    rgb_path = settings['georef_im_path']
    with rasterio.open(rgb_path) as src_rgb:
        rgb_image = src_rgb.read([3, 2, 1])  # Read the RGB bands
    
    # Check the maximum value of the image
    max_value = rgb_image.max()
    
    # Normalize the RGB image if necessary (for visualization)
    if max_value > 1.0:
        rgb_image = rgb_image.astype(float) / max_value
    
    # Load the single-band mask image
    mask_path = settings['land_mask']
    with rasterio.open(mask_path) as src_mask:
        mask_image = src_mask.read(1)  # Read the single band
    
    # Normalize the mask (if necessary) to have values between 0 and 1
    mask_image = mask_image.astype(float)
    mask_image = mask_image / mask_image.max()
    
    # Invert the mask
    mask_image = 1 - mask_image
    
    # Apply the mask to the RGB image
    masked_rgb_image = np.copy(rgb_image)
    for i in range(3):  # Apply mask on each channel (R, G, B)
        masked_rgb_image[i] = rgb_image[i] * mask_image
    
    # Plot the images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the original RGB image
    ax[0].imshow(np.moveaxis(rgb_image, 0, -1))  # Move the channels to the last dimension
    ax[0].set_title('Co-registration reference image')
    ax[0].axis('off')
    
    # Plot the masked RGB image
    ax[1].imshow(np.moveaxis(masked_rgb_image, 0, -1))  # Move the channels to the last dimension
    ax[1].set_title('Land mask region')
    ax[1].axis('off')
    
    plt.show(block=False)
    
    # save image
    save_loc = settings['georef_im_path'].replace('.tif', '_and_land_mask_figure.png')
    plt.savefig(save_loc, bbox_inches='tight', dpi=200)


