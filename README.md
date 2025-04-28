# CoastSat.PlanetScope

Yarran Doherty, UNSW Water Research Laboratory

First release 01/2021, useability updates 06/2024

## **Description**

CoastSat.PlanetScope is an open-source extension to the CoastSat python toolkit enabling users to extract time-series of shoreline position from PlanetScope Dove satellite imagery. Similar to CoastSat, the CoastSat.PlanetScope extension utilises a machine-learning shoreline detection algorithm to classify images into sand, water, whitewater and other pixel classes prior to a sub-pixel shoreline extraction process. An additional co-registration step is implemented to minimise the impact of geo-location errors. Transect intersection and a tidal correction  based on a generic beach slope is then applied to provide a timeseries of shoreline position. 

![](readme_files/extraction.png)

Output files include:
- Shoreline timeseries .geojson file for use in GIS software (no tidal correction)
- Tidally corrected shoreline transect intersection timeseries csv
- Image shoreline extraction plots
- Tidally corrected transect timeseries plots


## **Installation**

For users of Coastsat, it is possible to run the CoastSat.PlanetScope toolkit in the original CoastSat environment once the following packages are installed:
- [Rasterio](https://rasterio.readthedocs.io/en/latest/installation.html)
- [AROSICS](https://danschef.git-pages.gfz-potsdam.de/arosics/doc/installation.html)

It is recommended however to create a dedicated coastsat_ps environment using the provided environment.yml file. The advised method of installation is using [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). [Anaconda](https://www.anaconda.com/) may also be used, however this method is slower and more prone to package conflicts. Once Mamba or Anaconda are installed, open Mamba/Anaconda prompt and navigate to the local downloaded CoastSat.PlanetScope repo folder by entering "cd /d C:\add\filepath\here\to\CoastSat.PlanetScope". Once this has been done, enter the following commands one by one to install the planetscope environment from the provided .yml file (replace mamba with conda below if using Anaconda):

```
mamba env create -f environment.yml -n coastsat_ps

mamba activate coastsat_ps

spyder
```

If installation fails using the default environment.yml, several alternate yaml file options are provided in the alt_environment folder. If you are having issues installing and opening spyder, see further recommendations in the 'Known Issues & Workarounds' section below. 

Once spyder is open, navigate to the CoastSat.PlanetScope folder to set the working direcctory (top right hand box in spyder) and open the CoastSat_PS.py file to begin the example run through. Note that every time you want to run the code, you will need to activate the coastsat_ps environnment and open spyder using the last two lines of code above. 


## **Data Requirements**

PlanetScope images must be manually downloaded by the user. 
- Access to PlanetScope imagery can be obtained through a [free trial](https://www.planet.com/trial/), [research license](https://www.planet.com/markets/education-and-research/) or [paid subscription](https://www.planet.com/contact-sales/#contact-sales).
- Required PlanetScope file type is '4-band multispectral Analytic Ortho Scene'. Using the QGIS plugin, filter for "PlanetScope Scene" and download "Analytic Radiance (TOAR) 4-band GeoTiff" images. It is recommended to select the 'clip to AOI' options to reduce file size. Development was done with an AOI of ~5km2. 

- To run CoastSat.PlanetScope, keep all downloaded images and associated metadata in a single folder and outline this folder filepath in the CoastSat_PS.py settings.

All user input files (area of interest polygon, transects & tide data) should be saved in the folder "...CoastSat.PlanetScope/user_inputs"
- Analysis region of interest .kml file may be selected and downloaded using [this tool](http://geojson.io). 
- Transects .geojson file (optional) should match the user input settings epsg. If skipped, transects may be drawn manually with an interactive popup. Alternately, the provided NARRA_transect.geojson file may be manually modified in a text editor to add/remove/update transect names, coordinates and epsg. 
- Tide data .csv for tidal correction (optional) should be in UTC time and local mean sea level (MSL) elevation. See NARRA_tides.csv for csv data and column name formatting. 

Beach slopes for the tidal correction (step 5) can be extracted using the [CoastSat.Slope toolkit](https://github.com/kvos/CoastSat.slope)


## **Usage**

![](readme_files/timeseries.png)

It is recommended the toolkit be run in spyder. Ensure spyder graphics backend is set to 'automatic' for proper interactive plot rendering. 
- Preferences - iPython console - Graphics - Graphics Backend - Backend - Automatic

CoastSat.PlanetScope is run from the CoastSat_PS.py file. 
- Instructions and comments are provided in this file for each step. 
- It is recommended steps be run as individual cells for first time users. 

Settings and interactive steps are based on the CoastSat workflow and will be familiar to users of CoastSat. 

Interactive popup window steps include:
- Raw PlanetScope reference image selection for co-registration [step 1.2.]
- Top of Atmosphere merged reference image selection for shoreline extraction [step 2.1.]
- Reference shoreline digitisation (refer 'Reference shoreline' section of CoastSat readme for example) - [step 2.1.]
- Transect digitisation (optional - only if no transects.geojson file provided) - [step 2.1.]
- Manual error detection (optional - keep/discard popup window as per CoastSat) - [step 3.]

Results and plots are saved in '...CoastSat.PlanetScope/outputs/site_name/shoreline outputs'. 


## **Training Neural-Network Classifier**

Due to the preliminary stage of testing, validation was primarily completed at Narrabeen-Collaroy beach in Sydney, Australia. As such, the NN classifier is optimised for this site and may perform poorly at alternate sites with differing sediment composition. It is recommended a new classifier be trained for such regions. 

Steps are provided in "...CoastSat.PlanetScope/coastsat_ps/classifier/train_new_classifier.py". 
- Instructions are in this file and based of the CoastSat classifier training [methods](https://github.com/kvos/CoastSat/blob/master/doc/train_new_classifier.md). 
- CoastSat.PlanetScope must be run up to/including step 1.3. on a set of images to extract co-registered and top of atmosphere corrected scenes for classifier training. 


## **Validation Results**

- Accuracy validated against in-situ RTK-GPS survey data at Narrabeen-Collaroy beach in the Northen beaches of Sydney, Australia with a RMSE of 3.5m (n=438). 
- An equivelent validation study at Duck, North Carolina, USA provided an observed RMSE error of 4.7m (n=167). 

Detailed results and methodology outlined in:

Doherty Y., Harley M.D., Splinter K.D., Vos K. (2022). A Python Toolkit to Monitor Sandy Shoreline Change Using High-	Resolution PlanetScope Cubesats. Environmental Modelling & Software. https://doi.org/10.1016/j.envsoft.2022.105512

As a starting point for user validation studies, an example jupyter notebook comparing CoastSat (Landsat/Sentinel-2) shorelines against in-situ survey data can be found on the main [CoastSat](https://github.com/kvos/CoastSat) repo for Narrabeen-Collaroy beach. Note that CoastSat.PlanetScope results will require re-structuring to match the CoastSat validation input format. 


## **Development Opportunities**
- Currently the Planet provided udm2 useable pixel filter is not supported and a conversion into the old udm format is used. An updated udm2 processing step may improve cloud and sensor error detection. 
- The PSB.SD sensor type (see [here](https://developers.planet.com/docs/apis/data/sensors)) was released while this project was in its final stages of development. Utilisation of these additional 4 image bands may be an opportunity to further improve shoreline accuracy.
- Integration of existing CoastSat tools:
  - Automated extraction of FES2022 tide data
  - Automated integration with CoastSat.Slope
  - Add an example site vallidation codes
- Add post processing and mapping/visualisation codes similar to [this](https://ci-folium-web-map.s3.ap-southeast-2.amazonaws.com/UNSW-WRL-CHIS/aus0206_Narrabeen.html). 
- Additional vallidation and comparison studies:
  - Comparison between the three PlanetScope Dove sensor types (PS2, PS2.SD and PSB.SD)
  - Vallidation and testing at additional sites globally 
  - Testing along non-sandy coastlines


## **Known Issues & Workarounds**

The following issues have been identified by users and workarounds are presented below. My availability to maintain and update this repo is limited so user feedback, bug fixes and devlopments are encouraged! 

#### **Environment installation**
- Environment installation issues and package conflicts - see [here](https://github.com/ydoherty/CoastSat.PlanetScope/issues/2#issuecomment-830543064). Seems to be resolved for both mac and windows. Unexpected installation issues may still persist so alternate installation environments are provided in the event the standard yaml file does not work. If installation fails for all of the provided yaml files, try edit the default environment.yml file in a text editor to remove spyder and try again. If you do this you will either have to install spyder seperately, or run CoastSat.PlanetScope from an alternate IDE (ie VSCode). 

#### **Spyder crashing after opening**
- Spyder does not play well with complex environments. In my experience running "mamba uninstall spyder", then "mamba install spyder" will generally fix the issue. Failing that, the fastest method is usually to remove your environment with "mamba env remove -n coastsat_ps" and try install one of the provided alternate environments such as "environment_alt_AUG_24_explicit.yml". Trying to debug an environment can take hours so its usually much faster to start from scratch.
- It is also possible to manually install a standalone version of spyder and then link it to a python environment that doesn't have spyder installed. To use this method, download spyder from [here](https://docs.spyder-ide.org/5/installation.html#install-standalone). After this, edit the environment.yml file to replace 'spyder' with 'spyder-kernels' and then create the python environnment as per the installation instructions above. Using this method spyder needs to be opened manually rather than through the command line. Steps to link spyder to your environment are outlined [here](https://docs.spyder-ide.org/current/faq.html#using-packages-installer) and [here](https://docs.spyder-ide.org/current/faq.html#using-existing-environment). 
- If all else fails and you are unable to resolve your environment and/or open spyder, try install a new environment without spyder (steps outlined in the 'Environment installation' bullet point above). CoastSat.Planetscope can be run directly from the command line or using any alternate IDE (such as VSCode). 


#### **General**
- If the toolbox repeatedly crashes on an individual image, inspect the file in GIS software to ensure there are no issues with the image. For impacted images, the easiest workaround is to delete the image (and all assoicated files) from the downloads folder and re-run the toolbox. 
- To remove all files created by the toolbox in order to start fresh, simply delete the CoastSat.PlanetScope > outputs > RUN_NAME folder.
- If there are errors assoicated with file names, confirm input files (image tif, udm tif, image xml and image json) have the correct naming convention. The format should have the prefix "YYYYMMDD_HHMMSS_<SATELLITEID>" followed by "_3B_AnalyticMS_clip.tif", "_3B_AnalyticMS_DN_udm_clip.tif", "_3B_AnalyticMS_metadata_clip.xml" and "_metadata.json". For example 20230523_153456_101b_3B_AnalyticMS_clip.tif.
- Unresolved issues may exist when using large AOIs but this has not been tested. Split large regions into smaller subsets for processing if this is the case. 

#### **Co-registration**
- Arosics does not run cell 1.3 (image co-registration) in spyder on windows. This appears to be due to the way arosics handles multiprocessing on the windows operating system. A workaround is to copy and run the code directly in the spyder console or to run using a selection and F9. Instructions are provided in cell 1.3.
- Arosics raises errors when performing co-registration from cell 1.3. Sometimes this is due to the selected reference image from cell 1.2. Delete the files "SITE_NAME_im_ref.tif" and "SITE_NAME_land_mask.tif" in the folder CoastSat.PlanetScope > outputs > SITE_NAME > input_data and re-run cell 1.2 to select a new reference image.
- Most images flagged as failing co-registration. Coregistration requires aligning the images based on land pixels only. If the land mask is poorly defined, try adjusting the settings mentioned in cell 1.2 or train a new classifier trained on your site. You can also try setting 'generic_land_mask' = True in data_import.py. This uses a single mask to align all images rather than calculating a new mask for each image (which is necessary when sites change significantly through time). Trying global coregistration may help for some sites, or try adjusting advanced coregistration settings in data_import.py such as tie-point grid spacing and window_size. 
- For regions where downloaded images do not all have the same CRS/epsg, the arosics co-registration step will fail and provide an error message. If this occurs, edit the advanced settings in the CoastSat.PlanetScope > coastsat_ps > data_import.py file (line 37) to change the arosics_reproject setting to True.
- Inability to select a reference image. Comment out [this](https://github.com/ydoherty/CoastSat.PlanetScope/issues/2#issuecomment-828644872) section of code. See [here](https://github.com/ydoherty/CoastSat.PlanetScope/issues/2#issuecomment-840894375) for explanation. 

#### **Plots**
- Poor figure rendering in the interacive error checking from cell 3. The aspect ratio for plotted figures in the "Shoreline plots" outputs folder is hard coded. Figures will vary depending on the AOI shape and may look poor for certain configurations. The user can edit plotting settings manually in the coastsat_ps > plotting.py file in the initialise_plot function. Plotting is for error checking/visual purposes only and poor figure rendering does not impact shoreline extracion in any way. 
- Figure window white with a tiny dot for the shoreline image. This may be caused when the input transect geojson file is in the wrong coordinate reference system. The figure will show still show the image and transects even if they have a different CRS. The blank space between them in the figure will be white. Fixed by updating the transect file, or by choosing the option to manually draw transects. 

