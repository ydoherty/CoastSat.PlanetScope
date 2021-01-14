# Train/update a new classifier for CoastSat.PlanetScope. This can improve the accuracy 
    # of the shoreline detection if the users are experiencing issues with the 
    # default classifier.

# Run this script with working directory as:
    # "... > CoastSat.PlanetScope > coastsat_ps > classifier"

#%% Initial settings

# load modules
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pickle
import pathlib

# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

# coastsat modules
from classifier_functions import (get_ps_metadata, label_images, load_labels, 
                            format_training_data, plot_confusion_matrix, 
                            evaluate_classifier)


#%%
def create_folder(filepath):
    ''' Creates a filepath if it doesn't already exist
    Will not overwrite files that exist 
    Assign filepath string to a variable
    '''
    pathlib.Path(filepath).mkdir(exist_ok=True) 
    return filepath

# filepaths
filepath_train = create_folder(os.path.join(os.getcwd(), 'training_data'))
filepath_models = create_folder(os.path.join(os.getcwd(), 'models'))


#%% Instructions

# 0) Classifier can only be updated for one site at a time, repeat below steps for 
    # each site that requires training

# 1) Classifier requires merged TOA images. Run CoastSat.PlanetScope for a site up to/including 
    # step 1.3 (Pre-Processing - image coregistration and scene merging)

# 2) To manually train classifier for ALL output scenes from step 1, use
    # '...CoastSat.PlanetScope/outputs/SITE/toa_image_data/merged_data/local_coreg_merged'
    # or equivelent folder (local/global/off) as the variable 'filepath_images'
    
# 2.1) To run on a SUBSET of images, copy TOA/nan/cloud files from the above folder
    # to a new folder and set this folder as filepath_images variable (below)
    # Note: You only need a few images (~10) to train the classifier.
     
sitename = 'NARRA'

filepath_images = ('/Users/Yarran/Documents/GitHub/CoastSat.PlanetScope/outputs/NARRA/toa_image_data/merged_data/local_coreg_merged')

epsg = 28356

classifier_save_name = 'NN_4classes_PS'


#%% Update settings

settings ={'filepath_train':filepath_train, # folder where the labelled images will be stored
           'cloud_thresh':0.9, # percentage of cloudy pixels accepted on the image
           'inputs':{'filepath':filepath_images, 'sitename': sitename}, # folder where the images are stored
           'labels':{'sand':1,'white-water':2,'water':3,'other land features':4}, # labels for the classifier
           'colors':{'sand':[1, 0.65, 0],'white-water':[1,0,1],'water':[0.1,0.1,0.7],'other land features':[0.8,0.8,0.1]},
           'tolerance':0.02, # this is the pixel intensity tolerance, when using flood fill for sandy pixels
                             # set to 0 to select one pixel at a time
            }
        

#%% Label images [skip if only merging previously manually classified data sets]
# Label the images into 4 classes: sand, white-water, water and other land features.
# The labelled images are saved in the *filepath_train* and can be visualised 
    # afterwards for quality control. If you make a mistake, don't worry, this can 
    # be fixed later by deleting the labelled image.

# label the images with an interactive annotator

# create compatible metadata dict
metadata = get_ps_metadata(filepath_images, epsg)

# label images
label_images(metadata, settings)
    

#%% Train Classifier [uses all sites]

# A Multilayer Perceptron is trained with *scikit-learn*. To train the classifier, the training data needs to be loaded.
# You can use the data that was labelled here and/or the original CoastSat training data.

# load labelled images
train_sites = next(os.walk(os.path.join(os.getcwd(), 'training_data')))[1]
#train_sites = ['NARRA']
print('Loading data for sites:\n',train_sites, '\n')
features = load_labels(train_sites, settings)


#%% [OPTIONAL] - import previously trained features

save_pkl = 'CoastSat_PS_training_set_NARRA_50000.pkl'

# Load the original CoastSat.PlanetScope training data (and optionally merge it with your labelled data)
with open(os.path.join(settings['filepath_train'], save_pkl), 'rb') as f:
    features_original = pickle.load(f)
print('Loaded classifier features:')
for key in features_original.keys():
    print('%s : %d pixels'%(key,len(features_original[key])))

# # Option 1) add the white-water data from the original training data
# features['white-water'] = np.append(features['white-water'], features_original['white-water'], axis=0)

# # Option 2) Merge all the classes
for key in features.keys():
    features[key] = np.append(features[key], features_original[key], axis=0)

# Option 3) Use original data
# features = features_original 

print('\nUpdated classifier features:')
for key in features.keys():
    print('%s : %d pixels'%(key,len(features[key])))


#%% As the classes do not have the same number of pixels, it is good practice to 
    #subsample the very large classes (in this case 'water' and 'other land features'):

# Subsample randomly the land and water classes
    # as the most important class is 'sand', the number of samples in each class 
    #should be close to the number of sand pixels
    
n_samples = 25000
#for key in ['water', 'other land features']:
for key in ['sand', 'water', 'other land features']:
    features[key] =  features[key][np.random.choice(features[key].shape[0], n_samples, replace=False),:]
# print classes again
print('Re-sampled classifier features:')
for key in features.keys():
    print('%s : %d pixels'%(key,len(features[key])))
    
    
#%% [OPTIONAL] - save features

# Save name
save_pkl = 'CoastSat_PS_training_set_new.pkl'
save_loc = os.path.join(settings['filepath_train'], save_pkl)
    
# Save training data features as .pkl
with open(save_loc, 'wb') as f:
    pickle.dump(features, f)
print('New classifier training feature set saved to:\n', save_loc)
    

#%% When the labelled data is ready, format it into X, a matrix of features, and y, a vector of labels:

# format into X (features) and y (labels) 
classes = ['sand','white-water','water','other land features']
labels = [1,2,3,0]
X,y = format_training_data(features, classes, labels)


#%% Divide the dataset into train and test: train on 70% of the data and evaluate on the other 30%:

# divide in train and test and evaluate the classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)
classifier = MLPClassifier(hidden_layer_sizes=(100,50), solver='adam')
classifier.fit(X_train,y_train)
print('Accuracy: %0.4f' % classifier.score(X_test,y_test))


#%% [OPTIONAL] A more robust evaluation is 10-fold cross-validation (may take a few minutes to run):

# cross-validation
scores = cross_val_score(classifier, X, y, cv=10)
print('Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std() * 2))


#%% Plot a confusion matrix:

# plot confusion matrix
y_pred = classifier.predict(X_test)
plot_confusion_matrix(y_test, y_pred,
                                   classes=['other land features','sand','white-water','water'],
                                   normalize=False);
plt.show()


#%% [OPTIONAL] - Update classifier save name with number of samples and evaluation results

# Update save name with settings
classifier_save_name += '_' + str(n_samples)
for site in train_sites:
    classifier_save_name += '_' + site
classifier_save_name += '_' + str(int(classifier.score(X_test,y_test)*10000))
print('New save name is:\n',classifier_save_name)


#%% When satisfied with the accuracy and confusion matrix, train the model using 
    # ALL (no 70/30 split) training data and save

# train with all the data and save the final classifier
classifier = MLPClassifier(hidden_layer_sizes=(100,50), solver='adam')
classifier.fit(X,y)
save_loc = os.path.join(filepath_models, classifier_save_name + '.pkl')
joblib.dump(classifier, save_loc)
print('New classifier saved to:\n', save_loc)


#%% 4. Evaluate the classifier
# Load a classifier that you have trained (specify the classifiers filename) and evaluate it on the satellite images.
# This section will save the output of the classification for each site in a directory named \evaluation.
# Only evaluates images in the filepath_images folder

classifier_eval = classifier_save_name

# load and evaluate a classifier
classifier = joblib.load(os.path.join(filepath_models, classifier_eval  + '.pkl'))
settings['min_beach_area'] = 1000
settings['cloud_thresh'] = 0.9

# visualise the classified images
for site in train_sites:
    settings['inputs']['sitename'] = site
    # load metadata
    metadata = get_ps_metadata(filepath_images, epsg)

    # plot the classified images
    evaluate_classifier(classifier, classifier_eval, metadata, settings)

