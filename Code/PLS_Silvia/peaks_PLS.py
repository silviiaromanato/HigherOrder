import numpy as np
import pandas as pd
from scipy.io import loadmat
from compute import *
from helpers_PLS import *
import sys 
import os
os.chdir('..')
print(os.getcwd())
from features.helper_corr_mtx import *

PATH_YEO = '/media/miplab-nas2/Data2/Movies_Emo/Silvia/HigherOrder/Data/yeo_RS7_Schaefer100S.mat'

def extract_features(movie_name, columns = ['mean_chroma', 'mean_mfcc', 'spectralflux', 'rms', 'zcrs'], 
                             columns_images = ['average_brightness_left', 'average_saturation_left', 'average_hue_left'],
                             cluster = True
                             ):
    if cluster == True:
        PATH_EMO = f'/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/'
    else:
        PATH_EMO = '/Users/silviaromanato/Desktop/SEMESTER_PROJECT/Material/Data/EmoData/'
    length = extract_corrmat_allregressors(PATH_EMO, movie_name).shape[0]

    movie_name_with_ = re.sub(r"(\w)([A-Z])", r"\1_\2", movie_name)
    if cluster == True:
        df_sound = pd.read_csv(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/features_extracted/features_extracted/features_sound_{movie_name_with_}.csv')[columns]
        df_images = pd.read_csv(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/features_extracted/features_extracted/movie_features_{movie_name_with_}_exp.csv')[columns_images]
    else: 
        df_sound = pd.read_csv(f'/Users/silviaromanato/Desktop/SEMESTER_PROJECT/Material/Data/features_extracted/features_sound_{movie_name_with_}.csv')[columns]
        df_images = pd.read_csv(f'/Users/silviaromanato/Desktop/SEMESTER_PROJECT/Material/Data/features_extracted/movie_features_{movie_name_with_}_exp.csv')[columns_images]
    window_size1 = df_images.shape[0] // length
    images_mean = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size1), mode='valid') / window_size1, axis=0, arr=df_images)

    window_size2 = df_sound.shape[0] // length
    sound_mean = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size2), mode='valid') / window_size2, axis=0, arr=df_sound)

    # Select elements to represent the initial arrays
    selected_indices1 = np.linspace(0, len(images_mean) - 1, length, dtype=int)
    selected_indices2 = np.linspace(0, len(sound_mean) - 1, length, dtype=int)
    df_images = pd.DataFrame(images_mean[selected_indices1])
    df_sound = pd.DataFrame(sound_mean[selected_indices2])

    # set the column names
    df_images.columns = columns_images
    df_sound.columns = columns

    # concat the two dataframes 
    df_images.reset_index(drop=True, inplace=True)
    df_sound.reset_index(drop=True, inplace=True)
    features = pd.concat([df_images, df_sound], axis = 1)

    # perform z-score normalization
    # features = (features - features.mean()) / features.std()

    return features

def extract_features_concat():
    list_movies = ['AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'FirstBite', 'LessonLearned', 'Payload', 'Sintel', 'Spaceman', 'Superhero', 'TearsOfSteel', 'TheSecretNumber', 'ToClaireFromSonny', 'YouAgain'] 
    concatenated_features = pd.DataFrame()
    for movie in list_movies:
        features = extract_features(movie, columns = ['spectralflux', 'rms', 'zcrs'], 
                                            columns_images = ['average_brightness_left', 'average_saturation_left', 'average_hue_left', 'average_brightness_right', 'average_saturation_right', 'average_hue_right'],
                                                cluster = False
                                                )
        
        # compute the mean of the left and right features
        features['average_brightness'] = (features['average_brightness_left'] + features['average_brightness_right']) / 2
        features['average_saturation'] = (features['average_saturation_left'] + features['average_saturation_right']) / 2
        features['average_hue'] = (features['average_hue_left'] + features['average_hue_right']) / 2

        # drop the left and right features
        features.drop(columns = ['average_brightness_left', 'average_saturation_left', 'average_hue_left', 'average_brightness_right', 'average_saturation_right', 'average_hue_right'], inplace = True)
        
        concatenated_features = pd.concat([concatenated_features, features], axis = 0)
        concatenated_features.reset_index(drop=True, inplace=True)

    # perform z-score normalization
    concatenated_features = (concatenated_features - concatenated_features.mean()) / concatenated_features.std()

    return concatenated_features

nb = 30              # Number of participants
nPer = 1000         # Number of permutations for significance testing
nBoot = 1000        # Number of bootstrap iterations
seed = 10           # Seed for reproducibility
sl = 0.05          # Signficant level for statistical testing
p_star = 0.05
columns = ['DASS_dep', 'DASS_anx', 'DASS_str',	'bas_d', 'bas_f', 'bas_r', 'bis', 'BIG5_ext', 'BIG5_agr', 'BIG5_con', 'BIG5_neu', 'BIG5_ope']

if __name__ == '__main__': 

    # Input arguments
    PATH = sys.argv[1]
    feature = sys.argv[2] # these can be the features too
    PATH_DATA = sys.argv[3]
    region = sys.argv[4]
    threshold = float(sys.argv[5])
    todo = sys.argv[6]

    print('\n' + ' -' * 10 + f' for {feature} and {region} and {threshold} FOR {todo}', ' -' * 10)

    # Load the data Y and concatenated the feature
    Y = pd.read_csv(PATH_DATA, sep='\t', header=0)[columns]
    if todo == 'emotions':
        data_concat = concat_emo()
    if todo == 'features_extracted':
        data_concat = extract_features_concat()

    # Find the times where the generic feature (emotional or extracted) is peaking
    times_peaking = data_concat[f'{feature}'].loc[data_concat[f'{feature}'] > threshold].index
    print('The number of times where there are peaks is: ', len(times_peaking))
    if len(times_peaking) <= 10:
        print(f'There are no peaks for {feature}. We will not perform the PLS.\n')
        sys.exit()

    # Load the boostrapped results from the same region ad movie
    PATH_SAVE = f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/PLS_csv/PLSpeaks_{todo}_concat.csv'
    yeo_dict = loading_yeo(PATH_YEO)

    print('We are doing the peak part')
    # generic feature ----------> results
    X_movie = compute_X_concat(PATH, feature, threshold, control= False)
    X_movie = pd.DataFrame(X_movie)
    results = boostrap_subjects(X_movie, Y, region, sample_size = 25, num_rounds = 10)
    results['Feature'] = feature
    results['threshold'] = threshold

    print('We are doing the control')
    # control of the generic feature ---------------> results_controls
    results_control = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 'bootstrap_round', 'Feature', 'threshold'])
    for i in range(100):
        X_movie = compute_X_concat(PATH, feature, threshold, control=True, seed = 5 * i, mean = False)
        X_movie = pd.DataFrame(X_movie)
        results_control_i = boostrap_subjects(X_movie, Y, region, sample_size = 30, num_rounds = 1)
        results_control_i['Feature'] = f'Control_{i}_{feature}'
        results_control_i['threshold'] = threshold
        results_control = pd.concat([results_control, results_control_i], axis=0)
    
    # concatentate the feature and the control group
    results = pd.concat([results, results_control], axis=0)

    # save the results
    if os.path.exists(PATH_SAVE):
        PLS_results = pd.read_csv(PATH_SAVE)
    else:
        PLS_results = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 'bootstrap_round', 'Feature', 'threshold'])

    PLS_results = pd.concat([PLS_results, results], axis=0)
    PLS_results.to_csv(PATH_SAVE, index=False)

    print('\n' + f"------------ The PLS for {feature} peak, all movies concatenated and {region} was performed!!! ------------ \n")