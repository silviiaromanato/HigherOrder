import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import re
import seaborn as sns
from collections import defaultdict



def correlation_mtx_features(movie_name, columns = ['mean_chroma', 'mean_mfcc', 'spectralflux', 'rms', 'zcrs'], columns_images = ['average_brightness_left', 'average_saturation_left', 'average_hue_left']):

    PATH_EMO = f'/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/'
    emo = extract_corrmat_allregressors(PATH_EMO, movie_name)
    length = emo.shape[0]

    movie_name_with_ = re.sub(r"(\w)([A-Z])", r"\1_\2", movie_name)
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
    ############### CORRELATION ################
    correlation_features = Correlation_features(df_sound, df_images, movie_name)

    return correlation_features

def extract_annot(path_folder,film_ID):
    f = open (path_folder+'Annot13_'+film_ID+'_stim.json', "r")
    data_annot = json.loads(f.read())
    annot = pd.read_csv(path_folder+'Annot13_'+film_ID+'_stim.tsv', sep='\t', names=data_annot['Columns'])
    return annot

def extract_corrmat_allregressors(emo_path_folder,film_ID, individual='n', subject=None):
    annot=extract_annot(emo_path_folder, film_ID)
    corrmat_allregressors=np.corrcoef(annot.values)   
    return corrmat_allregressors

def extract_corrmat_emo(emo_path_folder,film_ID):
    positive_emotions = ['Love', 'Regard', 'WarmHeartedness', 'Pride','Satisfaction','Happiness']
    negative_emotions = ['Sad', 'Anxiety', 'Fear', 'Guilt','Disgust','Anger']
    all_emotions = positive_emotions + negative_emotions
    #Extract the required data data from the emodata file
    annot=extract_annot(emo_path_folder,film_ID)
    positive_emotions_matrix = annot[positive_emotions].values
    negative_emotions_matrix = annot[negative_emotions].values
    all_emotions_matrix = annot[all_emotions].values
    #Compute correlation
    corrmat_positive=np.corrcoef(positive_emotions_matrix)   
    corrmat_negative=np.corrcoef(negative_emotions_matrix)   
    corrmat_all_emo=np.corrcoef(all_emotions_matrix)   
    return corrmat_positive, corrmat_negative, corrmat_all_emo

def Correlation_features(df_sound, df_brightness, movie_name): 

    ############### MERGE #################
    df_brightness.reset_index(drop=True, inplace=True)

    # concatenate the two dataframes
    df_sound.reset_index(drop=True, inplace=True)
    df = pd.concat([df_sound, df_brightness], axis=1)
    df.reset_index(drop=True, inplace=True)

    ############### CORRELATION ################
    df = df.T
    corr_features = df.corr()
    corr_features.to_csv(f'/Users/silviaromanato/Desktop/SEMESTER_PROJECT/Material/Output/corr_{movie_name}_features.csv')

    return corr_features

def emo_corr_matrix(movie_name, emotion = 3):
    # emotion can be ['positive', 'negative', 'poistive and negative', 'all']
    PATH_EMO = f'/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/'
    emo = extract_corrmat_emo(PATH_EMO, movie_name)
    if emotion == 3:
        all_emo = extract_corrmat_allregressors(PATH_EMO, movie_name)
    else:
        all_emo = emo[emotion]
    return all_emo

def threshold_matrix_lower_upper(corr_matrix, perc_lower, perc_upper):

    # Flatten the correlation matrix and remove the diagonal entries
    corr_matrix = pd.DataFrame(corr_matrix)
    corr_values = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_values = corr_values.stack()

    # Calculate the 10th and 90th percentiles
    lower_10th = np.percentile(corr_values, perc_lower)
    upper_90th = np.percentile(corr_values, perc_upper)

    transformed_corr_matrix = corr_matrix.applymap(lambda x: 1 if x > upper_90th or x < lower_10th else 0)
    return transformed_corr_matrix