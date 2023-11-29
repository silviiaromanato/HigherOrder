import numpy as np
import pandas as pd
from scipy.io import loadmat
from compute import *
from helpers_PLS import *
import sys 

PATH_YEO = '/media/miplab-nas2/Data2/Movies_Emo/Silvia/HigherOrder/Data/yeo_RS7_Schaefer100S.mat'

nb = 30              # Number of participants
nPer = 1000         # Number of permutations for significance testing
nBoot = 1000        # Number of bootstrap iterations
seed = 10           # Seed for reproducibility
sl = 0.05          # Signficant level for statistical testing
p_star = 0.05
columns = ['DASS_dep', 'DASS_anx', 'DASS_str',	'bas_d', 'bas_f', 'bas_r', 'bis', 'BIG5_ext', 'BIG5_agr', 'BIG5_con', 'BIG5_neu', 'BIG5_ope']

to_do = {
    'control': True, 
    'emo': False
}

if __name__ == '__main__': 

    # Input arguments
    PATH = sys.argv[1]
    emotion = sys.argv[2]
    PATH_DATA = sys.argv[3]
    region = sys.argv[4]
    threshold = float(sys.argv[5])

    print('\n' + ' -' * 10 + f' for {emotion} and {region} and {threshold} FOR', ' -' * 10)

    # Load the data Y and concatenated the emotions
    Y = pd.read_csv(PATH_DATA, sep='\t', header=0)[columns]
    data_concat = concat_emo()

    # Find the times where the emotion is peaking
    times_peaking = data_concat[f'{emotion}'].loc[data_concat[f'{emotion}'] > threshold].index
    print('The number of times where there are peaks is: ', len(times_peaking))
    if len(times_peaking) <= 10:
        print('There are no peaks for this emotion. We will not perform the PLS.\n')
        sys.exit()

    # Load the boostrapped results from the same region ad movie
    PATH_SAVE = f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/PLS_csv/PLSpeaks_concat.csv'
    yeo_dict = loading_yeo(PATH_YEO)

    if to_do['emo'] == True:
        print('We are doing the emotion peak')
        # emotion ----------> results
        X_movie = compute_X_concat(PATH, emotion, threshold, control= False)
        X_movie = pd.DataFrame(X_movie)
        results = boostrap_subjects(X_movie, Y, region, sample_size = 25, num_rounds = 10)
        results['Emotion'] = emotion
        results['threshold'] = threshold
        print('The shape of the results is: ', results.columns, results.head())
    else:
        results = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 'bootstrap_round', 'Emotion', 'threshold'])

    if to_do['control'] == True:
        print('We are doing the control')
        # control of the emotion ---------------> results_controls
        results_control = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 'bootstrap_round', 'Emotion', 'threshold'])
        for i in range(100):
            X_movie = compute_X_concat(PATH, emotion, threshold, control=True, seed = 5 * i, mean = False)
            X_movie = pd.DataFrame(X_movie)
            results_control_i = boostrap_subjects(X_movie, Y, region, sample_size = 30, num_rounds = 1)
            results_control_i['Emotion'] = f'Control_{i}_{emotion}'
            results_control_i['threshold'] = threshold
            results_control = pd.concat([results_control, results_control_i], axis=0)
    else:
        results_control = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 'bootstrap_round', 'Emotion', 'threshold'])
    
    # concatentate the emotion and the control group
    results = pd.concat([results, results_control], axis=0)

    # save the results
    if os.path.exists(PATH_SAVE):
        PLS_results = pd.read_csv(PATH_SAVE)
        # take only the results for the region and emotion and threshold
        PLS_results = PLS_results.loc[(PLS_results['Emotion'] == emotion) & (PLS_results['threshold'] == threshold) & (PLS_results['Region'] == region)]
        print('The shape of the PLS results before removing the control for the task is: ', PLS_results.shape)
        PLS_results = PLS_results.loc[~PLS_results['Emotion'].str.contains('Control')]
        print('The shape of the PLS results is: ', PLS_results.shape)
    else:
        PLS_results = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 'bootstrap_round', 'Emotion', 'threshold'])

    PLS_results = pd.concat([PLS_results, results], axis=0)
    PLS_results.to_csv(PATH_SAVE, index=False)

    print('\n' + f"------------ The PLS for {emotion} peak, all movies concatenated and {region} was performed!!! ------------ \n")