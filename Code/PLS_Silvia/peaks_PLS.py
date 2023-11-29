import numpy as np
import os
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

if __name__ == '__main__': 

    # Input arguments
    PATH = sys.argv[1]
    feature = sys.argv[2] # these can be the features too
    PATH_DATA = sys.argv[3]
    region = sys.argv[4]
    threshold = float(sys.argv[5])
    todo = sys.argv[6]
    PATH_SAVE = f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/PLS_csv/PLSpeaks_{todo}_concat.csv'

    print('\n' + ' -' * 10 + f' for {feature} and {region} and {threshold} FOR {todo}', ' -' * 10)

    # Load the data Y and concatenated the feature
    Y = pd.read_csv(PATH_DATA, sep='\t', header=0)[columns]
    if todo == 'emotions':
        print('We are concatenating the emotions')
        data_concat = concat_emo()
    if todo == 'features_extracted':
        print('We are concatenating the features extracted')
        data_concat = extract_features_concat()
        print(data_concat.head())

    # Find the times where the generic feature (emotional or extracted) is peaking
    times_peaking = data_concat[f'{feature}'].loc[data_concat[f'{feature}'] > threshold].index
    print('The number of times where there are peaks is: ', len(times_peaking))
    if len(times_peaking) <= 10:
        print(f'There are no peaks for {feature}. We will not perform the PLS.\n')
        sys.exit()

    # Load the boostrapped results from the same region ad movie
    yeo_dict = loading_yeo(PATH_YEO)

    print('We are doing the peak part')
    # generic feature ----------> results
    X_movie = compute_X_concat(PATH, feature, threshold, control= False, todo = todo, mean = False)
    X_movie = pd.DataFrame(X_movie)
    results = boostrap_subjects(X_movie, Y, region, sample_size = 25, num_rounds = 50)
    results['Feature'] = feature
    results['threshold'] = threshold

    print('We are doing the control')
    # control of the generic feature ---------------> results_controls
    results_control = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 'bootstrap_round', 'Feature', 'threshold'])
    for i in range(30):
        X_movie = compute_X_concat(PATH, feature, threshold, control=True, seed = 5 * i, todo = todo, mean = False)
        X_movie = pd.DataFrame(X_movie)
        results_control_i = boostrap_subjects(X_movie, Y, region, sample_size = 25, num_rounds = 5)
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