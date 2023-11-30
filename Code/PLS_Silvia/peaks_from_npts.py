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
    number_points = int(sys.argv[5])
    todo = sys.argv[6]
    concatmovies = sys.argv[7]
    movie_name = sys.argv[8]
    bootstrap_rounds = int(int(sys.argv[9]))
    PATH_SAVE = f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/PLS_csv/PLSpeaks_{todo}_{concatmovies}_pts.csv'

    if concatmovies == 'concat':
        minimum_points = 30
    elif concatmovies == 'single':
        minimum_points = 15

    print('\n' + ' -' * 10 + f' for {feature} and {region} and {number_points} FOR {todo}', ' -' * 10)

    # Load the data Y and concatenated the feature
    Y = pd.read_csv(PATH_DATA, sep='\t', header=0)[columns]
    if concatmovies == 'concat':
        if todo == 'emotions':
            data = concat_emo()
        if todo == 'features_extracted':
            data = extract_features_concat(cluster = True)
    
    elif concatmovies == 'single':
        if todo == 'emotions':
            Y = pd.read_csv(PATH_DATA, sep='\t', header=0)[columns]
            labels = pd.read_json(f'/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/Annot13_{movie_name}_stim.json')
            data = pd.read_csv(f'/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/Annot13_{movie_name}_stim.tsv', sep = '\t', header = None)
            data.columns = labels['Columns']
        if todo == 'features_extracted':
            data = extract_features(movie_name, columns = ['spectralflux', 'rms', 'zcrs'], columns_images = ['average_brightness_left', 'average_saturation_left', 'average_hue_left', 'average_brightness_right', 'average_saturation_right', 'average_hue_right'], cluster = True)

    # Find the times where the generic feature (emotional or extracted) is peaking
    # with the set number of points

    threshold = 2.0
    count = 0
    times_peaking = data[f'{feature}'].loc[data[f'{feature}'] > threshold].index
    while (len(times_peaking) < number_points-5) | (len(times_peaking) > number_points+5):
        count += 1
        if len(times_peaking) < number_points-5:
            print(f'There are no peaks for {feature}. We will increase the threshold.\n')
            threshold += 0.05
        elif len(times_peaking) > number_points+5:
            print(f'There are too many peaks for {feature}. We will decrease the threshold.\n')
            threshold -= 0.05
        elif (len(times_peaking) >= number_points-5) & (len(times_peaking) <= number_points+5):
            print('The number of times where there are peaks is: ', len(times_peaking))
            break
        if count == 100:
            print('Too many iterations, we will not perform the PLS.\n')
            sys.exit()
    
    times_peaking = data[f'{feature}'].loc[data[f'{feature}'] > threshold].index
    print('The number of times where there are peaks is: ', len(times_peaking))

    # Load the boostrapped results from the same region ad movie
    yeo_dict = loading_yeo(PATH_YEO)

    print('\nWe are doing the peak part')
    # generic feature ----------> results
    if concatmovies == 'concat':
        X_movie = compute_X_concat(PATH, feature, threshold, control= False, todo = todo, mean = False)
    elif concatmovies == 'single':
        X_movie = compute_X_withtimes(PATH, movie_name, times_peaking, regions = 'ALL')
    X_movie = pd.DataFrame(X_movie)
    results = boostrap_subjects(X_movie, Y, region, sample_size = 25, num_rounds = bootstrap_rounds)
    results['Feature'] = feature
    results['threshold'] = threshold
    results['Number of points'] = number_points

    print('\nWe are doing the control')
    # control of the generic feature ---------------> results_controls
    results_control = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 'bootstrap_round', 'Feature', 'threshold', 'Number of points'])
    for i in range(bootstrap_rounds):
        print(f'\nControl {i}')
        if concatmovies == 'concat':
            X_movie = compute_X_concat(PATH, feature, threshold, control=True, seed = 5 * i, todo = todo, mean = False)
        elif concatmovies == 'single':
            X_movie = compute_X_withtimes(PATH, movie_name, times_peaking, regions = 'ALL')
        X_movie = pd.DataFrame(X_movie)
        results_control_i = boostrap_subjects(X_movie, Y, region, sample_size = 25, num_rounds = 5)
        results_control_i['Feature'] = f'Control_{i}_{feature}'
        results_control_i['threshold'] = threshold
        results_control_i['Number of points'] = number_points
        results_control = pd.concat([results_control, results_control_i], axis=0)
    
    results = pd.concat([results, results_control], axis=0)

    if os.path.exists(PATH_SAVE):
        PLS_results = pd.read_csv(PATH_SAVE)
    else:
        PLS_results = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 'bootstrap_round', 'Feature', 'threshold', 'Number of points'])

    PLS_results = pd.concat([PLS_results, results], axis=0)
    PLS_results.to_csv(PATH_SAVE, index=False)

    print('\n' + f"------------ The PLS for {feature} peak, {concatmovies} and {region} and {number_points} was performed!!! ------------ \n")