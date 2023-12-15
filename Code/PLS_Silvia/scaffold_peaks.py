import numpy as np
import os
import pandas as pd
from scipy.io import loadmat
from compute import *
from helpers_scaffolds import *
import sys 

nb = 30              # Number of participants
nPer = 1000         # Number of permutations for significance testing
nBoot = 1000        # Number of bootstrap iterations
seed = 10           # Seed for reproducibility
sl = 0.05          # Signficant level for statistical testing
p_star = 0.05
columns = ['DASS_dep', 'DASS_anx', 'DASS_str',	'bas_d', 'bas_f', 'bas_r', 'bis', 'BIG5_ext', 'BIG5_agr', 'BIG5_con', 'BIG5_neu', 'BIG5_ope']

if __name__ == '__main__': 

    # ------------------------- Input arguments -------------------------
    PATH = sys.argv[1]
    print(PATH)
    feature = sys.argv[2] # these can be the features too
    PATH_DATA_Y = sys.argv[3]
    region = sys.argv[4]
    number_points = int(sys.argv[5])
    todo = sys.argv[6]
    concatmovies = sys.argv[7]
    movie_name = sys.argv[8]
    bootstrap_rounds = int(int(sys.argv[9]))
    method = sys.argv[10]
    server = sys.argv[11]

    # ------------------------- Paths -------------------------
    if server == 'miplab':
        PATH_SAVE = f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/PLS_csv/PLSpeaks_{todo}_{concatmovies}_{method}_{region}_pts.csv'
        PATH_LABELS = f'/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/Annot13_{movie_name}_stim.json'
        PATH_DATA = f'/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/Annot13_{movie_name}_stim.tsv'
        PATH_YEO = '/media/miplab-nas2/Data2/Movies_Emo/Silvia/HigherOrder/Data/yeo_RS7_Schaefer100S.mat'
    else:
        PATH_SAVE = f'/storage/Projects/lab_projects/Silvia/Output/PLS_csv/PLSpeaks_{todo}_{concatmovies}_{method}_{region}_pts.csv'
        PATH_LABELS = f'/home/silvia/Flavia_E3/EmoData/Annot13_{movie_name}_stim.json'
        PATH_DATA = f'/home/silvia/Flavia_E3/EmoData/Annot13_{movie_name}_stim.tsv'
        PATH_YEO = '/home/silvia/Silvia/HigherOrder/Data/yeo_RS7_Schaefer100S.mat'

    print('\n' + ' -' * 10 + f' - {feature} - {region} - {number_points} - {todo} - {movie_name} - {concatmovies} - {method}', ' -' * 10)
    if concatmovies == 'concat':
        minimum_points = 30
    elif concatmovies == 'single':
        minimum_points = 15

    Y = pd.read_csv(PATH_DATA_Y, sep='\t', header=0)[columns]
    labels = pd.read_json(PATH_LABELS)
    if concatmovies == 'concat':
        data = concat_emo(server=server) if todo == 'emotions' else extract_features_concat(server = server) if todo == 'features_extracted' else None
    elif concatmovies == 'single':
        data = pd.read_csv(PATH_DATA, sep = '\t', header = None) if todo == 'emotions' else extract_features(movie_name, columns = ['spectralflux', 'rms', 'zcrs'], columns_images = ['average_brightness_left', 'average_saturation_left', 'average_hue_left', 'average_brightness_right', 'average_saturation_right', 'average_hue_right'], server = server) if todo == 'features_extracted' else None
        data.columns = labels['Columns']
    
    threshold_values = {
        'concat': {150: 1, 100: 1.5, 50: 2, 35: 2.5},
        'single': {70: 0.5, 50: 0.7, 35: 1, 20: 1.5}
    }
    threshold = threshold_values.get(concatmovies, {}).get(number_points)

    max_iterations = 1000
    tolerance_range = range(number_points - 5, number_points + 6)
    def get_times_peaking(threshold):
        return data[f'{feature}'].loc[data[f'{feature}'] > threshold].index

    times_peaking = get_times_peaking(threshold)
    for count in range(max_iterations):
        num_peaks = len(times_peaking)
        if num_peaks in tolerance_range:
            break
        if num_peaks < number_points - 5:
            threshold -= 0.01
        elif num_peaks > number_points + 5:
            threshold += 0.01
        times_peaking = get_times_peaking(threshold)
    else:
        print('Too many iterations, we will not perform the PLS.\n')
        sys.exit()
    threshold = round(threshold, 2)
    print('The number of times where there are peaks is: ', len(times_peaking), 'and the threshold is: ', threshold)

    # Load the boostrapped results from the same region ad movie
    yeo_dict = loading_yeo(PATH_YEO)

    print('\nWe are doing the peak part')
    if concatmovies == 'concat':
        X_movie = compute_X_concat(PATH, feature, threshold, PATH_YEO, control= False, todo = todo, mean = False, server = server)
    elif concatmovies == 'single':
        X_movie = compute_X_withtimes(PATH, movie_name, times_peaking, method = method, PATH_YEO = PATH_YEO, regions = region)
    X_movie = pd.DataFrame(X_movie)
    results = boostrap_subjects(X_movie, Y, region, movie_name, sample_size = 25, num_rounds = bootstrap_rounds)
    results['Feature'] = feature
    results['threshold'] = threshold
    results['Number of points'] = number_points
    results['Method'] = method

    print('\nWe are doing the control')
    results_control = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 
                                                'bootstrap_round', 'Feature', 'threshold', 'Number of points', 'Method'])
    for i in range(bootstrap_rounds):
        print(f'\nControl {i}')
        if concatmovies == 'concat':
            X_movie = compute_X_concat(PATH, feature, threshold, PATH_YEO, control=True, seed = 5 * i, todo = todo, mean = False, server = server)
        elif concatmovies == 'single':
            X_movie = compute_X_withtimes(PATH, movie_name, times_peaking, method = method, PATH_YEO = PATH_YEO, regions = region)
        X_movie = pd.DataFrame(X_movie)
        results_control_i = boostrap_subjects(X_movie, Y, region, movie_name, sample_size = 25, num_rounds = 5)
        results_control_i['Feature'] = f'Control_{i}_{feature}'
        results_control_i['threshold'] = threshold
        results_control_i['Number of points'] = number_points
        results_control_i['Method'] = method
        results_control = pd.concat([results_control, results_control_i], axis=0)
    
    results = pd.concat([results, results_control], axis=0)

    if os.path.exists(PATH_SAVE):
        PLS_results = pd.read_csv(PATH_SAVE)
    else:
        PLS_results = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 
                                                'bootstrap_round', 'Feature', 'threshold', 'Number of points', 'Method'])

    PLS_results = pd.concat([PLS_results, results], axis=0)
    PLS_results.to_csv(PATH_SAVE, index=False)

    print('\n' + f"------------ The PLS for {feature} - {region} - {number_points} - {todo} - {movie_name} - {concatmovies} - {method} - was performed!!! ------------ \n")