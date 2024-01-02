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
max_iterations = 5000
columns = ['DASS_dep', 'DASS_anx', 'DASS_str',	'bas_d', 'bas_f', 'bas_r', 'bis', 'BIG5_ext', 'BIG5_agr', 'BIG5_con', 'BIG5_neu', 'BIG5_ope']
control = False

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

    print('\n' + ' -' * 10 + f' - {feature} - {region} - {number_points} - {todo} - {movie_name} - {concatmovies} - {method}', ' -' * 10)

    # ------------------------- Paths -------------------------
    if server == 'miplab':
        PATH_SAVE = f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/peaks/{method.capitalize()}/PLSpeaks_{todo}_{concatmovies}_{method}_{region}_pts.csv'
        PATH_LABELS = f'/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/Annot13_{movie_name}_stim.json'
        PATH_DATA = f'/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/Annot13_{movie_name}_stim.tsv'
        PATH_YEO = '/media/miplab-nas2/Data2/Movies_Emo/Silvia/HigherOrder/Data/yeo_RS7_Schaefer100S.mat'
    else:
        PATH_SAVE = f'/storage/Projects/lab_projects/Silvia/Output/PLS_csv/PLSpeaks_{todo}_{concatmovies}_{method}_{region}_pts.csv'
        PATH_LABELS = f'/home/silvia/Flavia_E3/EmoData/Annot13_{movie_name}_stim.json'
        PATH_DATA = f'/home/silvia/Flavia_E3/EmoData/Annot13_{movie_name}_stim.tsv'
        PATH_YEO = '/home/silvia/Silvia/HigherOrder/Data/yeo_RS7_Schaefer100S.mat'

    Y, data, threshold_values, minimum_points = get_data(concatmovies, todo, movie_name, columns, server)
    yeo_dict = loading_yeo(PATH_YEO)

    # ------------------------ Get threshold ------------------------

    threshold, times_peaking = get_threshold(threshold_values, number_points)

    # ------------------------ Compute the X matrix ------------------------

    X_movie = get_x(concatmovies)

    # ------------------------ Compute the PLS ------------------------
    
    results = boostrap_subjects(X_movie, Y, region, movie_name, sample_size = 25, num_rounds = bootstrap_rounds)
    results = add_columns(results)
    
    # ------------------------ Compute the control ------------------------
    if control:

        results_control = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 'bootstrap_round', 'Feature', 'threshold', 'Number of points', 'Method'])

        for i in range(bootstrap_rounds):

            # X matrix
            X_movie = get_x(concatmovies)

            # Control
            results_control_i = boostrap_subjects(X_movie, Y, region, movie_name, sample_size = 25, num_rounds = 2)
            results_control_i = add_columns(results_control_i)

            # Concatenate the results
            results_control = pd.concat([results_control, results_control_i], axis=0)
        
        # Concatenate the results
        results = pd.concat([results, results_control], axis=0)

    # ------------------------ Save the results ------------------------
    if os.path.exists(PATH_SAVE):
        PLS_results = pd.read_csv(PATH_SAVE)
    else:
        PLS_results = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 'bootstrap_round', 'Feature', 'threshold', 'Number of points', 'Method'])

    PLS_results = pd.concat([PLS_results, results], axis=0)
    PLS_results.to_csv(PATH_SAVE, index=False)

    # ------------------------ Finished ------------------------
    print('\n' + f"------------ The PLS for {feature} - {region} - {number_points} - {todo} - {movie_name} - {concatmovies} - {method} - was performed!!! ------------ \n")