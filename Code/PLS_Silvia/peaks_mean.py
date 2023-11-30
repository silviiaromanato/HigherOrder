import numpy as np
import pandas as pd
from scipy.io import loadmat
from compute import *
from helpers_PLS import *
import sys 

PATH_YEO = '/media/miplab-nas2/Data2/Movies_Emo/Silvia/HigherOrder/Data/yeo_RS7_Schaefer100S.mat'
PATH_SAVE = f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/PLS_csv/PLSmean_concat.csv'

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
    emotions_type = sys.argv[2]
    PATH_DATA = sys.argv[3]
    region = sys.argv[4]
    threshold = float(sys.argv[5])

    print('\n' + ' -' * 10 + f' for {emotions_type} emotions and {region} and {threshold} FOR: concatenated movies',' -' * 10)

    # Define the emotions
    if emotions_type == 'positive':
        emotions = ['Love', 'Regard', 'WarmHeartedness', 'Pride', 'Satisfaction', 'Happiness']
    elif emotions_type == 'negative':
        emotions = ['Sad', 'Anxiety', 'Fear', 'Guilt', 'Disgust', 'Anger']

    # Load the data
    Y = pd.read_csv(PATH_DATA, sep='\t', header=0)[columns]
    data_concat = concat_emo()

    # Create the  mean of the emotions
    emo_avg = data_concat[emotions].mean(axis=1)
    times_peaking = emo_avg.loc[emo_avg > threshold].index
    print('The number of times where there are peaks is: ', len(times_peaking))
    threshold_decreased = threshold
    while len(times_peaking) <= 30:
        print(f'There are no peaks for {emotions_type}. We will decrease the threshold.\n')
        threshold_decreased -= 0.05
        times_peaking = emo_avg.loc[emo_avg > threshold_decreased].index
        print('The number of times where there are peaks is: ', len(times_peaking))
        if threshold_decreased == threshold - 0.5:
            print('We have reached the minimum threshold. We will not perform the PLS.\n')
            sys.exit()

    # Load the boostrapped results from the same region ad movie
    yeo_dict = loading_yeo(PATH_YEO)

    # emotion ----------> results
    X_movie = compute_X_concat(PATH, emotions, threshold, control=False, mean = True)
    X_movie = pd.DataFrame(X_movie)
    results = boostrap_subjects(X_movie, Y, region, sample_size = 25, num_rounds = 50)
    results['Emotion'] = emotions_type
    results['threshold'] = threshold

    # Control of the emotion ---------------> results_control
    results_control = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 'bootstrap_round', 'Emotion', 'threshold'])
    for i in range(50):
        X = compute_X_concat(PATH, emotions, threshold, control=True, mean = True, seed = i)
        X = pd.DataFrame(X)
        results_control_bootstrap = boostrap_subjects(X, Y, region, sample_size = 25, num_rounds = 5)
        results_control_bootstrap['Emotion'] = f'Control_{i}_{emotions_type}'
        results_control_bootstrap['threshold'] = threshold
        results_control = pd.concat([results_control, results_control_bootstrap], axis=0)
    
    # Concatentate the emotion and the control group
    results = pd.concat([results, results_control], axis=0)

    # Save the results
    if os.path.exists(PATH_SAVE):
        PLS_results = pd.read_csv(PATH_SAVE)
    else:
        PLS_results = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 'bootstrap_round', 'Emotion', 'threshold'])

    PLS_results = pd.concat([PLS_results, results], axis=0)
    PLS_results.to_csv(PATH_SAVE, index=False)

    print('\n' + f"------------ The PLS for {emotions_type} peak, concatenated movies and {region} was performed!!! ------------ \n")