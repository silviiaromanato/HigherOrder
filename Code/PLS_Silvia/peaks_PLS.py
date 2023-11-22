import numpy as np
import pandas as pd
from scipy.io import loadmat
from compute import *
from helpers_PLS import *
import sys 

import warnings
warnings.simplefilter('ignore', DeprecationWarning)

PATH_YEO = '/media/miplab-nas2/Data2/Movies_Emo/Silvia/HigherOrder/Data/yeo_RS7_Schaefer100S.mat'

def compute_X_withtimes(PATH, movie, times, regions = None):

    yeo_dict = loading_yeo(PATH_YEO)
    yeo_indices = yeo_dict[regions] if regions != 'ALL' else None
    N = 114 if regions == 'ALL' else len(yeo_indices)

    list_subjects = []
    for i in glob.glob(PATH+'*'):
        if (i.split('/')[-1].split('-')[0] == 'TC_114_sub') & (i.split('/')[-1].split('-')[1].endswith(f'{movie}.txt')):
            list_subjects.append(i)
    mtx_upper_triangular = []
    for i, PATH_SUBJ in enumerate(list_subjects):
        data_feature = pd.read_csv(PATH_SUBJ, sep=' ', header=None)
        data_feature = data_feature.iloc[times,:]
        if regions == 'ALL':
            connectivity_matrix = np.corrcoef(data_feature, rowvar=False)
        else:
            connectivity_matrix = np.corrcoef(data_feature, rowvar=False)[:,yeo_indices]
        upper_triangular = connectivity_matrix[np.triu_indices_from(connectivity_matrix, k=1)]
        mtx_upper_triangular.append(upper_triangular)
    mtx_upper_triangular = np.array(mtx_upper_triangular)
    X = pd.DataFrame(mtx_upper_triangular)
    print('The shape of X for BOLD is: ', X.shape)

    return X

def run_pls(X_movie, Y, region):
    res = run_decomposition(X_movie, Y)
    res_permu = permutation(res, nPer, seed, sl)
    print('The pvalues are: ', res_permu['P_val'])
    results=pd.DataFrame(list(zip(varexp(res['S']),res_permu['P_val'])), columns=['Covariance Explained', 'P-value'])
    results['Movie']=movie_name
    results['LC']=np.arange(1, results.shape[0]+1)
    results['Region'] = region
    results['Covariance Explained'] = results['Covariance Explained'].astype(float)
    return results

def boostrap_subjects(X_movie, Y, region, sample_size = 20, num_rounds = 100):
    print(f'Performing BOOSTRAPPING on {sample_size} subjects for {num_rounds} rounds')
    results = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 'bootstrap_round'])
    for i in range(num_rounds):
        print('The round is: ', i)
        idx = np.random.choice(np.arange(X_movie.shape[0]), size=sample_size, replace=True)
        X_movie_sample = X_movie.iloc[idx,:]
        Y_sample = Y.iloc[idx,:]
        pls = run_pls(X_movie_sample, Y_sample, region)
        pls = pd.DataFrame(pls)
        pls['bootstrap_round'] = i
        results = pd.concat([results, pls], axis=0)
        print('The results are: ', results.head())
    return results

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
    movie_name = sys.argv[2]
    emotion = sys.argv[3]
    PATH_DATA = sys.argv[4]
    region = sys.argv[5]
    threshold = int(sys.argv[6])

    print('\n' + ' -' * 10 + f' for {emotion} and {region} FOR: ', movie_name, ' -' * 10)

    # Load the data
    Y = pd.read_csv(PATH_DATA, sep='\t', header=0)[columns]
    labels = pd.read_json(f'/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/Annot13_{movie_name}_stim.json')
    data = pd.read_csv(f'/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/Annot13_{movie_name}_stim.tsv', sep = '\t', header = None)
    data.columns = labels['Columns']

    # Find the times where the emotion is peaking
    times_peaking = data[f'{emotion}'].loc[data[f'{emotion}'] > threshold].index
    print('The number of times where there are peaks is: ', len(times_peaking))
    if len(times_peaking) == 0:
        print('There are no peaks for this emotion. We will not perform the PLS.\n')
        sys.exit()

    # Load the boostrapped results from the same region ad movie
    PATH_SAVE = f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/PLS_csv/PLSpeaks_withcontrol_results.csv'
    yeo_dict = loading_yeo(PATH_YEO)

    # emotion ----------> results
    X_movie = compute_X_withtimes(PATH, movie_name, times_peaking, regions = region)
    X_movie = pd.DataFrame(X_movie)
    results = boostrap_subjects(X_movie, Y, region, sample_size = 25, num_rounds = 10)
    results['Emotion'] = emotion
    results['threshold'] = threshold
    print('The shape of the results is: ', results.columns)

    # control of the emotion ---------------> results_control
    results_control = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 'bootstrap_round', 'Emotion', 'threshold'])
    for i in range(10):
        random_number = np.random.randint(1, 1001)
        print('The control round is: ', i)
        np.random.seed(i * random_number)
        control_times = np.random.choice(data[f'{emotion}'].index, size=len(times_peaking), replace=False)
        print('The control times are: ', control_times)
        X_movie = compute_X_withtimes(PATH, movie_name, control_times, regions = region)
        X_movie = pd.DataFrame(X_movie)
        results_control_i = boostrap_subjects(X_movie, Y, region, sample_size = 25, num_rounds = 5)
        results_control_i['Emotion'] = f'Control_{i}_{emotion}'
        results_control_i['threshold'] = threshold
        results_control = pd.concat([results_control, results_control_i], axis=0)
        print('The shape of the results_control is: ', results_control.columns, results_control.shape)
    
    # concatentate the emotion and the control group
    results = pd.concat([results, results_control], axis=0)

    # save the results
    if os.path.exists(PATH_SAVE):
        PLS_results = pd.read_csv(PATH_SAVE)
    else:
        PLS_results = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 'bootstrap_round', 'Emotion', 'threshold'])

    print('The movies that PLS was trained on are: ', PLS_results['Movie'].unique())
    print('The movie that is being added is: ', results['Movie'].unique()[0])
    PLS_results = pd.concat([PLS_results, results], axis=0)
    print('The shape of the PLS results is: ', PLS_results.shape)
    PLS_results.to_csv(PATH_SAVE, index=False)

    print('\n' + f"------------ The PLS for {emotion} peak, {movie_name} and {region} was performed!!! ------------ \n")