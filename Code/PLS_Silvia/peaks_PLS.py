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
        print('The shape of the data feature is: ', data_feature.shape)
        # Select only the times of interest
        data_feature = data_feature.iloc[times,:]
        print('The shape of the data feature after the times selection is: ', data_feature.shape)
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

def run_pls(X_movie, Y):
    res = run_decomposition(X_movie, Y)
    res_permu = permutation(res, nPer, seed, sl)
    #res_bootstrap = bootstrap(res, nBoot, seed)
    print('The pvalues are: ', res_permu['P_val'])

    # Save the results
    results=pd.DataFrame(list(zip(varexp(res['S']),res_permu['P_val'])), columns=['Covariance Explained', 'P-value'])
    data_cov_significant=results[results['P-value'] < p_star]
    data_cov_significant.sort_values('P-value')
    results['Movie']=movie_name
    results['LC']=np.arange(1, results.shape[0]+1)
    results['Region'] = region
    results['Covariance Explained'] = results['Covariance Explained'].astype(float)
    return data_cov_significant

def boostrap_subjects(X_movie, Y, sample_size = 20, num_rounds = 100):
    """
    Compute the bootstrap for the subjects

    -------------------------------------
    Input:
    - param X_movie: X dataset
    - param Y: Y dataset
    - param sample_size: number of subjects to sample
    - param num_rounds: number of rounds to perform
    -------------------------------------
    Output:
    - results: results of the boostrap
    """
    print('Performing BOOSTRAPPING on 20 subjects for 100 rounds')
    results = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region'])
    for i in range(100):
        print('The round is: ', i)
        idx = np.random.choice(np.arange(X_movie.shape[0]), size=sample_size, replace=True)
        X_movie_sample = X_movie.iloc[idx,:]
        Y_sample = Y.iloc[idx,:]
        pls = run_pls(X_movie_sample, Y_sample)
        print('The shape of the pls is: ', pls)
        # convert PLS to a dataframe
        pls = pd.DataFrame(pls)

        # concatenate on the veritcal axis
        results = pd.concat([results, pls], axis=0)
    return results

nb = 30              # Number of participants
nPer = 1000         # Number of permutations for significance testing
nBoot = 1000        # Number of bootstrap iterations
seed = 10           # Seed for reproducibility
sl = 0.05          # Signficant level for statistical testing
p_star = 0.05
columns = ['DASS_dep', 'DASS_anx', 'DASS_str',	'bas_d', 'bas_f', 'bas_r', 'bis', 'BIG5_ext', 'BIG5_agr', 'BIG5_con', 'BIG5_neu', 'BIG5_ope']

if __name__ == '__main__': 
    PATH = sys.argv[1]
    movie_name = sys.argv[2]
    emotion = sys.argv[3]
    PATH_DATA = sys.argv[4]
    region = sys.argv[5]

    # Load the peaks of the emotion in question
    print('The emotion is: ', emotion, 'computing the times where the peaks are...')
    labels = pd.read_json(f'/Users/silviaromanato/Desktop/SEMESTER_PROJECT/Material/Data/EmoData/Annot13_{movie_name}_stim.json')[['Columns']]
    data = pd.read_csv(f'/Users/silviaromanato/Desktop/SEMESTER_PROJECT/Material/Data/EmoData/Annot13_{movie_name}_stim.tsv', sep = '\t', header = None)
    data.columns = labels['Columns']
    times_peaking = data[f'{emotion}'].loc[data[f'{emotion}'] > 1].index

    # Load the boostrapped results from the same region ad movie
    PATH_SAVE = f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/PLS_csv/PLSpeaks_{emotion}_{region}_results.csv'
    print('The path of the PLS results is: ', PATH_SAVE, 'It exists?', os.path.exists(PATH_SAVE))
    if os.path.exists(PATH_SAVE):
        PLS_results = pd.read_csv(PATH_SAVE)
        print('The shape of the PLS results is: ', PLS_results.shape)
        movies_done = PLS_results['Movie'].unique()
        print('The movies that PLS was trained on are: ', movies_done)
        print('Each movie has the following number of LCs: ', PLS_results.groupby('Movie').count()['LC'])

        if movie_name in movies_done:
            print('The movie was already done. We will not perform the PLS.')
            sys.exit()

    yeo_dict = loading_yeo(PATH_YEO)

    # Load the Y behavioural dataset
    Y = pd.read_csv(PATH_DATA, sep='\t', header=0)[columns]

    print('\n' + ' -' * 10 + f' for {emotion}, {movie_name} and {region} FOR: ', movie_name, ' -' * 10)

    X_movie = compute_X_withtimes(PATH, movie_name, times_peaking, regions = region)
    X_movie = pd.DataFrame(X_movie)
    print('The shape of the X movie is: ', X_movie.shape)

    # Perform the PLSC Behavioural analysis
    res = run_decomposition(X_movie, Y)
    res_permu = permutation(res, nPer, seed, sl)
    res_bootstrap = bootstrap(res, nBoot, seed)
    print('The pvalues are: ', res_permu['P_val'])

    # Save the results
    results=pd.DataFrame(list(zip(varexp(res['S']),res_permu['P_val'])), columns=['Covariance Explained', 'P-value'])
    data_cov_significant=results[results['P-value'] < p_star]
    data_cov_significant.sort_values('P-value')
    results['Movie']=movie_name
    results['LC']=np.arange(1, results.shape[0]+1)
    results['Region'] = region
    results['Covariance Explained'] = results['Covariance Explained'].astype(float)

    if os.path.exists(PATH_SAVE):
        PLS_results = pd.read_csv(PATH_SAVE)
    else:
        PLS_results = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region'])

    print('The shape of the PLS results is: ', PLS_results.shape)
    movies_done = PLS_results['Movie'].unique()
    print('The movies that PLS was trained on are: ', movies_done)
    print('Each movie has the following number of LCs: ', PLS_results.groupby('Movie').count()['LC'])

    movie_making = results['Movie'].unique()[0]
    print('The movie that is being added is: ', movie_making)

    PLS_results = pd.concat([PLS_results, results], axis=0)
    print('The shape of the PLS results is: ', PLS_results)
    PLS_results.to_csv(PATH_SAVE, index=False)

    print('\n' + f"------------ The PLS for {emotion} peak, {movie_name} and {region} was performed!!! ------------")