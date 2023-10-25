import numpy as np
import pandas as pd
from scipy.io import loadmat
from compute import *
from helpers_PLS import *
import sys 

import warnings
warnings.simplefilter('ignore', DeprecationWarning)

PATH_YEO = '/media/miplab-nas2/Data2/Movies_Emo/Silvia/HigherOrder/Data/yeo_RS7_Schaefer100S.mat'

def loading_yeo(path=PATH_YEO):
    ##Loading the yeoROIS
    yeoROIs=np.array([i[0]-1 for i in loadmat(path)['yeoROIs']])
    yeoROI_dict={label_Yeo:np.where(yeoROIs==idx_Yeo)[0] for idx_Yeo,label_Yeo in enumerate(['VIS','SM','DA','VA','L','FP','DMN'])}
    yeoROI_dict['SC'] = np.arange(100, 114)
    return(yeoROI_dict)

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
    method = sys.argv[3]
    PATH_DATA = sys.argv[4]
    region = sys.argv[5]


    # Load the boostrapped results from the same region ad movie
    PATH_SAVE = f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/PLS_{method}_{region}_results.csv'
    print('The path of the PLS results is: ', PATH_SAVE, 'It exists?', os.path.exists(PATH_SAVE))
    if os.path.exists(PATH_SAVE):
        PLS_results = pd.read_csv(PATH_SAVE)
        print('The shape of the PLS results is: ', PLS_results.shape)
        movies_done = PLS_results['Movie'].unique()
        print('The movies that PLS was trained on are: ', movies_done)
        print('Each movie has the following number of LCs: ', PLS_results.groupby('Movie').count()['LC'])

        if movie_name in movies_done:
            print('The movie was already done. We will not perform the boostrapping')
            sys.exit()

    yeo_dict = loading_yeo(PATH_YEO)

    # Load the Y behavioural dataset
    Y = pd.read_csv(PATH_DATA, sep='\t', header=0)[columns]

    print('\n' + ' -' * 10 + f' for {method}, {movie_name} and {region} FOR: ', movie_name, ' -' * 10)

    X_movie = compute_X(PATH, movie_name, method=method, regions = region)
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

    # Save the results
    PATH_SAVE = f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/PLS_bootstrap/PLS_{method}_{region}_bootstrap_results.csv'
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

    print('The movie was not done. It will be added')
    PLS_results = pd.concat([PLS_results, results], axis=0)
    print('The shape of the PLS results is: ', PLS_results)
    PLS_results.to_csv(PATH_SAVE, index=False)

    print('\n' + f"------------ The PLS for {method}, {movie_name} and {region} was performed!!! ------------")