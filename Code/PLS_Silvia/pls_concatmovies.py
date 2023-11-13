import numpy as np
import pandas as pd
from scipy.io import loadmat
from compute import *
from helpers_PLS import *
import sys 

import warnings
warnings.simplefilter('ignore', DeprecationWarning)

PATH_YEO = '/media/miplab-nas2/Data2/Movies_Emo/Silvia/HigherOrder/Data/yeo_RS7_Schaefer100S.mat'
list_movies = ['AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'FirstBite', 'LessonLearned', 'Payload', 'Rest', 'Sintel', 'Spaceman', 'Superhero', 'TearsOfSteel', 'TheSecretNumber', 'ToClaireFromSonny', 'YouAgain']

def run_pls(X_movie, Y):
    res = run_decomposition(X_movie, Y)
    res_permu = permutation(res, nPer, seed, sl)
    # res_bootstrap = bootstrap(res, nBoot, seed)
    print('The pvalues are: ', res_permu['P_val'])
    results=pd.DataFrame(list(zip(varexp(res['S']),res_permu['P_val'])), columns=['Covariance Explained', 'P-value'])
    data_cov_significant=results[results['P-value'] < p_star]
    data_cov_significant.sort_values('P-value')
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
    print(f'Performing BOOSTRAPPING on 20 subjects for {num_rounds} rounds')
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

def compute_X(PATH, list_movies, regions = None):
    
    yeo_dict = loading_yeo(PATH_YEO)
    yeo_indices = yeo_dict[regions] if regions != 'ALL' else None
    N = 114 if regions == 'ALL' else len(yeo_indices)

    # create a dictionary called list_subjects where the key is a string of a number and  the value is a list
    dict_movies = {}
    for movie in list_movies:
        list_X = []
        for i in glob.glob(PATH+'*'):
            if (i.split('/')[-1].split('-')[0] == 'TC_114_sub') & (i.split('/')[-1].split('-')[1].endswith(f'{movie}.txt')):
                list_X.append(i)
        dict_movies[movie] = list_X

    # create a dataframe from the dictionary
    data_subjects = pd.DataFrame.from_dict(dict_movies) # shape (30, 15)
    data_subjects.reset_index(drop=True, inplace=True)

    list_datafeatures = []
    for subject in range(data_subjects.shape[0]):
        list_df = []
        for movie in range(data_subjects.shape[1]):
            data_features = pd.read_csv(data_subjects.iloc[subject, movie], sep=' ', header=None)
            list_df.append(data_features)
        data_features_concat = pd.concat(list_df, axis=0)
        list_datafeatures.append(data_features_concat)
    print('The length of list_datafeatures is: ', len(list_datafeatures))
    print('The shape of list_datafeatures[0] is: ', list_datafeatures[0].shape)

    mtx_upper_triangular = []
    for data_feature in list_datafeatures:
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


nb = 30              # Number of participants
nPer = 1000         # Number of permutations for significance testing
nBoot = 1000        # Number of bootstrap iterations
seed = 10           # Seed for reproducibility
sl = 0.05          # Signficant level for statistical testing
p_star = 0.05
columns = ['DASS_dep', 'DASS_anx', 'DASS_str',	'bas_d', 'bas_f', 'bas_r', 'bis', 'BIG5_ext', 'BIG5_agr', 'BIG5_con', 'BIG5_neu', 'BIG5_ope']

if __name__ == '__main__': 
    PATH = sys.argv[1]
    PATH_DATA = sys.argv[2]
    region = sys.argv[3]
    number_of_rounds = sys.argv[4]
    method = "bold"
    list_movies = ['AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'FirstBite', 'LessonLearned', 'Payload', 'Rest', 'Sintel', 'Spaceman', 'Superhero', 'TearsOfSteel', 'TheSecretNumber', 'ToClaireFromSonny', 'YouAgain']
    PATH_SAVE = f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/PLS_csv/PLS_bold_{region}_concatmovies.csv'
    print('The path of the PLS results is: ', PATH_SAVE, 'It exists?', os.path.exists(PATH_SAVE))

    yeo_dict = loading_yeo(PATH_YEO)
    Y = pd.read_csv(PATH_DATA, sep='\t', header=0)[columns]

    X_movie = compute_X(PATH, list_movies, regions = region)
    X_movie = pd.DataFrame(X_movie)

    PLS_results = boostrap_subjects(X_movie, Y, sample_size = 20, num_rounds = 10)
    print('The head of PLS_results is: ', PLS_results.shape)
    PLS_results.to_csv(PATH_SAVE, index=False)

    print('\n' + f"------------ The PLS for {method} and {region} for all the movies concatenated was performed!!! ------------")