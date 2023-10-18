from PLS_run_script import *
import numpy as np
import h5py 
import pandas as pd
from behavPLS import BehavPLS
from configparser import ConfigParser
from argparse import ArgumentParser
from scipy.io import loadmat
import seaborn as sns
import pickle
import glob
from matplotlib import pyplot as plt
import sys 
from itertools import combinations

def compute_X(PATH, movie, method, regions = None):
    """
    Compute the X dataset for the PLS analysis

    -------------------------------------
    Input:
    - param PATH: path of the subjects' data
    - param movie: movie to consider
    -------------------------------------
    Output:
    - X: X dataset
    """
    yeo_dict = loading_yeo(PATH_YEO)
    yeo_indices = yeo_dict[regions] if regions != 'ALL' else None
    N = 114 if region == 'ALL' else len(yeo_indices)

    if method == 'bold':
        list_subjects = []
        for i in glob.glob(PATH+'*'):
            if (i.split('/')[-1].split('-')[0] == 'TC_114_sub') & (i.split('/')[-1].split('-')[1].endswith(f'{movie}.txt')):
                list_subjects.append(i)
        for i, PATH_SUBJ in enumerate(list_subjects):
            data_feature = pd.read_csv(PATH_SUBJ, sep=' ', header=None).T
            connectivity_matrix = np.corrcoef(data_feature, rowvar=False)
            connectivity_matrix = np.array(connectivity_matrix)
            if i == 0:
                X = connectivity_matrix
            else:
                X += connectivity_matrix
        X = X / len(list_subjects)
        print('The shape of X for BOLD is: ', X.shape)
    
    elif method == 'scaffold':
        scaffold_current=np.zeros((30,int(N*(N-1)/2)))
        for i in glob.glob(PATH+'*'):
            if (i.split('/')[-1].split('-')[0] == 'Scaffold_frequency_TC_114_sub') & (i.split('/')[-1].split('-')[1].endswith(f'{movie}.hd5')):
                file=h5py.File(i,'r',swmr=True)
                u,v=np.triu_indices(n=N,k=1)
                subjID = int(i.split('/')[-1].split('-')[1][1:3]) - 1
                if subjID > 10:
                    if subjID > 16:
                        subjID -= 2
                    else:
                        subjID -= 1
                for t in range(1,len(file)+1):
                    print('The shape of the file is: ', file[str(t)][:].shape)
                    scaffold_current[subjID,:]+=file[str(t)][:][u,v]
                scaffold_current[subjID]=scaffold_current[subjID]/len(file)
        X = scaffold_current.copy()
        print('The shape of X for SCAFFOLD is: ', X.shape)

    elif method == 'triangles':

        if regions == 'ALL':
            length = int((114 * (114-1) * (114-2)) / (3*2))
        else:
            indices_yeo_all = []
            for idx_triangles,(i,j,k) in enumerate(combinations(np.arange(114),3)):
                flag=[i in yeo_indices, j in yeo_indices, k in yeo_indices]
                if sum(flag) == 3: ## All the nodes belong to the same Yeo networks
                    indices_yeo_all.append(idx_triangles)
            indices_yeo_all=np.array(indices_yeo_all)
            number_indices = len(yeo_indices)
            length = int((number_indices * (number_indices-1) * (number_indices-2)) / (3*2))

        current_tri = np.zeros((30, length))
        for string in glob.glob(PATH+'*'):
            if (string.endswith(f'{movie}.hd5')):
                file=h5py.File(string,'r',swmr=True)
                subjID = int(string.split('/')[-1].split('_')[4][1:3]) - 1
                if subjID > 10:
                    if subjID > 16:
                        subjID -= 2
                    else:
                        subjID -= 1
                for t in range(0,len(file)):
                    sub_matrix = np.array(file[str(t)][:])
                    print('The shape of sub_matrix is: ', sub_matrix.shape)
                    current_tri[subjID,:]+=sub_matrix
                current_tri[subjID]=current_tri[subjID]/len(file)
                print('The shape of current_tri is: ', current_tri.shape)
        X = current_tri.copy()
        print('The shape of X for TRIANGLES is: ', X.shape)

    elif method == 'edges':
        list_subjects = []
        for i in glob.glob(PATH+'*'):
            if (i.split('/')[-1].split('-')[0] == 'TC_114_sub') & (i.split('/')[-1].split('-')[1].endswith(f'{movie}.txt')):
                list_subjects.append(i)
        for i, PATH_SUBJ in enumerate(list_subjects):
            data_feature = pd.read_csv(PATH_SUBJ, sep=' ', header=None).T
            data_feature = np.array(data_feature)
            u, v = np.triu_indices(data_feature.shape[0], k=1)                
            edge_file_array = data_feature[u,:] * data_feature[v,:]
            connectivity_matrix = np.corrcoef(edge_file_array, rowvar=False)
            if i == 0:
                X = connectivity_matrix
            else:
                X += connectivity_matrix
        X = X / len(list_subjects)
        print(f'The shape of X for EDGES for {region} is: ', X.shape)

    return X

if __name__ == '__main__':
    PATH = sys.argv[1]
    movie_name = sys.argv[2]
    method = sys.argv[3]
    PATH_DATA = sys.argv[4]
    region = sys.argv[5]

    yeo_dict = loading_yeo(PATH_YEO)

    name_of_region = 'ALL' if region == 'ALL' else region

    print('\n' + ' -' * 10 + f' for {method}, {movie_name} and {name_of_region} FOR: ', movie_name, ' -' * 10)

    X_movie = compute_X(PATH, movie_name, method=method, regions = region)
    X_movie = pd.DataFrame(X_movie)
    print('The shape of the X movie is: ', X_movie.shape)

    # save the X matrix in /media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/FMRI 
    PATH_X = '/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/FMRI/' + f'X_{method}_{movie_name}_{region}.pkl'
    pickle.dump(X_movie, open(PATH_X, 'wb'))
    print('The X matrix has been saved in: ', PATH_X)
