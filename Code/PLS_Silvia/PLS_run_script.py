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
from compute import *
from matplotlib import pyplot as plt
import sys 
from itertools import combinations

import warnings
warnings.simplefilter('ignore', DeprecationWarning)

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

    if regions != None:
        yeo_indices = yeo_dict[regions]

    if method == 'bold':
        list_subjects = []
        for i in glob.glob(PATH+'*'):
            if (i.split('/')[-1].split('-')[0] == 'TC_114_sub') & (i.split('/')[-1].split('-')[1].endswith(f'{movie}.txt')):
                list_subjects.append(i)
        mtx_upper_triangular = []
        for i, PATH_SUBJ in enumerate(list_subjects):
            data_feature = pd.read_csv(PATH_SUBJ, sep=' ', header=None)
            if regions is None:
                connectivity_matrix = np.corrcoef(data_feature, rowvar=False)
            else:
                connectivity_matrix = np.corrcoef(data_feature, rowvar=False)[:,yeo_indices]
            upper_triangular = connectivity_matrix[np.triu_indices_from(connectivity_matrix, k=1)]
            mtx_upper_triangular.append(upper_triangular)
        mtx_upper_triangular = np.array(mtx_upper_triangular)
        X = pd.DataFrame(mtx_upper_triangular)
        print('The shape of X for BOLD is: ', X.shape)
    
    elif method == 'scaffold':
        scaffold_current=np.zeros((30,int(114*113/2)))
        for i in glob.glob(PATH+'*'):
            if (i.split('/')[-1].split('-')[0] == 'Scaffold_frequency_TC_114_sub') & (i.split('/')[-1].split('-')[1].endswith(f'{movie}.hd5')):
                try:
                    file=h5py.File(i,'r',swmr=True)
                except:
                    continue
                N=114
                u,v=np.triu_indices(n=N,k=1)
                subjID = int(i.split('/')[-1].split('-')[1][1:3]) - 1
                if subjID > 10:
                    if subjID > 16:
                        subjID -= 2
                    else:
                        subjID -= 1
                for t in range(1,len(file)+1):
                    scaffold_current[subjID,:]+=file[str(t)][:][u,v]
                scaffold_current[subjID]=scaffold_current[subjID]/len(file)
        X = scaffold_current.copy()
        print('The shape of X for SCAFFOLD is: ', X.shape)

    elif method == 'triangles':

        indices_yeo_all = []
        for idx_triangles,(i,j,k) in enumerate(combinations(np.arange(114),3)):
            flag=[i in yeo_indices, j in yeo_indices, k in yeo_indices]
            if sum(flag) == 3: ## All the nodes belong to the same Yeo networks
                indices_yeo_all.append(idx_triangles)
        indices_yeo_all=np.array(indices_yeo_all)
        number_indices = len(yeo_indices)
        length = int((number_indices * (number_indices-1) * (number_indices-2)) / (3*2))
        print(length)

        current_tri = np.zeros((30, length))
        for string in glob.glob(PATH+'*'):
            if (string.endswith(f'{movie}.hd5')):
                try:
                    file=h5py.File(string,'r',swmr=True)
                except:
                    continue
                
                subjID = int(string.split('/')[-1].split('_')[4][1:3]) - 1
                if subjID > 10:
                    if subjID > 16:
                        subjID -= 2
                    else:
                        subjID -= 1
                for t in range(0,len(file)):
                    sub_matrix = np.array(file[str(t)][:])[indices_yeo_all]
                    current_tri[subjID,:]+=sub_matrix
                current_tri[subjID]=current_tri[subjID]/len(file)
        X = current_tri.copy()
        print('The shape of X for TRIANGLES is: ', X.shape)

    return X

def standa(X,Y):  
    X_normed = X.copy()
    Y_normed = Y.copy()
    
    X_normed=(X_normed-np.nanmean(X_normed,axis=0))/(np.nanstd(X_normed,axis=0, ddof=1))
    Y_normed=(Y_normed-np.nanmean(Y_normed,axis=0))/(np.nanstd(Y_normed,axis=0, ddof=1))

    return X_normed, Y_normed

def run_decomposition(X,Y):                                                   
        res={}
        print("... Normalisation ...")
        X_std, Y_std = standa(X, Y)
        res['X']=X
        res['Y']=Y
        res['X_std']= X_std
        res['Y_std']= Y_std
     
        print("...SVD ...")
        R=R_cov(X_std, Y_std)
        U,S, V = SVD(R, ICA=True)
        ExplainedVarLC =varexp(S)
        Lx, Ly= PLS_scores(X_std, Y_std, U, V)

        res['R']=R
        res['U']=U
        res['S']=S
        res['V']=V
       
        return res
    
def permutation(res_original, nPerms, seed,seuil):
        print("...Permu...")
        res={}
        res['Sp_vect']=permu(res_original['X_std'],res_original['Y_std'], res_original['U'], nPerms, seed)
        res['P_val'], res['sig_LC'] = myPLS_get_LC_pvals(res['Sp_vect'],res_original['S'],nPerms, seuil)
        
        return res 

def myPLS_bootstrapping(X0,Y0,U,V, nBoots, seed=1):
    """ Boostrap on X0 & Y0 and recompute SVD 

    INPUT 
    - X0 (T x V DataFrame) : Voxels-wise serie (not normalized)
    - Y0 (T x M DataFrame) : Behavior/design data (not normalized)
    - U (M x L(#LCs)  DataFrame) : Left singular vector from SVD decomposition
    - V (V x L (#LCs)DataFrame) : Right Singular Vector from SVD decompositon (transposed)
    - nBoots (int) : number of bootstrap sample 
    - seed (int) : seed for random number generator
    
    OUTPUT 
    - boot_results (dic) : containg results from Bootstrapping --> Ub_vect nboots x M x L matrix
                                                               --> Vb_vect nboots x V x L matrix                                                       
    - boot_stat (dic) : containing statistique from Boostrapping --> bsr_u MxL matrix (DataFrame) storing stability score for U
                                                                 --> bsr_v VxL matrix (DataFrame) storing stability score for V
                                                                 --> u_std MxL matrix (Array) storing standard deviation for U
                                                                 --> v_std VxL matrix (Array) storing standard deviation for V 
    """
  
    rs = RandomState(MT19937(SeedSequence(seed)))
    boot_results = {}
    
    Ub_vect = np.zeros((nBoots,) + U.shape)
    Vb_vect= np.zeros((nBoots,) + V.shape)
    
    for i in range(nBoots):
        ## X & Y resampling
        Xb = X0.sample(frac=1, replace=True, random_state=rs)
        Yb = Y0.sample(frac=1, replace=True, random_state=rs)
        
        
        Xb,Yb = standa(Xb,Yb)
        ## Cross-covariance
        Rb = R_cov(Xb, Yb)
        
        ## SVD 
        Ub, Sb, Vb = SVD(Rb, seed=seed)
        
        ## Procrustas transform (correction for axis rotation/reflection)
        rotatemat1 = rotatemat(U, Ub)
        rotatemat2 = rotatemat(V, Vb)
        
        ## Full rotation
        rotatemat_full = (rotatemat1 + rotatemat2) / 2
        Vb = Vb @ rotatemat_full
        Ub = Ub @ rotatemat_full
        
        ## Store Singular vectors
        Ub_vect[i] = Ub
        Vb_vect[i] = Vb
    
    boot_results['u_std'] = np.std(Ub_vect, axis=0)
    boot_results['v_std'] = np.std(Vb_vect, axis=0)
    
    boot_results['bsr_u'] = U / boot_results['u_std']
    boot_results['bsr_v'] = V / boot_results['v_std']

    return boot_results    
    
def bootstrap(res_original,nBoot,seed): 
    print("... Bootstrap...")
    res={}
    res= myPLS_bootstrapping(res_original['X'],res_original['Y'] , res_original['U'],res_original['V'], 
                                      nBoot, seed)

    return res

def exp_var(S, Sp_vect, LC_pvals, name, movie_name, METHOD): 
    """
    Plot the cumulative explained variance and save it
    
    Inputs 
    -------
    S: ndarray
        Array of singular values
        
    Sp_vect: ndarray
        Array of LC vectors
        
    LC_pvals: ndarray
        Array of LC p-values
        
    name: str
        Name of the plot file to save
        
   
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    indices_significant=np.where(LC_pvals<0.05)[0]
    
    # Number of LCs
    nc = np.arange(len(LC_pvals)) + 1
    
    # Create another axes object for secondary y-Axis
    ax2 = ax.twinx()
    
    # Plot singular values
    #     ax.plot(nc, np.diag(S), color='grey', marker='o', fillstyle='none',clip_on=False)
    ax.plot(nc, varexp(S), color='grey', marker='o', fillstyle='none',clip_on=False)
    ax.plot(nc[indices_significant], varexp(S)[indices_significant], color='yellow', marker='o',clip_on=False)
    
    # Plot the cumulative explained covariance
    ax2.plot(nc, (np.cumsum(varexp(S))*100), color='steelblue', ls='--',clip_on=False)
    
    # Labeling & Axes limits
    labels = [f"LC {idx+1} (p={np.round(i, 4)})" for idx, i in enumerate(LC_pvals)]
    #plt.title('Explained Covariance by each LC')
    ax.set_ylabel('Explained Covariance')
    
    #     ticks_loc = ax.get_xticks().tolist()
    #     ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    #     print(len(ticks_loc))
    #     ax.set_xticks(ticks_loc)
    ax.set_xticks(np.arange(1, len(LC_pvals)+0.1))
    ax.set_xticklabels(labels, rotation=45,ha='right')
    ax.set_xlim([1, len(LC_pvals)])
    ax2.set_xticks(nc, labels)
    ax2.set_ylabel('Explained correlation', color='steelblue')
    ax2.set_ylim([0, 100])
    # plt.grid()

    # Defining display layout
    plt.tight_layout()

    plt.savefig(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Images/explained_covariance_movie_{METHOD}_{movie_name}.png', dpi=300)
    print('The plot was saved in: ', f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Images/explained_covariance_movie_{METHOD}_{movie_name}.png')

PATH_YEO = '/media/miplab-nas2/Data2/Movies_Emo/Silvia/HigherOrder/Data/yeo_RS7_Schaefer100S.mat'

def loading_yeo(path=PATH_YEO):
    ##Loading the yeoROIS
    yeoROIs=np.array([i[0]-1 for i in loadmat(path)['yeoROIs']])
    yeoROI_dict={label_Yeo:np.where(yeoROIs==idx_Yeo)[0] for idx_Yeo,label_Yeo in enumerate(['VIS','SM','DA','VA','L','FP','DMN','SC','C'])}
    yeoROI_dict['SC']=np.array(sorted(np.hstack((yeoROI_dict['SC'],yeoROI_dict['C']))))
    del yeoROI_dict['C']
    return(yeoROI_dict)

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

    print('The path is: ', PATH)
    print('The movie is: ', movie_name)
    print('The method is: ', method)
    print('The region is: ', region)

    # Load the areas from yeo
    yeo_dict = loading_yeo(PATH_YEO)

    if region == None:
        name_of_region = 'ALL'
    else:
        name_of_region = region

    # Load the Y behavioural dataset
    Y = pd.read_csv(PATH_DATA, sep='\t', header=0)[columns]
    print('The shape of the Y behavioural dataset is: ', Y.shape)

    print('\n' + ' -' * 10 + f' for {method}, {movie_name} and {name_of_region} FOR: ', movie_name, ' -' * 10)
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
    results['LC']=np.arange(1,13)
    results['Region'] = name_of_region

    # Concatenate the results
    PATH_SAVE = f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/PLS_{method}_{name_of_region}_results.csv'
    # if os.path.exists(PATH_SAVE):
    #     PLS_results = pd.read_csv(PATH_SAVE)
    #     PLS_results = pd.concat([PLS_results, results], axis=0)
    # else:
    PLS_results = pd.DataFrame(results)
    print('The shape of the PLS results is: ', PLS_results.shape, ' and the number of significant LCs is: ', data_cov_significant.shape[0])
    PLS_results.to_csv(PATH_SAVE, index=False)

    print('\n' + f"------------ The PLS for {method}, {movie_name} and {name_of_region} was performed!!! ------------")