import numpy as np
import h5py 
import pandas as pd
from behavPLS import BehavPLS
from plot import *
from nilearn.masking import compute_brain_mask
from configparser import ConfigParser
from argparse import ArgumentParser
from scipy.io import loadmat
import seaborn as sns
import pickle
import glob
from compute import *
from matplotlib import pyplot as plt

def compute_X(path_subjects, movie):
    """
    Compute the X dataset for the PLS analysis

    -------------------------------------
    Input:
    - param path_subjects: path of the subjects' data
    - param movie: movie to consider
    -------------------------------------
    Output:
    - X: X dataset
    """
    list_subjects = []
    for i in glob.glob(path_subjects+'*'):
        if (i.split('/')[-1].split('-')[0] == 'TC_114_sub') & (i.split('/')[-1].split('-')[1].endswith(f'{movie}.txt')):
            list_subjects.append(i)

    mtx_upper_triangular = []
    for i, PATH_SUBJ in enumerate(list_subjects):
        data_feature = pd.read_csv(PATH_SUBJ, sep=' ', header=None)

        # Obtain the connectivity matrix
        connectivity_matrix = np.corrcoef(data_feature, rowvar=False)

        # Obtain the upper triangular part of the matrix
        upper_triangular = connectivity_matrix[np.triu_indices_from(connectivity_matrix, k=1)]

        # Append the upper triangular part of the matrix to the list
        mtx_upper_triangular.append(upper_triangular)

    # Convert the list into a numpy array
    mtx_upper_triangular = np.array(mtx_upper_triangular)

    X = pd.DataFrame(mtx_upper_triangular)
    print('The matrix X for the movie ', movie, 'was computed and has shape: ', X.shape)

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

def exp_var(S, Sp_vect, LC_pvals, name, movie_name): 
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

    plt.savefig(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Images/explained_covariance_movie_{movie_name}.png', dpi=300)
    print('The plot was saved in: ', f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Images/explained_covariance_movie_{movie_name}.png')

path_subjects = '/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/Data/'
list_movies = ['AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'FirstBite', 
                'LessonLearned', 'Payload', 'Rest', 'Sintel', 'Spaceman', 'Superhero', 
                'TearsOfSteel', 'TheSecretNumber', 'ToClaireFromSonny', 'YouAgain']
PATH_DATA = '/media/miplab-nas2/Data2/Movies_Emo/BIDS/TheRealfMRIStudy/rawdata/participants.tsv'
columns = ['DASS_dep', 'DASS_anx', 'DASS_str',	'bas_d', 'bas_f', 'bas_r', 'bis', 'BIG5_ext', 'BIG5_agr', 'BIG5_con', 'BIG5_neu', 'BIG5_ope']

nb = 30              # Number of participants
nPer = 1000         # Number of permutations for significance testing
nBoot = 1000        # Number of bootstrap iterations
seed = 10           # Seed for reproducibility
sl = 0.05          # Signficant level for statistical testing
p_star = 0.05
if __name__ == '__main__':  

    # Load the Y behavioural dataset
    Y = pd.read_csv(PATH_DATA, sep='\t', header=0)[columns]
    print('The shape of the Y behavioural dataset is: ', Y.shape)

    # Load the X brain dataset
    X = np.stack([compute_X(path_subjects, movie) for movie in list_movies], axis=2)
    print('The shape of the X dataset is: ', X.shape)

    PLS_results = pd.DataFrame(columns=['Covariance Explained', 'P-value', 'Movie', 'LC'])
    for movie_number, movie_name in enumerate(list_movies):
        print('\n' + ' -' * 10 + ' Movie name: ', movie_name, ' Movie number: ', movie_number, ' -' * 10)
        
        # Select the movie
        X_movie = pd.DataFrame(X[:, :, movie_number])

        # Perform the PLSC Behavioural analysis
        res = run_decomposition(X_movie, Y)
        res_permu = permutation(res, nPer, seed, sl)
        res_bootstrap = bootstrap(res, nBoot, seed)
        print('The pvalues are: ', res_permu['P_val'])

        # Save the results
        results=pd.DataFrame(list(zip(varexp(res['S']),res_permu['P_val'])),
                                columns=['Covariance Explained', 'P-value'])
        data_cov_significant=results[results['P-value'] < p_star]
        data_cov_significant.sort_values('P-value')
        results['Movie']=movie_name
        results['LC']=np.arange(1,13)

        # Concatenate the results
        PLS_results = pd.concat([PLS_results, results], axis=0)
        print('The shape of the PLS results is: ', PLS_results.shape, ' and the number of significant LCs is: ', data_cov_significant.shape[0])

        # Print the results
        exp_var(res['S'], res_permu['Sp_vect'], res_permu['P_val'], "Discrete_Var", movie_number)



