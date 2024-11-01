import numpy as np
import h5py 
import pandas as pd
import glob
from compute_pred import *
from matplotlib import pyplot as plt
from itertools import combinations
from scipy.io import loadmat


PATH_YEO = '/media/miplab-nas2/Data2/Movies_Emo/Silvia/HigherOrder/Data/yeo_RS7_Schaefer100S.mat'

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
    N = 114 if regions == 'ALL' else len(yeo_indices)

    if method == 'bold':
        list_subjects = []
        for i in glob.glob(PATH+'*'):
            if (i.split('/')[-1].split('-')[0] == 'TC_114_sub') & (i.split('/')[-1].split('-')[1].endswith(f'{movie}.txt')):
                list_subjects.append(i)
        mtx_upper_triangular = []
        for i, PATH_SUBJ in enumerate(list_subjects):
            data_feature = pd.read_csv(PATH_SUBJ, sep=' ', header=None)
            if regions == 'ALL':
                connectivity_matrix = np.corrcoef(data_feature, rowvar=False)
            else:
                connectivity_matrix = np.corrcoef(data_feature, rowvar=False)[:,yeo_indices]
            upper_triangular = connectivity_matrix[np.triu_indices_from(connectivity_matrix, k=1)]
            mtx_upper_triangular.append(upper_triangular)
        mtx_upper_triangular = np.array(mtx_upper_triangular)
        X = pd.DataFrame(mtx_upper_triangular)
        print('The shape of X for BOLD is: ', X.shape)
    
    elif method == 'scaffold':
        scaffold_current=np.zeros((30,int(N*(N-1)/2)))
        for i in glob.glob(PATH+'*'):
            if (i.split('/')[-1].split('-')[0] == 'Scaffold_frequency_TC_114_sub') & (i.split('/')[-1].split('-')[1].endswith(f'{movie}.hd5')):
                try:
                    file = h5py.File(i, 'r', swmr=True)
                except Exception as e:
                    print(f"An error occurred while opening the file: {str(e)}")
                    continue
                u,v=np.triu_indices(n=N,k=1)
                subjID = int(i.split('/')[-1].split('-')[1][1:3]) - 1
                if subjID > 10:
                    if subjID > 16:
                        subjID -= 2
                    else:
                        subjID -= 1
                for t in range(1,len(file)+1):
                    if regions == 'ALL':
                        scaffold_current[subjID,:]+=file[str(t)][:][u,v]
                    else:
                        scaffold_current[subjID,:]+=file[str(t)][:][yeo_indices,:][:,yeo_indices][u,v]
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
                if regions == 'ALL':
                    for t in range(0,len(file)):
                        sub_matrix = np.array(file[str(t)][:])
                        current_tri[subjID,:]+=sub_matrix
                    current_tri[subjID]=current_tri[subjID]/len(file)
                else:
                    for t in range(0,len(file)):
                        sub_matrix = np.array(file[str(t)][:])[indices_yeo_all]
                        current_tri[subjID,:]+=sub_matrix
                    current_tri[subjID]=current_tri[subjID]/len(file)
        X = current_tri.copy()
        print('The shape of X for TRIANGLES is: ', X.shape)

    elif method == 'edges':
        list_subjects = []
        for i in glob.glob(PATH+'*'):
            if (i.split('/')[-1].split('-')[0] == 'TC_114_sub') & (i.split('/')[-1].split('-')[1].endswith(f'{movie}.txt')):
                list_subjects.append(i)
        mtx_upper_triangular = []
        for i, PATH_SUBJ in enumerate(list_subjects):
            data_feature = pd.read_csv(PATH_SUBJ, sep=' ', header=None)
            data_feature = np.array(data_feature)
            if regions == 'ALL':
                u, v = np.triu_indices(data_feature.shape[1], k=1)                
                edge_file_array = data_feature[:,u] * data_feature[:,v]
                connectivity_matrix = np.corrcoef(edge_file_array, rowvar=False)
            else:
                data_feature_reduced = data_feature[yeo_indices,:]
                u, v = np.triu_indices(n=data_feature_reduced.shape[1], k=1)
                edge_file_array = data_feature_reduced[:,u] * data_feature_reduced[:,v]
                connectivity_matrix = np.corrcoef(edge_file_array, rowvar=False)
            upper_triangular =  connectivity_matrix[np.triu_indices_from(connectivity_matrix, k=1)]
            mtx_upper_triangular.append(upper_triangular)
        mtx_upper_triangular = np.array(mtx_upper_triangular)
        X = pd.DataFrame(mtx_upper_triangular)
        print(f'The shape of X for EDGES for {regions} is: ', X.shape)

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
    res= myPLS_bootstrapping(res_original['X'],res_original['Y'], res_original['U'],res_original['V'], nBoot, seed)
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

def loading_yeo(path=PATH_YEO):
    ##Loading the yeoROIS
    yeoROIs=np.array([i[0]-1 for i in loadmat(path)['yeoROIs']])
    yeoROI_dict={label_Yeo:np.where(yeoROIs==idx_Yeo)[0] for idx_Yeo,label_Yeo in enumerate(['VIS','SM','DA','VA','L','FP','DMN'])}
    yeoROI_dict['SC'] = np.arange(100, 114)
    return(yeoROI_dict)