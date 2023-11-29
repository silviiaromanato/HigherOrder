import numpy as np
import h5py 
import pandas as pd
import glob
from compute import *
from matplotlib import pyplot as plt
from itertools import combinations
from scipy.io import loadmat


PATH_YEO = '/media/miplab-nas2/Data2/Movies_Emo/Silvia/HigherOrder/Data/yeo_RS7_Schaefer100S.mat'

def compute_X(PATH, movie, method, regions=None):
    """
    Compute the X dataset for the PLS analysis.

    Parameters:
    - PATH (str): Path of the subjects' data.
    - movie (str): Movie to consider.
    - method (str): Method to use ('bold', 'scaffold', 'triangles', 'edges').
    - regions (list/str, optional): Specific regions to consider. Default is None.

    Returns:
    - pd.DataFrame/np.ndarray: X dataset.
    """
    yeo_dict = loading_yeo(PATH_YEO)
    yeo_indices = yeo_dict[regions] if regions != 'ALL' else None
    N = 114 if regions == 'ALL' else len(yeo_indices)

    if method == 'bold':
        return process_bold_method(PATH, movie, regions, yeo_indices, N)
    elif method == 'scaffold':
        return process_scaffold_method(PATH, movie, regions, yeo_indices, N)
    elif method == 'triangles':
        return process_triangles_method(PATH, movie, regions, yeo_indices, N)
    elif method == 'edges':
        return process_edges_method(PATH, movie, regions, yeo_indices, N)
    else:
        raise ValueError("Invalid method specified.")

def process_scaffold_method(PATH, movie, regions, yeo_indices, N):
    scaffold_current=np.zeros((30,int(N*(N-1)/2)))
    for i in glob.glob(PATH+'*'):
        if (i.split('/')[-1].split('-')[0] == 'Scaffold_frequency_TC_114_sub') & (i.split('/')[-1].split('-')[1].endswith(f'{movie}.hd5')):
            file = h5py.File(i, 'r', swmr=True)
            u,v=np.triu_indices(n=N,k=1)
            subjID = subjid_computat(i)
            for t in range(1,len(file)+1):
                if regions == 'ALL':
                    scaffold_current[subjID,:]+=file[str(t)][:][u,v]
                else:
                    scaffold_current[subjID,:]+=file[str(t)][:][yeo_indices,:][:,yeo_indices][u,v]
            scaffold_current[subjID]=scaffold_current[subjID]/len(file)
    X = scaffold_current.copy()
    print('The shape of X for SCAFFOLD is: ', X.shape)
    return X

def subjid_computat(i):
    subjID = int(i.split('/')[-1].split('-')[1][1:3]) - 1
    if subjID > 10:
        if subjID > 16:
            subjID -= 2
        else:
            subjID -= 1
    return subjID

def process_triangles_method(PATH, movie, regions, yeo_indices, N):
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
            # try:
            file=h5py.File(string,'r',swmr=True)
            # except:
            #     continue
            subjID = subjid_computat(string)
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

def process_edges_method(PATH, movie, regions, yeo_indices, N):
    list_subjects = list_of_subjects(PATH, movie)
    mtx_upper_triangular = []
    for PATH_SUBJ in list_subjects:
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

def list_of_subjects(PATH, movie):
    list_subjects = []
    for i in glob.glob(PATH+'*'):
        if (i.split('/')[-1].split('-')[0] == 'TC_114_sub') & (i.split('/')[-1].split('-')[1].endswith(f'{movie}.txt')):
            list_subjects.append(i)
    return list_subjects

def process_bold_method(PATH, movie, regions, yeo_indices, N):
    list_subjects = list_of_subjects(PATH, movie)
    mtx_upper_triangular = []
    for i, PATH_SUBJ in enumerate(list_subjects):
        data_feature = pd.read_csv(PATH_SUBJ, sep=' ', header=None) # This is N (114) x T (timepoints)
        if regions == 'ALL':
            connectivity_matrix = np.corrcoef(data_feature, rowvar=False) # This is N (114) x N (114)
        else:
            connectivity_matrix = np.corrcoef(data_feature, rowvar=False)[:,yeo_indices]
        upper_triangular = connectivity_matrix[np.triu_indices_from(connectivity_matrix, k=1)] 
        mtx_upper_triangular.append(upper_triangular)
    mtx_upper_triangular = np.array(mtx_upper_triangular)
    X = pd.DataFrame(mtx_upper_triangular)
    print('The shape of X for BOLD is: ', X.shape)
    return X

# def compute_X(PATH, movie, method, regions = None):
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
            data_feature = pd.read_csv(PATH_SUBJ, sep=' ', header=None) # This is N (114) x T (timepoints)
            if regions == 'ALL':
                connectivity_matrix = np.corrcoef(data_feature, rowvar=False) # This is N (114) x N (114)
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
         # print("... Normalisation ...")
        X_std, Y_std = standa(X, Y)
        res['X']=X
        res['Y']=Y
        res['X_std']= X_std
        res['Y_std']= Y_std
     
        # print("...SVD ...")
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
        # print("...Permu...")
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
     # print("... Bootstrap...")
    res={}
    res= myPLS_bootstrapping(res_original['X'],res_original['Y'], res_original['U'],res_original['V'], nBoot, seed)
    return res

def loading_yeo(path=PATH_YEO):
    ## Loading the yeoROIS
    yeoROIs=np.array([i[0]-1 for i in loadmat(path)['yeoROIs']])
    yeoROI_dict={label_Yeo:np.where(yeoROIs==idx_Yeo)[0] for idx_Yeo,label_Yeo in enumerate(['VIS','SM','DA','VA','L','FP','DMN'])}
    yeoROI_dict['SC'] = np.arange(100, 114)
    return(yeoROI_dict)

def compute_X_concat(PATH, emotions, threshold, regions = 'ALL', control = False, seed = 1, mean = False):

    list_movies = ['AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'FirstBite', 'LessonLearned', 'Payload', 'Sintel', 'Spaceman', 'Superhero', 'TearsOfSteel', 'TheSecretNumber', 'ToClaireFromSonny', 'YouAgain']

    # Load the functional regions that we want to include.
    yeo_dict = loading_yeo(PATH_YEO)
    yeo_indices = yeo_dict[regions] if regions != 'ALL' else None
    N = 114 if regions == 'ALL' else len(yeo_indices)

    # Create a dictionary where the keys are the movies and the values are the paths to the txt files
    dict_movies = {}
    for movie in list_movies:
        list_X = []
        for i in glob.glob(PATH+'*'):
            if (i.split('/')[-1].split('-')[0] == 'TC_114_sub') & (i.split('/')[-1].split('-')[1].endswith(f'{movie}.txt')):
                list_X.append(i)
        dict_movies[movie] = list_X

    # Create a df subject x movies where each cell is a path to the txt file
    data_subjects = pd.DataFrame.from_dict(dict_movies) # shape (30, 15)
    data_subjects.reset_index(drop=True, inplace=True)

    # Create a list of dataframes where each dataframe is the data of the subjects for each movie
    list_datafeatures = []
    for subject in range(data_subjects.shape[0]):
        list_df = []

        for movie in range(data_subjects.shape[1]):
            # Read the labels and the data from the emotions
            labels = pd.read_json(f'/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/Annot13_{list_movies[movie]}_stim.json')
            data = pd.read_csv(f'/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/Annot13_{list_movies[movie]}_stim.tsv', sep = '\t', header = None)
            data.columns = labels['Columns']

            # Take the time peaks
            if mean == True: # take the mean of the emotions if mean == True
                data = data[emotions].mean(axis=1)
            times_peaking = data.loc[data > threshold].index
            if control == True:
                np.random.seed(seed)
                times_peaking = np.random.choice(data.index, size=len(times_peaking), replace=False)

            # Read the data from the txt file and select the times where the emotion is peaking
            data_features = pd.read_csv(data_subjects.iloc[subject, movie], sep=' ', header=None)
            data_features = data_features.iloc[times_peaking,:]
            list_df.append(data_features)

        # Concatenate the dataframes of the subject
        data_features_concat = pd.concat(list_df, axis=0)
        list_datafeatures.append(data_features_concat)

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

def run_pls(X_movie, Y, region):
    nPer = 1000         # Number of permutations for significance testing
    seed = 10           # Seed for reproducibility
    sl = 0.05          # Signficant level for statistical testing
    res = run_decomposition(X_movie, Y)
    res_permu = permutation(res, nPer, seed, sl)
    print('The pvalues are: ', res_permu['P_val'])
    results=pd.DataFrame(list(zip(varexp(res['S']),res_permu['P_val'])), columns=['Covariance Explained', 'P-value'])
    results['Movie']='concatenated'
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
    return results

def concat_emo():
    list_movies = ['AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'FirstBite', 'LessonLearned', 'Payload', 'Sintel', 'Spaceman', 'Superhero', 'TearsOfSteel', 'TheSecretNumber', 'ToClaireFromSonny', 'YouAgain']
    data_concat = pd.DataFrame()
    for movie_name in list_movies:
        labels = pd.read_json(f'/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/Annot13_{movie_name}_stim.json')
        data = pd.read_csv(f'/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/Annot13_{movie_name}_stim.tsv', sep = '\t', header = None)
        data.columns = labels['Columns']
        data_concat = pd.concat([data_concat, data], axis=0)
    print('The shape of the data_concat is: ', data_concat.shape)
    return data_concat
