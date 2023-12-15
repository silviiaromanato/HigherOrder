import numpy as np
import h5py 
import pandas as pd
import glob
from compute import *
from matplotlib import pyplot as plt
from itertools import combinations
from scipy.io import loadmat
import re
import sys
import seaborn as sns

def compute_X(PATH, movie, method, PATH_YEO, regions=None):
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
    times = None

    if method == 'bold':
        return process_bold_method(PATH, movie, regions, yeo_indices, N)
    elif method == 'scaffold':
        return process_scaffold_method(PATH, movie, regions, yeo_indices, times, N)
    elif method == 'triangles':
        return process_triangles_method(PATH, movie, regions, yeo_indices, times, N)
    elif method == 'edges':
        return process_edges_method(PATH, movie, regions, yeo_indices, N)
    else:
        raise ValueError("Invalid method specified.")

def subjid_computat(i):
    subjID = int(i.split('/')[-1].split('-')[1][1:3]) - 1
    if subjID > 10:
        if subjID > 16:
            subjID -= 2
        else:
            subjID -= 1
    return subjID

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
    
    print('There are nan values in the X: ', X_normed, X_normed.shape)
    X_normed=(X_normed-np.nanmean(X_normed,axis=0))/(np.nanstd(X_normed,axis=0, ddof=1))
    Y_normed=(Y_normed-np.nanmean(Y_normed,axis=0))/(np.nanstd(Y_normed,axis=0, ddof=1))
    print('There are nan values in the X: ', np.isnan(X_normed).any().sum())
    return X_normed, Y_normed

def run_decomposition(X,Y):                                                   
        res={}
        print("... Normalisation ...")
        X_std, Y_std = standa(X, Y)
        res['X']=X
        res['Y']=Y
        res['X_std']= X_std
        res['Y_std']= Y_std
     
        # print("...SVD ...")
        R=R_cov(X_std, Y_std)
        R = np.nan_to_num(R)
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

def R_cov(X, Y) : 
    """
    Computes the Correlation Matrix
    
    Input 
    -------
        - X (T x V Dataframe) : Voxel-wise serie
        - Y (T x M DataFrame) : Behavior dataset 
    Ouput
    -------
        - R (M x V Array) : Correlation Matrix
    """
    if(X.shape[0] != Y.shape[0]): raise Exception("Input arguments X and Y should have the same number of rows")
        
    R = (np.array(Y.T) @ np.array(X))
    return R

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

def loading_yeo(path):
    ## Loading the yeoROIS
    yeoROIs=np.array([i[0]-1 for i in loadmat(path)['yeoROIs']])
    yeoROI_dict={label_Yeo:np.where(yeoROIs==idx_Yeo)[0] for idx_Yeo,label_Yeo in enumerate(['VIS','SM','DA','VA','L','FP','DMN'])}
    yeoROI_dict['SC'] = np.arange(100, 114)
    return(yeoROI_dict)

def compute_X_concat(PATH, emotions, threshold, PATH_YEO, regions = 'ALL', control = False, seed = 1, todo = 'emotions', mean = False, server = 'miplab'):

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

        count_times = 0
        for movie in range(data_subjects.shape[1]):
            if todo == 'emotions':
                
                if server == 'miplab':
                    labels = pd.read_json(f'/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/Annot13_{list_movies[movie]}_stim.json')
                    data = pd.read_csv(f'/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/Annot13_{list_movies[movie]}_stim.tsv', sep = '\t', header = None)
                elif server == 'enrico':
                    labels = pd.read_json(f'/home/silvia/Flavia_E3/EmoData/Annot13_{list_movies[movie]}_stim.json')
                    data = pd.read_csv(f'/home/silvia/Flavia_E3/EmoData/Annot13_{list_movies[movie]}_stim.tsv', sep = '\t', header = None)
                data.columns = labels['Columns']

                # Take the time peaks
                if mean == True: # take the mean of the emotions if mean == True
                    data = data[emotions].mean(axis=1)
                else:
                    data = data[emotions]

            elif todo == 'features_extracted':
                features = extract_features(list_movies[movie], columns = ['spectralflux', 'rms', 'zcrs'], 
                                            columns_images = ['average_brightness_left', 'average_saturation_left', 'average_hue_left', 
                                                              'average_brightness_right', 'average_saturation_right', 'average_hue_right'],  cluster = True)
                data = features[emotions]

            times_peaking = data.loc[data > threshold].index
            count_times += len(times_peaking)
            if times_peaking.shape[0] == 0:
                continue

            if control == True:
                np.random.seed(seed)
                times_peaking = np.random.choice(data.index, size=len(times_peaking), replace=False)
            times_peaking = np.sort(times_peaking)

            # Read the data from the txt file and select the times where the emotion is peaking
            data_features = pd.read_csv(data_subjects.iloc[subject, movie], sep=' ', header=None)
            data_features = data_features.iloc[times_peaking,:]
            list_df.append(data_features)

        # Concatenate the dataframes of the subject
        if  len(list_df) == 0:
            continue
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

def run_pls(X_movie, Y, region, movie_name = 'concatenated'):
    nPer = 1000         # Number of permutations for significance testing
    seed = 10           # Seed for reproducibility
    sl = 0.05           # Signficant level for statistical testing
    res = run_decomposition(X_movie, Y)
    # print if there are nan values 
    res_permu = permutation(res, nPer, seed, sl)
    results=pd.DataFrame(list(zip(varexp(res['S']),res_permu['P_val'])), columns=['Covariance Explained', 'P-value'])
    results['Movie']=movie_name
    results['LC']=np.arange(1, results.shape[0]+1)
    results['Region'] = region
    results['Covariance Explained'] = results['Covariance Explained'].astype(float)
    return results

def boostrap_subjects(X_movie, Y, region, movie_name, sample_size = 20, num_rounds = 100):
    print(f'\nPerforming BOOSTRAPPING on {sample_size} subjects for {num_rounds} rounds')
    results = pd.DataFrame(columns = ['Covariance Explained', 'P-value', 'Movie', 'LC', 'Region', 'bootstrap_round'])
    for i in range(num_rounds):
        print('The round is: ', i)
        idx = np.random.choice(np.arange(X_movie.shape[0]), size=sample_size, replace=True)
        X_movie_sample = X_movie.iloc[idx,:]
        Y_sample = Y.iloc[idx,:]
        pls = run_pls(X_movie_sample, Y_sample, region, movie_name)
        pls = pd.DataFrame(pls)
        pls['bootstrap_round'] = i
        results = pd.concat([results, pls], axis=0)
    return results

def concat_emo(server = 'miplab'):
    list_movies = ['AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'FirstBite', 'LessonLearned', 'Payload', 'Sintel', 'Spaceman', 'Superhero', 'TearsOfSteel', 'TheSecretNumber', 'ToClaireFromSonny', 'YouAgain']
    data_concat = pd.DataFrame()
    for movie_name in list_movies:
        if server == 'enrico':
            PATH = '/home/silvia/Flavia_E3/EmoData'
        elif server == 'miplab':
            PATH = '/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData'
        labels = pd.read_json(PATH + f'/Annot13_{movie_name}_stim.json')
        data = pd.read_csv(PATH + f'/Annot13_{movie_name}_stim.tsv', sep = '\t', header = None)
        data.columns = labels['Columns']
        data_concat = pd.concat([data_concat, data], axis=0)
    print('The shape of the data_concat is: ', data_concat.shape)
    return data_concat

def counting_points(movies, thresholds, emotions):
    """
    Counts the number of points for each emotion in each movie.

    Args:
    movies (list): List of movie names.
    thresholds (list): List of threshold values for emotions.
    emotions (list): List of emotions to analyze.

    Returns:
    dict: A dictionary with the count of points for each emotion above the threshold.
    """
    emotion_data = concatenate_movie_emotions(movies)
    return count_features_above_threshold(emotion_data, thresholds, emotions)

def concatenate_movie_emotions(movies):
    """
    Concatenates emotion data from multiple movies.

    Args:
    movies (list): List of movie names.

    Returns:
    DataFrame: A concatenated DataFrame of emotion data for all movies.
    """
    concatenated_emotions = pd.DataFrame()
    for movie in movies:
        concatenated_emotions = pd.concat([concatenated_emotions, read_movie_data(movie)], axis=0)
    return concatenated_emotions

def read_movie_data(movie):
    """
    Reads emotion data for a single movie.

    Args:
    movie (str): The name of the movie.

    Returns:
    DataFrame: DataFrame of emotion data for the movie.
    """
    labels_path = f'/Users/silviaromanato/Desktop/SEMESTER_PROJECT/Material/Data/EmoData/Annot13_{movie}_stim.json'
    data_path = f'/Users/silviaromanato/Desktop/SEMESTER_PROJECT/Material/Data/EmoData/Annot13_{movie}_stim.tsv'

    labels = pd.read_json(labels_path)[['Columns']]
    data = pd.read_csv(data_path, sep='\t', header=None)
    data.columns = labels['Columns']

    return data

def count_features_above_threshold(data, thresholds, features):
    """
    Counts the number of points for each feature above given thresholds.

    Args:
    data (DataFrame): DataFrame containing feature data.
    thresholds (list): List of threshold values for features.
    features (list): List of features to analyze.

    Returns:
    dict: A dictionary with the count of points for each feature above the threshold.
    """
    count_points = {}
    for threshold in thresholds:
        count_points[threshold] = {feature: count_points_above_threshold(data, feature, threshold) for feature in features}
        count_points[threshold]['All Movie'] = len(data)
    return count_points

def count_points_above_threshold(data, feature, threshold):
    """
    Counts the number of points for a given feature above a threshold.

    Args:
    data (DataFrame): DataFrame containing feature data.
    feature (str): The feature to analyze.
    threshold (float): The threshold value.

    Returns:
    int: Number of points above the threshold for the given feature.
    """
    return len(data[data[feature] > threshold])

def extract_annot(path_folder,film_ID):
    f = open(str(path_folder)+'Annot13_'+str(film_ID)+'_stim.json', "r")
    data_annot = json.loads(f.read())
    annot = pd.read_csv(str(path_folder)+'Annot13_'+str(film_ID)+'_stim.tsv', sep='\t', names=data_annot['Columns'])
    return annot

def extract_corrmat_allregressors(emo_path_folder,film_ID):
    annot=extract_annot(emo_path_folder, film_ID)
    corrmat_allregressors=np.corrcoef(annot.values)   
    return corrmat_allregressors

def extract_corrmat_emo(emo_path_folder,film_ID):
    positive_emotions = ['Love', 'Regard', 'WarmHeartedness', 'Pride','Satisfaction','Happiness']
    negative_emotions = ['Sad', 'Anxiety', 'Fear', 'Guilt','Disgust','Anger']
    all_emotions = positive_emotions + negative_emotions
    #Extract the required data data from the emodata file
    annot=extract_annot(emo_path_folder,film_ID)
    positive_emotions_matrix = annot[positive_emotions].values
    negative_emotions_matrix = annot[negative_emotions].values
    all_emotions_matrix = annot[all_emotions].values
    #Compute correlation
    corrmat_positive=np.corrcoef(positive_emotions_matrix)   
    corrmat_negative=np.corrcoef(negative_emotions_matrix)   
    corrmat_all_emo=np.corrcoef(all_emotions_matrix)   
    return corrmat_positive, corrmat_negative, corrmat_all_emo

def extract_features(movie_name, columns = ['mean_chroma', 'mean_mfcc', 'spectralflux', 'rms', 'zcrs'], 
                             columns_images = ['average_brightness_left', 'average_saturation_left', 'average_hue_left', 
                                               'average_brightness_right', 'average_saturation_right', 'average_hue_right'],
                             server = 'miplab'
                             ):
    if server == 'enrico':
        PATH_EMO = '/home/silvia/Flavia_E3/EmoData/'
    elif server == 'miplab':
        PATH_EMO = '/media/miplab-nas2/Data2/Movies_Emo/Flavia_E3/EmoData/'
    length = extract_corrmat_allregressors(PATH_EMO, movie_name).shape[0]

    movie_name_with_ = re.sub(r"(\w)([A-Z])", r"\1_\2", movie_name)
    if server == 'enrico':
        df_sound = pd.read_csv(f'/home/silvia/Silvia/HigherOrder/Data/Output/features_extracted/features_extracted/features_sound_{movie_name_with_}.csv')[columns]
        df_images = pd.read_csv(f'/home/silvia/Silvia/HigherOrder/Data/Output/features_extracted/features_extracted/movie_features_{movie_name_with_}_exp.csv')[columns_images]
    elif server == 'miplab': 
        df_sound = pd.read_csv(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/features_extracted/features_extracted/features_sound_{movie_name_with_}.csv')[columns]
        df_images = pd.read_csv(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/features_extracted/features_extracted/movie_features_{movie_name_with_}_exp.csv')[columns_images]
    else:
        df_sound = pd.read_csv(f'/Users/silviaromanato/Desktop/SEMESTER_PROJECT/Material/Data/features_extracted/features_sound_{movie_name_with_}.csv')[columns]
        df_images = pd.read_csv(f'/Users/silviaromanato/Desktop/SEMESTER_PROJECT/Material/Data/features_extracted/movie_features_{movie_name_with_}_exp.csv')[columns_images]
    window_size1 = df_images.shape[0] // length
    images_mean = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size1), mode='valid') / window_size1, axis=0, arr=df_images)

    window_size2 = df_sound.shape[0] // length
    sound_mean = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size2), mode='valid') / window_size2, axis=0, arr=df_sound)

    # Select elements to represent the initial arrays
    selected_indices1 = np.linspace(0, len(images_mean) - 1, length, dtype=int)
    selected_indices2 = np.linspace(0, len(sound_mean) - 1, length, dtype=int)
    df_images = pd.DataFrame(images_mean[selected_indices1])
    df_sound = pd.DataFrame(sound_mean[selected_indices2])

    # set the column names
    df_images.columns = columns_images
    df_sound.columns = columns

    # concat the two dataframes 
    df_images.reset_index(drop=True, inplace=True)
    df_sound.reset_index(drop=True, inplace=True)
    features = pd.concat([df_images, df_sound], axis = 1)

    # compute the mean of the left and right features
    features['average_brightness'] = (features['average_brightness_left'] + features['average_brightness_right']) / 2
    features['average_saturation'] = (features['average_saturation_left'] + features['average_saturation_right']) / 2
    features['average_hue'] = (features['average_hue_left'] + features['average_hue_right']) / 2

    # drop the left and right features
    features.drop(columns = ['average_brightness_left', 'average_saturation_left', 'average_hue_left', 'average_brightness_right', 'average_saturation_right', 'average_hue_right'], inplace = True)
    
    return features

def extract_features_concat(cluster = True):
    list_movies = ['AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'FirstBite', 'LessonLearned', 'Payload', 'Sintel', 'Spaceman', 'Superhero', 'TearsOfSteel', 'TheSecretNumber', 'ToClaireFromSonny', 'YouAgain'] 
    concatenated_features = pd.DataFrame()
    for movie in list_movies:
        features = extract_features(movie, columns = ['spectralflux', 'rms', 'zcrs'], 
                                            columns_images = ['average_brightness_left', 'average_saturation_left', 'average_hue_left', 'average_brightness_right', 'average_saturation_right', 'average_hue_right'],
                                                cluster = cluster
                                                )
        
        concatenated_features = pd.concat([concatenated_features, features], axis = 0)
        concatenated_features.reset_index(drop=True, inplace=True)

    concatenated_features = (concatenated_features - concatenated_features.mean()) / concatenated_features.std()

    return concatenated_features

def retrieve_significant_data(peaks_data, data_all_movie, count_pts, thresholds, emotions):

    # Take the significant LC for the PLS on the peaks
    significant_peaks = peaks_data[peaks_data['P-value'] < 0.05]
    significant_peaks = significant_peaks.groupby(['Region', 'bootstrap_round', 'Feature', 'threshold']).sum()['Covariance Explained'].reset_index()
    significant_peaks['Control'] = significant_peaks['Feature'].apply(lambda x: 1 if x.split('_')[0] == 'Control' else 0)
    significant_peaks['Feature'] = significant_peaks['Feature'].apply(lambda x: x.split('_')[-1])
    significant_peaks['Feature'] = significant_peaks['Feature'].apply(lambda x: 'average_' + x if x in ['brightness', 'saturation', 'hue'] else x)

    # Take the significant LC for the PLS on all the movie
    data_all_movie['Feature'] = 'All movie'
    significant_all_movie = data_all_movie[data_all_movie['P-value'] < 0.05]
    significant_all_movie = significant_all_movie.groupby(['Region', 'bootstrap_round', 'Feature']).sum()['Covariance Explained'].reset_index()
    significant_all_movie['Control'] = 0

    # Merge the dataframes
    significant = pd.concat([significant_peaks, significant_all_movie], ignore_index=True)

    for emotion in emotions:
        for thr in thresholds:
            if emotion.startswith('average_'):
                print(emotion, count_pts[thr][emotion])
                significant.loc[(significant['Feature'] == emotion) & (significant['threshold'] == thr), 'Number of points'] = count_pts[thr][emotion]
            else:
                significant.loc[(significant['Feature'] == emotion.split('_')[-1]) & (significant['threshold'] == thr), 'Number of points'] = count_pts[thr][emotion]
    significant.loc[significant['Feature'] == 'All movie', 'Number of points'] = count_pts[thr]['All Movie']
    return significant

def plot_peaks(significant, emotions, thresholds):
    # Define your thresholds
    palette = sns.color_palette("Set2", 8)

    # Assuming 'significant' is your DataFrame and 'emotions' is defined
    order_emotions = ['All movie'] + list(emotions)  # Adjusted for simplicity

    # Set up the subplot grid
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 15))  # Adjust the size as needed

    for i, thr in enumerate(thresholds):
        # Filter data for the current threshold
        significant.loc[significant['Feature'] == 'All movie', 'threshold'] = thr
        df_thr = significant[significant['threshold'] == thr]

        # Plot the boxplot in the i-th subplot
        sns.boxplot(x='Feature', y='Covariance Explained', data=df_thr,  hue='Control',
                    palette=palette, order=order_emotions, ax=axes[i])

        # Adjust x-ticks
        list_n = []
        for emo in order_emotions:
            if emo.startswith('average_'):
                n = df_thr[df_thr['Feature'] == emo]['Number of points'].unique()
            else:
                n = df_thr[df_thr['Feature'] == emo.split('_')[0]]['Number of points'].unique()
            if len(n) == 0:
                n = 0
            else:
                n = n[0]
            list_n.append(f'{emo}: {int(n)}')

        axes[i].set_xticklabels(list_n, rotation=90, fontsize=15)
        axes[i].set_title(f'Concatenated movies - threshold {thr}', fontsize=20)
        axes[i].set_ylim(0, 1)

        for item in ([axes[i].title, axes[i].xaxis.label, axes[i].yaxis.label] +
                     axes[i].get_xticklabels() + axes[i].get_yticklabels()):
            item.set_fontsize(25)

    plt.tight_layout()
    plt.show()

def increase_thr(significant, emotions):

    # Define your thresholds
    palette = sns.color_palette("Set2", 8)[3:8]

    # Assuming 'significant' is your DataFrame and 'emotions' is defined
    order_emotions = emotions  # Adjusted for simplicity
    df = significant[significant['Control'] == 0]

    plt.figure(figsize=(15, 10))
    sns.boxplot(x='Feature', y='Covariance Explained', data=df,  hue='threshold', palette=palette, order=order_emotions)

    plt.xlabel('Emotions', fontsize=20)
    plt.title(f'Covariance explained with less points (threshold increase)', fontsize=25)
    # increase the font size of the x and y ticks
    plt.xticks(fontsize=17)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

def process_bold_method_withtimes(PATH, movie, times, regions, yeo_indices, N):
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

def compute_X_withtimes(PATH, movie, times, method, PATH_YEO, regions = None):

    yeo_dict = loading_yeo(PATH_YEO)
    yeo_indices = yeo_dict[regions] if regions != 'ALL' else None
    N = 114 if regions == 'ALL' else len(yeo_indices)

    if method == 'bold':
        X = process_bold_method_withtimes(PATH, movie, times, regions, yeo_indices, N)
    if method == 'scaffold':
        X = process_scaffold_method(PATH, movie, regions, yeo_indices, times, N)# X is all 0

    if method == 'triangles':
        X = process_triangles_method(PATH, movie, regions, yeo_indices, times, N)
    
    return X

def process_triangles_method(PATH, movie, regions, yeo_indices, times, N):
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
            if times is None:
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
            else:
                if regions == 'ALL':
                    for t in times:
                        sub_matrix = np.array(file[str(t)][:])
                        current_tri[subjID,:]+=sub_matrix
                    current_tri[subjID]=current_tri[subjID]/len(times)
                else:
                    for t in times:
                        sub_matrix = np.array(file[str(t)][:])[indices_yeo_all]
                        current_tri[subjID,:]+=sub_matrix
                    current_tri[subjID]=current_tri[subjID]/len(times)
    X = current_tri.copy()
    print('The shape of X for TRIANGLES is: ', X.shape)
    return X

def process_scaffold_method(PATH, movie, regions, yeo_indices, times, N):
    print(PATH, movie, regions, yeo_indices, times)
    scaffold_current=np.zeros((30,int(N*(N-1)/2)))
    for i in glob.glob(PATH+'*'):
        print(i)
        print((i.split('/')[-1].split('-')[0] == 'Scaffold_frequency_TC_114_sub') & (i.split('/')[-1].split('-')[1].endswith(f'{movie}.hd5')))
        if (i.split('/')[-1].split('-')[0] == 'Scaffold_frequency_TC_114_sub') & (i.split('/')[-1].split('-')[1].endswith(f'{movie}.hd5')):
            print(i)
            file = h5py.File(i, 'r', swmr=True)
            u,v=np.triu_indices(n=N,k=1)
            subjID = subjid_computat(i)
            if times is None:
                for t in range(1,len(file)+1):
                    if regions == 'ALL':
                        scaffold_current[subjID,:]+=file[str(t)][:][u,v]
                        print('The shape of scaffold_current is: ', scaffold_current)
                    else:
                        scaffold_current[subjID,:]+=file[str(t)][:][yeo_indices,:][:,yeo_indices][u,v]
                        print('The shape of scaffold_current is: ', scaffold_current)
                scaffold_current[subjID]=scaffold_current[subjID]/len(file)
            else:
                for t in times:
                    if regions == 'ALL':
                        scaffold_current[subjID,:]+=file[str(t)][:][u,v]
                        print('The shape of scaffold_current is: ', scaffold_current)
                    else:
                        scaffold_current[subjID,:]+=file[str(t)][:][yeo_indices,:][:,yeo_indices][u,v]
                        print('The shape of scaffold_current is: ', scaffold_current)
                scaffold_current[subjID]=scaffold_current[subjID]/len(times)
    X = scaffold_current.copy()
    print('The shape of X for SCAFFOLD is: ', X.shape)
    return X

def preprocess_peaks_concat(peaks_data, data_all):
    peaks_data.dropna(inplace=True)
    peaks_data = peaks_data[peaks_data['Region'] == 'ALL']

    features_control = peaks_data['Feature'].unique()                       # list of the emotions
    features = [x for x in features_control if not x.startswith('Control')] # list of the emotions without the control ones
    thresholds = peaks_data['threshold'].unique()                           # list of the thresholds

    # Read the data for the PLS computed on all the movies concatenated
    data_all['Region'] = 'ALL'
    data_all['Type'] = 'bold'
    data_all.reset_index(inplace=True, drop=False)
    data_all.rename(columns={'index': 'bootstrap_round'}, inplace=True)

    return peaks_data, data_all, features, thresholds