import pandas as pd
import numpy as np
import csv 
import json
import os
from sklearn.utils.extmath import randomized_svd
import scipy.stats as stats
from numpy.random import RandomState,SeedSequence
from numpy.random import MT19937
from scipy import signal
import scipy
from sklearn.utils.validation import check_X_y
import nibabel as nib

def alignement(df,  offset_TR, onset_TR=72, affiche=False):
    """
    Aligns the fMRI to a specific start and offset 
    to return only signals measured during movies watching only 

    Input 
    -------
        - df (DataFrame) : voxel-wise series 
        - offset_TR (int) : specific time (in TR) corresponding to the end of the film 
        - Onset_TR (int): specific time (in TR) corresponding to the beginning of the film
        - affiche (Bool) : Determine if the number of scans removed is displayed
    Output 
    -------
        - df (DataFrame)
    """
    if(affiche):print(f"Number of scans :{len(df)} \n")
        
    ## Remove scans before movie wieving
    df.drop(range(0,onset_TR), axis=0, inplace=True)
    if(affiche):print(f"Number of scans after onset removing :{len(df)} \n")
        
    ## Remove scans after movie wieving    
    df.drop(range(offset_TR, len(df)), inplace=True)
    if(affiche):print(f"Number of scans after offset removing:{len(df)} \n")
       
    df.reset_index(drop=True, inplace=True)
    
    return df

def scrubbing(folder, onset, dur, vox, verbose=False, level=0.1):
    
    ## Read Motion File
    mc = pd.read_csv(f"{folder}/mc/prefiltered_func_data_mcf_rel.rms", header=None)
    
    ## Get index to remove & keep 
    mc=mc.iloc[onset:dur+onset-1].reset_index(drop=True)
    rmove=np.where(mc>level)[0]
    keep= np.where(mc<level)[0]
    
    ## Replace wrong values with None 
    if(len(rmove!=0)):
        pcs=len(keep)/(len(rmove)+len(keep))
        vox[rmove]=np.nan
        if(verbose) : print(f"% of data keep: {pcs*100} \n")
            
    return vox
 
def resampling(df, dur, resol=1.3):
    """
    Resamples and aligns the behavioral dataset to match the corresponding scan sets 
    
    Input 
    -------
        - df (DataFrame) : Behavorial Dataset
        - dur (int) : duration of the film
        - resol (float) : resolution of the fMRI dataset 
    Output
    -------
        - df (DataFrame) : behavioral dataset with the same resolution and size as the scans (fMRI) dataset
    
    """
    
    TRdur13 = round(np.shape(df)[0]/resol)
    ## Resampling
    df = pd.DataFrame(signal.resample(df,TRdur13), columns=df.columns)
    
    ## Alignement
    df.drop(range(dur-1, len(df)), axis=0, inplace=True)
    return df

# def standa(X,Y, durations):  
#     X_normed = X.copy()
#     Y_normed = Y.copy()
    
#     id_0=0
#     id_1=durations[0]
    
#     for dur in durations:  
#         id_1 = id_0+dur
#         X_normed[id_0:id_1]=(X_normed[id_0:id_1]-np.nanmean(X_normed[id_0:id_1], 
#                                                             axis=0))/(np.nanstd(X_normed[id_0:id_1],axis=0, ddof=0))
#         Y_normed[id_0:id_1]=(Y_normed[id_0:id_1]-np.nanmean(Y_normed[id_0:id_1],
#                                                             axis=0))/(np.nanstd(Y_normed[id_0:id_1],axis=0, ddof=0))
#         id_0+=dur
        
#     return X_normed, Y_normed
    
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
    ## Check X & Y dimension
    if(X.shape[0] != Y.shape[0]): raise Exception("Input arguments X and Y should have the same number of rows")
        
    R = (np.array(Y.T) @ np.array(X))
    return R


def SVD(R, ICA=False, seed=1, n_components=None,):
    """
    Singular Value Decomposition of R
    
    Input 
    -------
        - R (L (#LCs) x V Array) : Correlation Matrix
        - ICA (bool): if True turn LCs such that max is positive
        - n_component (int) : number of LCs to keep for the decomposition
        - seed (int)
    Output
    -------
        - U ( M X L (#LCs) DataFrame) : left singular vector matrix
        - S ( L x L (#LCs) DataFrame) : Diagonal Singular value matrix
        - V ( V x L (#LCs) Dataframe) :  Right Singular vector matrix Transposed
       
    """
    R = np.array(R)
    R = np.nan_to_num(R)

    n_components = min(R.shape)
    
    ## Run most computationally efficient SVD
    U, d, V = randomized_svd(R, n_components=n_components, random_state=seed)
    
    ## Get V instead of V.T
    V=V.T
    
    ## ICA convention
    result = np.where(np.abs(V)==np.amax(np.abs(V), axis=0))
    if(ICA):
        for i in range(len(result[0])):
            if(np.sign(V[result[0][i],result[1][i]]))<0 : 
                V[:,result[1][i]]=-V[:,result[1][i]]
                U[:,result[1][i]]=-U[:,result[1][i]]
    
    return pd.DataFrame(U), pd.DataFrame(np.diag(d)), pd.DataFrame(V)


def varexp(singular):
    """
    Computes the explained variance from the Singular values matrix 
   
    Input 
    -------
        - Singular (L x L (#LCs) DataFrame) : Singular matrix from SVD decomposition
    Ouptut
    -------
        - var (L(#LCs)x 1 vector) : explained variance for each singular value
    
    """
    ## Verify if matrix dimension 2x2 
    if singular.ndim != 2:
        raise ValueError('Provided DataFrame must be a square diagonal '
                         'matrix, not array of shape {}'
                         .format(singular.shape))
    ## Variance computation    
    var = (np.diag(singular)**2 / np.sum(np.diag(singular)**2))
    return var

def PLS_scores(X, Y, U, V):
    """
    Compute the PLS scores ("Brain" & "Design") by projecting the original data (X and Y) 
    onto their respective salience patterns (V and U)
    
    Input 
    -------
        - X (T x V DataFrame) : voxel-wise series 
        - Y (T x M DataFrame) : Emotional items 
        - U (M x L(#LCs)  DataFrame) : Left singular vector from SVD decomposition
        - V (V x L (#LCs)DataFrame) : Right Singular Vector from SVD decompositon (transposed)
    Output  
    -------
        - Lx (T x L(#LCs) Dataframe) : Imaging/Brain scores
        - Ly (T x L(#LCs) DataFrame) : Design/Behavior scores
    """
    Lx= X@np.array(V)
    Ly= Y@np.array(U)
    return Lx, Ly
        
def rotatemat (origlv, bootlv):
    """
    Compute Procrustean Transform (correction for axis rotation/reflection du)
    
    Input 
    -------
        - origlv : Original Matrix
        - bootlv : Matrix after resampling
    Output 
    -------
        - new_mat : New matrix with rotation correction to re-order Lvs as the original matrix
    
    """
    ## Define coordinate space between original and bootstrap LVs
    tmp=origlv.T@bootlv
    

    ## Orthogonalze space
    [V,W,U]=SVD(tmp);
   
    ## Determine procrustean transform
    new_mat=U@np.array(V).T

    return new_mat

def permu(X,Y,U,nPerms, seed=1):
    """
    Implementation of the Permutation testing for PLS
    
    Input 
    -------
        - X (T x V DataFrame): voxel-wise series (standarized)
        - Y (T x M DataFrame): Emotional items (standarized)
        - U (M x L(#LCs)  DataFrame) : Left singular vector from SVD decomposition
        - nPerms (int) :  number of permutation to perform
        - seed (int) 
    Output 
    -------
        - Sp_new (L (#LCs) x nPerms array): Permueted singular values, used to compute p-values to asses LCs significance
    """
    Sp_new=[]
    
    ## Check that dimensions of X & Y are correct
    if(X.shape[0] != Y.shape[0]): raise Exception("Input arguments X and Y should have the same number of rows")
    
    ## Fixe the seed
    rs = RandomState(MT19937(SeedSequence(seed)))
    
    ## Permutation
    for i in range(nPerms):
        
        Xp=X
        ## Permute Y rows 
        Yp=Y.sample(frac=1,replace=False, random_state=rs)
    
        ## Generate cross-covariance matrix between X and permuted Y
        Rp = R_cov(np.array(Xp), np.array(Yp))
 
        ## Singular value decomposition of Rp
        Up, Sp, Vp = SVD(np.array(Rp),seed )

        ## Procrustes rotation
        rot_mat=rotatemat(U,Up)
    
        Up = Up @ Sp @ rot_mat
        
        Sp = (np.sqrt(np.sum(np.square(Up), axis = 0)))
       
        ## Concatenate the resulting Singular values matrix
        Sp_new.append(Sp)
        
    return np.array(Sp_new).T

def myPLS_get_LC_pvals(Sp_vect,S,nPerms, seuil=0.01) : 
    """
    Compute p-values for all Latent Components (LCs) using the permuted singular values
    
    Input  
    -------
        - Sp_new (L (#LCs) x nPerms array): Permueted singular values
        - S ( L x L (#LCs) DataFrame) : Diagonal Singular value matrix
        - nPerms (int) : Number of Permutation
        - seuil : significant level (0.01 per default)
    Output  
    -------
        - sprob (L(#LCs) x 1 vector) : 
        - sign_PLC (vector) : indexe(s) of significant LCs 
    """
    sig_PLC=[]
    
    ## Check the number of permutations with Sp_perm greater than S 
    sp = np.sum(Sp_vect > np.diag(S)[:, None], axis=1) + 1
    
    ## Compute p-value for each LC- approximation by counting 
    ## the number of permuted singular values above the measured singular values
    sprob = sp / (Sp_vect.shape[-1] + 1)
    
    ## Index of significant LCs
    signif_LC = np.argwhere(sprob<seuil)
    
    ## Number of significant LCs
    nSignifLC = signif_LC.shape[0]
    
    ## Display & Concatenate significant LCs
    for i in range(nSignifLC):
        sig_PLC.append(signif_LC[i][0])
        #print(f"LC {sig_PLC[-1]} with p-value = {sprob[sig_PLC[-1]]}")
        
    return sprob, sig_PLC


        
def myPLS_bootstrapping(X0,Y0,U,V, nBoots, dur, seed=1):
    """
    Boostrap on X0 & Y0 and recompute SVD 
    
    Input 
    -------
    - X0 (T x V DataFrame) : Voxels-wise serie (not normalized)
    - Y0 (T x M DataFrame) : Behavior/design data (not normalized)
    - U (M x L(#LCs)  DataFrame) : Left singular vector from SVD decomposition
    - V (V x L (#LCs)DataFrame) : Right Singular Vector from SVD decompositon (transposed)
    - nBoots (int) : number of bootstrap sample 
    - seed (int)
    - type_ (str) : type of standarization (only z-scored, z-scored per films)
    - durations (array) : duration of each film used for the z-score per films standarization
    - stand : string defining the type of standarization (None for to use the standarization methods, emo to just apply z-scores)
    Output 
    -------
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
        
        
        Xb,Yb = standa(Xb,Yb,dur)
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

        
            
def boot_select(LC_index, boot_res, X, level=3): 
    """
   
    Select the important voxels based on the boot stability scores
    
    Inputs
    -------
    LC_index : int
        integer indices of the latent variables
    boot_res : pandas DataFrame
        Dataframe with the bootstrap stability scores
    X : pandas DataFrame
        Dataframe with the original input data
    level : float
        The cutoff threshold for selecting voxels
    
    Ouput
    -------
    select_X : numpy array
        Array of selected voxels
    """
    select_X=np.zeros_like(X.iloc[:, LC_index])
    index = np.argwhere(np.array(abs(boot_res.iloc[:,LC_index]))>level)
    select_X[index[:,0]]=X.iloc[index[:,0], LC_index]
    return select_X, index

def corr_brain_maps (Nifti1, Nifti2, LV_id) : 
    nifti_file1 = nib.load(Nifti1)
    nifti_data1 = nifti_file1.get_fdata()
    
    
    # Load the second Nifti file
    nifti_file2 = nib.load(Nifti2)
    nifti_data2 = nifti_file2.get_fdata()
    
    # Flatten the arrays to 1D
    array1 = nifti_data1.flatten()
    array2 = nifti_data2.flatten()
    # Calculate the correlation coefficient
    correlation =scipy.stats.pearsonr(array1, array2)
    # Print the correlation coefficient
    print(f"Correlation coefficient for the LV {LV_id}:", correlation)
    
def corr_behav_saliences(df1, df2, LV_id) : 
   
    # Calculate the correlation coefficient
    correlation = scipy.stats.pearsonr(np.array(df1[LV_id[0]]), np.array(df2[LV_id[1]]))
    # Print the correlation coefficient
    print(f"Correlation coefficient for the LV {LV_id[0]+1} & {LV_id[1]+1}", correlation)

def zscore(mat):
    """Performs z-scoring over matrix columns"""
    zscored_mat = mat.copy()
    for col in range(mat.shape[1]):
        current_col = mat[:,col]
        current_mean = np.mean(current_col)
        current_std = np.std(current_col)
        new_col = (current_col - current_mean)/current_std
        zscored_mat[:,col] = new_col
    return zscored_mat