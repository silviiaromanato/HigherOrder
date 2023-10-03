from BehavPLS_opt import BehavPLS
from compute_opt import *
import pickle as pk
from scipy.io import loadmat
from scipy.stats import zscore
import glob
import tqdm
import h5py
from sklearn.manifold import Isomap,MDS
import umap.umap_ as umap
from itertools import combinations


def set_default_parameter():
    ##Number of participants
    nb= 100
    # Number of permutations for significance testing
    nPer = 1000
    # Number of bootstrap iterations
    nBoot = 1000
    # Seed
    seed = 10
    # signficant level for statistical testing
    seuil=0.05
    return(nb,nPer,nBoot,seed,seuil)  

def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]  

def compute_edgeTS(time_series):
    N,T = time_series.shape
    u, v = np.triu_indices(N, k=1)
    return(np.array(time_series[u,:]*time_series[v,:]))

def save_pkl (data, name): 
    with open(f'./pkl/{name}.pkl', 'wb') as f:
        pk.dump(data, f)
    with open(f'./pkl/{name}.pkl', 'rb') as f:
        loaded_dict = pk.load(f)

def load_Xdata(method_label):
    path_subjects='/home/asantoro/HCP/'
    if method_label == 'BOLD':
        X_mat= load_BOLD_data(path_subjects)
    if method_label == 'edges':
        X_mat= load_edges_data(path_subjects)
    if method_label == 'triangles':
        X_mat= load_triangles_data(path_subjects)
    if method_label == 'scaffold':
        X_mat= load_scaffold_data(path_subjects)
    return (X_mat)

def loading_yeo(path="/home/asantoro/HCP_misc/Misc/yeoOrder/yeo_RS7_Schaefer100S.mat"):
    ##Loading the yeoROIS
    yeoROIs=np.array([i[0]-1 for i in loadmat(path)['yeoROIs']])
    yeoROI_dict={label_Yeo:np.where(yeoROIs==idx_Yeo)[0] for idx_Yeo,label_Yeo in enumerate(['VIS','SM','DA','VA','L','FP','DMN','SC','C'])}
    yeoROI_dict['SC']=np.array(sorted(np.hstack((yeoROI_dict['SC'],yeoROI_dict['C']))))
    del yeoROI_dict['C']
    return(yeoROI_dict)


def load_Xdata_subnetworks(method_label,label_YEO):
    path_subjects='/home/asantoro/HCP/'
    yeoROI_dict=loading_yeo()
    if method_label == 'BOLD':
        X_mat= load_BOLD_data_subnetwork(path_subjects,yeoROI_dict,label_YEO)
    if method_label == 'edges':
        X_mat= load_edges_data_subnetwork(path_subjects,yeoROI_dict,label_YEO)
    if method_label == 'triangles':
        X_mat= load_triangles_data_subnetwork(path_subjects,yeoROI_dict,label_YEO)
    if method_label == 'scaffold':
        X_mat= load_scaffold_data_subnetwork(path_subjects,yeoROI_dict,label_YEO)
    return (X_mat)

def load_BOLD_data(path_subjects):
    
    subject_FC_data={}
    list_subjs=np.array(sorted([i.split('/')[-1] for i in glob.glob(path_subjects+'*')]))
    for idx,i in enumerate(list_subjs):
        ##Loading day 1
        path_data=path_subjects+i+'/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        data_day1=loadmat(path_data)['TS'] ## Size: (119, 1200)                

        ##Loading day 2
        path_data=path_subjects+i+'/rfMRI_REST2_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        data_day2=loadmat(path_data)['TS'] ## Size: (119, 1200)
        
        subject_FC_data[(i,1)]=upper_tri_masking(np.corrcoef(data_day1))
        subject_FC_data[(i,2)]=upper_tri_masking(np.corrcoef(data_day2))
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(subject_FC_data[(subjID,1)]+subject_FC_data[(subjID,2)]))
    X_mat=np.array(X_mat)
    return(X_mat)

def load_BOLD_data_subnetwork(path_subjects,yeoROI_dict,label_YEO):
    if isinstance(label_YEO, str):
        indices_yeo=yeoROI_dict[label_YEO]
    else:
        indices_yeo=[]
        for l in label_YEO:
            indices_yeo.extend(yeoROI_dict[l])
        indices_yeo=np.array(sorted(list(set(indices_yeo))))
    subject_FC_data={}
    list_subjs=np.array(sorted([i.split('/')[-1] for i in glob.glob(path_subjects+'*')]))
    for idx,i in enumerate(list_subjs):
        ##Loading day 1
        path_data=path_subjects+i+'/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        data_day1=loadmat(path_data)['TS'] ## Size: (119, 1200)                

        ##Loading day 2
        path_data=path_subjects+i+'/rfMRI_REST2_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        data_day2=loadmat(path_data)['TS'] ## Size: (119, 1200)
        
        subject_FC_data[(i,1)]=upper_tri_masking(np.corrcoef(data_day1[:,indices_yeo]))
        subject_FC_data[(i,2)]=upper_tri_masking(np.corrcoef(data_day2[:,indices_yeo]))
        print('The shape of the data is: ',subject_FC_data[(i,1)].shape)
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(subject_FC_data[(subjID,1)]+subject_FC_data[(subjID,2)]))
    X_mat=np.array(X_mat)
    print('The shape of the data is: ',X_mat.shape)
    return(X_mat)

def load_edges_data(path_subjects):
    subject_eFC_data={}

    list_subjs=np.array(sorted([i.split('/')[-1] for i in glob.glob(path_subjects+'*')]))
    for idx,subjID in enumerate(list_subjs):
        ##Loading day 1
        path_data=path_subjects+subjID+'/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        data_day1=loadmat(path_data)['TS'] ## Size: (119, 1200)                
        
        ##Loading day 2
        path_data=path_subjects+subjID+'/rfMRI_REST2_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        data_day2=loadmat(path_data)['TS'] ## Size: (119, 1200)
        
        subject_eFC_data[(subjID,1)]=upper_tri_masking(np.corrcoef(compute_edgeTS(data_day1)))
        subject_eFC_data[(subjID,2)]=upper_tri_masking(np.corrcoef(compute_edgeTS(data_day2)))
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(subject_eFC_data[(subjID,1)]+subject_eFC_data[(subjID,2)]))
    X_mat=np.array(X_mat)
    return(X_mat)

def load_edges_data_subnetwork(path_subjects,yeoROI_dict,label_YEO):
    if isinstance(label_YEO, str):
        indices_yeo=yeoROI_dict[label_YEO]
    else:
        indices_yeo=[]
        for l in label_YEO:
            indices_yeo.extend(yeoROI_dict[l])
        indices_yeo=np.array(sorted(list(set(indices_yeo))))
    subject_eFC_data={}

    list_subjs=np.array(sorted([i.split('/')[-1] for i in glob.glob(path_subjects+'*')]))
    for idx,subjID in enumerate(list_subjs):
        ##Loading day 1
        path_data=path_subjects+subjID+'/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        data_day1=loadmat(path_data)['TS'] ## Size: (119, 1200)                
        
        ##Loading day 2
        path_data=path_subjects+subjID+'/rfMRI_REST2_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        data_day2=loadmat(path_data)['TS'] ## Size: (119, 1200)
        
        subject_eFC_data[(subjID,1)]=upper_tri_masking(np.corrcoef(compute_edgeTS(data_day1[indices_yeo,:])))
        subject_eFC_data[(subjID,2)]=upper_tri_masking(np.corrcoef(compute_edgeTS(data_day2[indices_yeo,:])))
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(subject_eFC_data[(subjID,1)]+subject_eFC_data[(subjID,2)]))
    X_mat=np.array(X_mat)
    return(X_mat)


def load_triangles_data(path_subjects,N=119,T=1200):
    subject_triangles_data={}
    list_subjs=np.array(sorted([i.split('/')[-1] for i in glob.glob(path_subjects+'*')]))
    path_data="/home/asantoro/COST/Project2_Higher_Order_Brain/Results/02_HCP_subjects_100_all/"
    path="/home/asantoro/COST/Project2_Higher_Order_Brain/Jupyter/"
    if os.path.isfile(path+'data_triangles_REST.pkl'):
        subject_triangles_data={}
        with open(path+'data_triangles_REST.pkl','rb+') as f:
            subject_triangles_data=pk.load(f)
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(subject_triangles_data[(subjID,1)]+subject_triangles_data[(subjID,2)]))
    X_mat=np.array(X_mat)
    return(X_mat)

def load_triangles_data_subnetwork(path_subjects,yeoROI_dict,label_YEO,flag_triangles=3,N=119):
    indices_yeo_all=[]
    if isinstance(label_YEO, str):
        indices_yeo=yeoROI_dict[label_YEO]
    else:
        indices_yeo=[]
        for l in label_YEO:
            indices_yeo.extend(yeoROI_dict[l])
        indices_yeo=np.array(sorted(list(set(indices_yeo))))
    for idx_triangles,(i,j,k) in enumerate(combinations(np.arange(N),3)):
            flag=[i in indices_yeo, j in indices_yeo, k in indices_yeo]
            if sum(flag) == flag_triangles: ## All the nodes belong to the same Yeo networks
                indices_yeo_all.append(idx_triangles)
    indices_yeo_all=np.array(indices_yeo_all)

    subject_triangles_data={}
    list_subjs=np.array(sorted([i.split('/')[-1] for i in glob.glob(path_subjects+'*')]))
    path_data="/home/asantoro/COST/Project2_Higher_Order_Brain/Results/02_HCP_subjects_100_all/"
    path="/home/asantoro/COST/Project2_Higher_Order_Brain/Jupyter/"
    if os.path.isfile(path+'data_triangles_REST.pkl'):
        subject_triangles_data={}
        with open(path+'data_triangles_REST.pkl','rb+') as f:
            subject_triangles_data=pk.load(f)
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(subject_triangles_data[(subjID,1)][indices_yeo_all]+subject_triangles_data[(subjID,2)][indices_yeo_all]))
    X_mat=np.array(X_mat)
    return(X_mat)


def load_scaffold_data(path_subjects,N=119,T=1200):
    data_scaffold_REST={}
    list_subjs=np.array(sorted([i.split('/')[-1] for i in glob.glob(path_subjects+'*')]))
    path_data="/home/asantoro/COST/Project2_Higher_Order_Brain/Results/02_HCP_subjects_100_all/"
    path="/home/asantoro/COST/Project2_Higher_Order_Brain/Jupyter/"
    if os.path.isfile(path+'data_scaffold_REST.pkl'):
        data_scaffold_REST={}
        with open(path+'data_scaffold_REST.pkl','rb+') as f:
            data_scaffold_REST=pk.load(f)
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(upper_tri_masking(data_scaffold_REST[(subjID,1)])+upper_tri_masking(data_scaffold_REST[(subjID,2)])))
    X_mat=np.array(X_mat)
    return(X_mat)


def load_scaffold_data_subnetwork(path_subjects,yeoROI_dict,label_YEO):
    if isinstance(label_YEO, str):
        indices_yeo=yeoROI_dict[label_YEO]
    else:
        indices_yeo=[]
        for l in label_YEO:
            indices_yeo.extend(yeoROI_dict[l])
        indices_yeo=np.array(sorted(list(set(indices_yeo))))
    data_scaffold_REST={}
    list_subjs=np.array(sorted([i.split('/')[-1] for i in glob.glob(path_subjects+'*')]))
    path_data="/home/asantoro/COST/Project2_Higher_Order_Brain/Results/02_HCP_subjects_100_all/"
    path="/home/asantoro/COST/Project2_Higher_Order_Brain/Jupyter/"
    if os.path.isfile(path+'data_scaffold_REST.pkl'):
        data_scaffold_REST={}
        with open(path+'data_scaffold_REST.pkl','rb+') as f:
            data_scaffold_REST=pk.load(f)
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(upper_tri_masking(data_scaffold_REST[(subjID,1)][indices_yeo,:][:,indices_yeo])+
                          upper_tri_masking(data_scaffold_REST[(subjID,2)][indices_yeo,:][:,indices_yeo])))
    X_mat=np.array(X_mat)
    return(X_mat)


if __name__ == "__main__":  
    path_data_behavioral='/home/asantoro/COST/Project2_Higher_Order_Brain/Alessandra_Griffa_code_PLSC/share_Andrea/New_Analyses_PLSC/matlab_matrix_HCP_data.mat'
    Y=loadmat(path_data_behavioral)
    Ylabel=[i[0] for i in Y['domains'][0]] ## This corresponds to the behavioral data labels (i.e. 10 cognitive scores)
    Y=Y['Bpca'] ## (these are the scores for the different subjects, array of 100x10)

    nb,nPer,nBoot,seed,seuil = set_default_parameter()

    # for method in ['BOLD','triangles','scaffold']:
    functional_network='ALL'
    list_functional_networks=['VIS','SM','DA','VA','L','FP','DMN','SC']
    # for idx1,functional_network1 in enumerate(list_functional_networks):
    #     for idx2,functional_network2 in enumerate(list_functional_networks[idx1:]):
    for method in ['BOLD','triangles','scaffold']:
    # for method in ['edges']:
        # functional_network=(functional_network1,functional_network2)
        if functional_network=='ALL':
            X=load_Xdata(method)    
        else:
            X=load_Xdata_subnetworks(method,functional_network)
        print("doing the following method: %s --- Functional network: %s" % (method,functional_network))
        # isomap = MDS(n_components=100,max_iter=300)
        # X = isomap.fit_transform(X)
        dataset=BehavPLS(X,Y,nb_sub=nb,nPerms=nPer,nBoot=nBoot,seed=seed,seuil=seuil,verbose=True)
        res_decompo = dataset.run_decomposition()
        save_pkl(res_decompo, f"pls_res_{method}_{functional_network}")
       
       
        res_permu=dataset.permutation()
        save_pkl(res_permu, f"perm_res_{method}_{functional_network}")
            
        res_bootstrap = dataset.bootstrap()
        save_pkl(res_bootstrap, f"boot_res_{method}_{functional_network}")


        ##### Dimensionality Reduction

        X = umap.UMAP(random_state=42,n_components=100,init='random').fit_transform(X)

        dataset=BehavPLS(X,Y,nb_sub=nb,nPerms=nPer,nBoot=nBoot,seed=seed,seuil=seuil,verbose=True)
        res_decompo = dataset.run_decomposition()
        save_pkl(res_decompo, f"pls_res_{method}_{functional_network}_UMAP")
       
       
        res_permu=dataset.permutation()
        save_pkl(res_permu, f"perm_res_{method}_{functional_network}_UMAP")
            
        res_bootstrap = dataset.bootstrap()
        save_pkl(res_bootstrap, f"boot_res_{method}_{functional_network}_UMAP")


        