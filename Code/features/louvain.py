import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helper_corr_mtx import *
import networkx as nx
import numpy as np
import pandas as pd
import sys
from clusim.clustering import Clustering, print_clustering
import clusim.sim as sim
import community.community_louvain
import community
import os

def threshold_matrix_creation(matrix, movie, lower_threshold=5, upper_threshold=95):
    corr_features = threshold_matrix_lower_upper(matrix, perc_lower= lower_threshold, perc_upper=upper_threshold)
    return corr_features

def compute_modified_modularity_function(thresh_mat):
    
    def update_consensus_matrix(cluster, num_nodes):
        result = np.zeros((num_nodes, num_nodes))
        for i in cluster:
            for j in cluster:
                result[i, j] = 1
        return result

    # Create the graph object that will be used for community detection
    G = nx.Graph(thresh_mat)
    num_nodes = len(G)

    # Creating the p_ij matrix that will contain the probability of belonging to a given cluster 
    Consensus_matrix = np.zeros(thresh_mat.shape)

    # Define number of iterations
    N_iter = 500
    
    # Perform iterations
    for iter in range(N_iter):
        if iter % 100 == 0:
            print(f'Iteration {iter}')
        clusters = community.best_partition(G, random_state=iter)
        clusters = {k: [node for node, clust in clusters.items() if clust == k] for k in set(clusters.values())}
        for cluster in clusters:
            cluster_matrix = update_consensus_matrix(clusters[cluster], num_nodes)
            Consensus_matrix += cluster_matrix

    # Normalize the consensus matrix
    Consensus_matrix /= N_iter
    
    # Initialize the new Graph with adjacency matrix = consensus matrix
    G_consensus = nx.Graph(Consensus_matrix)
    
    # Compute final clustering based on the consensus graph   
    final_clusters = community.best_partition(G_consensus, random_state=1)
   
    return final_clusters

def cluster_matrix(init_matrix, final_cls):
    cluster_assignment_matrix = np.full(init_matrix.shape, np.nan)
    for cluster_id, cluster in enumerate(final_cls, start=1):
        for i in cluster:
            for j in cluster:
                if i != j:  # Assign the cluster number only if i and j are different
                    cluster_assignment_matrix[i, j] = cluster_id
    return cluster_assignment_matrix

def elem2clu(final_clusters):
    element_to_cluster_features = {}
    for cluster_id, cluster in enumerate(final_clusters, start=1):
        for element in cluster:
            element_to_cluster_features[element] = cluster_id
    return element_to_cluster_features

def compute_similarity_mtx(ecs):
    n = len(ecs)
    sim_mtx = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            element = ecs[list(ecs.keys())[i]], ecs[list(ecs.keys())[j]]
            sim_mtx[i,j] = sim.element_sim(element[0], element[1])
    return sim_mtx

def correlation_mtx_fmri(movie_name, method):
    PATH_DATA = '/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/FMRI'
    if method == 'bold':
        mtx_tt =pd.read_csv(PATH_DATA + f'/average_{method}/corr_matrix_ave_{method.upper()}_{movie_name}.txt', sep=' ', header=None)
    elif method == 'scaffold':
        mtx_tt =pd.read_csv(PATH_DATA + f'/average_{method}/corr_matrix_ave_{method}sfrequency_{movie_name}.txt', sep=' ', header=None)
    else:
        mtx_tt =pd.read_csv(PATH_DATA + f'/average_{method}/corr_matrix_ave_{method}_{movie_name}.txt', sep=' ', header=None)
    mtx_tt = mtx_tt.fillna(0)
    return mtx_tt

if __name__ == '__main__': 
    
    movie = sys.argv[1]
    lower_threshold = sys.argv[2]
    upper_threshold = sys.argv[3]
    emotions = sys.argv[4]

    print('The emotions are: ', emotions == 1, type(emotions))
    if emotions == 1:
        print('we are performing the louvain on the emotions.')
        types = ['features', 'emo1', 'emo2', 'emo3', 'emo4']
    elif emotions == 0:
        types = ['features', 'bold', 'edges', 'scaffold', 'triangles']
    print('The types are:', types)
    
    corr_ ={}
    for i, type in enumerate(types):
        corr_[type] = correlation_mtx_features(movie, columns = ['spectralflux', 'rms', 'zcrs'], 
                                            columns_images= ['average_brightness_left', 'average_saturation_left', 'average_hue_left',
                                                            'average_brightness_right', 'average_saturation_right', 'average_hue_right'])
        if emotions:
            corr_[type] = emo_corr_matrix(movie, emotion = 0)
            corr_[type] = emo_corr_matrix(movie, emotion = 1)
            corr_[type] = emo_corr_matrix(movie, emotion = 2)
            corr_[type] = emo_corr_matrix(movie, emotion = 3)
        else:
            corr_[type] = correlation_mtx_fmri(movie, type)
            corr_[type] = correlation_mtx_fmri(movie, type)
            corr_[type] = correlation_mtx_fmri(movie, type)
            corr_[type] = correlation_mtx_fmri(movie, type)

    print('Thresholding the matrices')
    thr_ = {}
    for type in types:
        thr_[type] = threshold_matrix_creation(corr_[type], movie, lower_threshold, upper_threshold)

    print('Computing the final clusters')
    final_clusters = {}
    for type in types:
        if not os.path.exists(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/louvain/{movie}_clusters_{type}.json'):    
            final_clusters[type] = compute_modified_modularity_function(thr_[type])
            with open(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/louvain/{movie}_clusters_{type}.json', 'w') as fp:
                json.dump(final_clusters[type], fp)
        else:
            with open(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/louvain/{movie}_clusters_{type}.json', 'r') as fp:
                final_clusters[type] = json.load(fp)

        for key in final_clusters[type]:
            final_clusters[type][key] = [final_clusters[type][key]]

    print('Computing the ecs')
    ecs = {}
    for type in types:
        ecs[type] = Clustering(final_clusters[type])
    
    print('Computing the similarity')
    sim_mtx = compute_similarity_mtx(ecs)

    # save the results
    print('Saving the results...')
    if emotions == 1:
        print('Saving the results for emotions: path is ', f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/louvain/{movie}_sim_mtx_emo.csv')
        np.savetxt(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/louvain/{movie}_sim_mtx_emo.csv', sim_mtx, delimiter=',')
    else:
        print('Saving the results for fmri: path is ', f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/louvain/{movie}_sim_mtx_fmri.csv')
        np.savetxt(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/louvain/{movie}_sim_mtx_fmri.csv', sim_mtx, delimiter=',')
    print('Done!\n')