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

def mtx_creation(movie):

    corr_features = correlation_mtx_features(movie, columns = ['spectralflux', 'rms', 'zcrs'], 
                                            columns_images= ['average_brightness_left', 'average_saturation_left', 'average_hue_left',
                                                            'average_brightness_right', 'average_saturation_right', 'average_hue_right'])
    corr_emo1 = emo_corr_matrix(movie, emotion = 0)
    corr_emo2 = emo_corr_matrix(movie, emotion = 1)
    corr_emo3 = emo_corr_matrix(movie, emotion = 2)
    corr_emo4 = emo_corr_matrix(movie, emotion = 3)

    return corr_features, corr_emo1, corr_emo2, corr_emo3, corr_emo4

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
        clusters = community.best_partition(G, random_state=iter)
        print('The clusters are: ', clusters, 'and the type is: ', type(clusters))
        # create a dictionary with key = cluster_id and value = list of nodes in the cluster
        clusters = {k: [node for node, clust in clusters.items() if clust == k] for k in set(clusters.values())}
        print('The clusters are: ', clusters, 'and the type is: ', type(clusters))
        for cluster in clusters:
            print('The cluster is: ', cluster)
            cluster_matrix = update_consensus_matrix(cluster, num_nodes)
            Consensus_matrix += cluster_matrix

    # Normalize the consensus matrix
    Consensus_matrix /= N_iter
    
    # Initialize the new Graph with adjacency matrix = consensus matrix
    G_consensus = nx.Graph(Consensus_matrix)
    
    # Compute final clustering based on the consensus graph   
    final_clusters = community.best_partition(G_consensus, random_state=1) #change seed?
   
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

def compute_similarity_mtx(ecs_features, ecs_emo1, ecs_emo2, ecs_emo3, ecs_emo4):
    # create a similarity mariix
    sim_mtx = np.zeros((5,5))
    sim_mtx[0,0] = sim.element_sim(ecs_features, ecs_features, method='nmi')
    sim_mtx[0,1] = sim.element_sim(ecs_features, ecs_emo1, method='nmi')
    sim_mtx[0,2] = sim.element_sim(ecs_features, ecs_emo2, method='nmi')
    sim_mtx[0,3] = sim.element_sim(ecs_features, ecs_emo3, method='nmi')
    sim_mtx[0,4] = sim.element_sim(ecs_features, ecs_emo4, method='nmi')
    sim_mtx[1,0] = sim.element_sim(ecs_emo1, ecs_features, method='nmi')
    sim_mtx[1,1] = sim.element_sim(ecs_emo1, ecs_emo1, method='nmi')
    sim_mtx[1,2] = sim.element_sim(ecs_emo1, ecs_emo2, method='nmi')
    sim_mtx[1,3] = sim.element_sim(ecs_emo1, ecs_emo3, method='nmi')
    sim_mtx[1,4] = sim.element_sim(ecs_emo1, ecs_emo4, method='nmi')
    sim_mtx[2,0] = sim.element_sim(ecs_emo2, ecs_features, method='nmi')
    sim_mtx[2,1] = sim.element_sim(ecs_emo2, ecs_emo1, method='nmi')
    sim_mtx[2,2] = sim.element_sim(ecs_emo2, ecs_emo2, method='nmi')
    sim_mtx[2,3] = sim.element_sim(ecs_emo2, ecs_emo3, method='nmi')
    sim_mtx[2,4] = sim.element_sim(ecs_emo2, ecs_emo4, method='nmi')
    sim_mtx[3,0] = sim.element_sim(ecs_emo3, ecs_features, method='nmi')
    sim_mtx[3,1] = sim.element_sim(ecs_emo3, ecs_emo1, method='nmi')
    sim_mtx[3,2] = sim.element_sim(ecs_emo3, ecs_emo2, method='nmi')
    sim_mtx[3,3] = sim.element_sim(ecs_emo3, ecs_emo3, method='nmi')
    sim_mtx[3,4] = sim.element_sim(ecs_emo3, ecs_emo4, method='nmi')
    sim_mtx[4,0] = sim.element_sim(ecs_emo4, ecs_features, method='nmi')
    sim_mtx[4,1] = sim.element_sim(ecs_emo4, ecs_emo1, method='nmi')
    sim_mtx[4,2] = sim.element_sim(ecs_emo4, ecs_emo2, method='nmi')
    sim_mtx[4,3] = sim.element_sim(ecs_emo4, ecs_emo3, method='nmi')
    sim_mtx[4,4] = sim.element_sim(ecs_emo4, ecs_emo4, method='nmi')
    return sim_mtx

if __name__ == '__main__': 
    
    movie = sys.argv[1]
    lower_threshold = sys.argv[2]
    upper_threshold = sys.argv[3]

    corr_features, corr_emo1, corr_emo2, corr_emo3, corr_emo4 = mtx_creation(movie)
    
    # Threshold the matrices
    print('Thresholding the matrices')
    corr_features = threshold_matrix_creation(corr_features, movie)
    corr_emo1 = threshold_matrix_creation(corr_emo1, movie)
    corr_emo2 = threshold_matrix_creation(corr_emo2, movie)
    corr_emo3 = threshold_matrix_creation(corr_emo3, movie)
    corr_emo4 = threshold_matrix_creation(corr_emo4, movie)

    # Compute the final clusters
    print('Computing the final clusters')
    final_clusters_features = compute_modified_modularity_function(corr_features)
    final_clusters_emo1 = compute_modified_modularity_function(corr_emo1)
    final_clusters_emo2 = compute_modified_modularity_function(corr_emo2)
    final_clusters_emo3 = compute_modified_modularity_function(corr_emo3)
    final_clusters_emo4 = compute_modified_modularity_function(corr_emo4)

    # compute element to clster dictionary
    print('Computing the element to cluster dictionary')
    element_to_cluster_features = elem2clu(final_clusters_features)
    element_to_cluster_emo1 = elem2clu(final_clusters_emo1)
    element_to_cluster_emo2 = elem2clu(final_clusters_emo2)
    element_to_cluster_emo3 = elem2clu(final_clusters_emo3)
    element_to_cluster_emo4 = elem2clu(final_clusters_emo4)

    # compute the ecs
    print('Computing the ecs')
    ecs_features = Clustering(element_to_cluster_features)
    ecs_emo1 = Clustering(element_to_cluster_emo1)
    ecs_emo2 = Clustering(element_to_cluster_emo2)
    ecs_emo3 = Clustering(element_to_cluster_emo3)
    ecs_emo4 = Clustering(element_to_cluster_emo4)

    # compute the similarity
    print('Computing the similarity')
    sim_mtx = compute_similarity_mtx(ecs_features, ecs_emo1, ecs_emo2, ecs_emo3, ecs_emo4)

    # save the results
    np.savetxt(f'/Users/silviaromanato/Desktop/SEMESTER_PROJECT/Material/Output/{movie}_sim_mtx.csv', sim_mtx, delimiter=',')