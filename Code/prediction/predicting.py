import numpy as np
import pandas as pd
from scipy.io import loadmat
from compute_pred import *
from helpers_pred import *
import sys 
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import warnings
warnings.simplefilter('ignore', DeprecationWarning)

def cpm(X_train, y_train, threshold, movie, method, region):
    # 1. INPUTS: Divide the data into training and test sets
    M, N = X_train.shape[1], X_train.shape[0] 
    all_mats, all_behav = X_train, y_train
    all_mats = np.array(all_mats)
    all_behav = np.array(all_behav)

    # Preallocate prediction arrays
    behav_pred_pos = np.zeros((N, 1))
    behav_pred_neg = np.zeros((N, 1))
    
    # 2. CROSS VALIDATION: Perform leave-one-out procedure
    for leftout in range(N):
        # print(f'\nLeaving out subj # {leftout + 1:06.3f}')
        start = time.time()

        # Leave out subject from matrices and behavior
        train_mats = np.delete(all_mats, leftout, axis=0)
        train_behav = np.delete(all_behav, leftout, axis=0)

        # 3. RELATE EDGES: Correlate all edges with behavior
        r_mat = np.zeros((M))
        p_mat = np.zeros((M))

        for i in range(M):
            r, p = scipy.stats.pearsonr(train_mats[:, i], train_behav[:])
            r_mat[i] = r
            p_mat[i] = p

        # 4. EDGE SELECTION: Set threshold and define masks
        pos_mask = np.multiply(p_mat < threshold, r_mat > 0)
        neg_mask = np.multiply(p_mat < threshold, r_mat < 0)

        # 5. SINGLE SUBJ SUMMARY VALUES: Compute sum scores
        behav_sum_pos = np.zeros((N - 1, 1))
        behav_sum_neg = np.zeros((N - 1, 1))

        # Divide by two to control for the fact that we're counting each pair twice because the matrix is symmetric
        for i in range(N - 1):
            behav_sum_pos[i] = np.sum(np.multiply(train_mats[i, :], pos_mask))
            behav_sum_neg[i] = np.sum(np.multiply(train_mats[i, :], neg_mask))

        # 6. MODEL FITTING: Fit model on training set
        lin_model_pos = LinearRegression()
        lin_model_pos.fit(behav_sum_pos, train_behav)

        lin_model_neg = LinearRegression()
        lin_model_neg.fit(behav_sum_neg, train_behav)

        # 7. PREDICTION: Predict on left-out subject
        # Extract sum scores for left-out subject
        leftout_sum_pos = np.sum(np.multiply(all_mats[leftout], pos_mask))
        leftout_sum_neg = np.sum(np.multiply(all_mats[leftout], neg_mask))
        leftout_sum_pos = leftout_sum_pos.reshape(1,1)
        leftout_sum_neg = leftout_sum_neg.reshape(1,1)

        # Predict behavior for left-out subject
        behav_pred_pos[leftout] = lin_model_pos.predict(leftout_sum_pos)
        behav_pred_neg[leftout] = lin_model_neg.predict(leftout_sum_neg)

        end = time.time()
        minutes = (end - start) / 60
        seconds = (end - start) % 60

        # print(f'This iteration took: {minutes:06.3f} minutes and {seconds:06.3f} seconds')

    # 8. EVALUATION: Compute correlations between predicted and observed behavior
    behav_pred_corr_pos = np.array(scipy.stats.pearsonr(all_behav, behav_pred_pos[:, 0]))
    behav_pred_corr_neg = np.array(scipy.stats.pearsonr(all_behav, behav_pred_neg[:, 0]))
    print(behav_pred_corr_pos, behav_pred_corr_neg)

    # compute mean squared error
    mse_pos = mean_squared_error(all_behav, behav_pred_pos[:, 0])
    mse_neg = mean_squared_error(all_behav, behav_pred_neg[:, 0])

    # 9. VISUALIZATION: Make a pretty figure
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(all_behav, behav_pred_pos)
    plt.xlabel('Observed behavior')
    plt.ylabel('Predicted behavior')
    plt.title('Positive edges: p-val ', behav_pred_corr_pos[1])
    plt.subplot(1, 2, 2)
    plt.scatter(all_behav, behav_pred_neg)
    plt.xlabel('Observed behavior')
    plt.ylabel('Predicted behavior')
    plt.title('Negative edges: p-val ', behav_pred_corr_neg[1])
    plt.suptitle(f'CPM: {movie}, {method}, {region}')
    plt.tight_layout()
    plt.savefig(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/prediction/images/CPM_{movie}_{method}_{region}.png')

    return behav_pred_corr_neg, behav_pred_corr_pos, mse_pos, mse_neg

PATH_YEO = '/media/miplab-nas2/Data2/Movies_Emo/Silvia/HigherOrder/Data/yeo_RS7_Schaefer100S.mat'
columns = ['BIG5_ext', 'BIG5_agr', 'BIG5_con', 'BIG5_neu', 'BIG5_ope']

if __name__ == '__main__': 
    PATH = sys.argv[1]
    movie_name = sys.argv[2]
    method = sys.argv[3]
    PATH_DATA = sys.argv[4]
    region = sys.argv[5]

    yeo_dict = loading_yeo(PATH_YEO)

    # Load the Y behavioural dataset
    Y = pd.read_csv(PATH_DATA, sep='\t', header=0)[columns]
    extrovercy = Y['BIG5_ext']
    agreeableness = Y['BIG5_agr']
    conscientiousness = Y['BIG5_con']
    neuroticism = Y['BIG5_neu']
    openness = Y['BIG5_ope']

    # Normalize the data
    # extrovercy = (extrovercy - extrovercy.mean()) / extrovercy.std()
    # agreeableness = (agreeableness - agreeableness.mean()) / agreeableness.std()
    # conscientiousness = (conscientiousness - conscientiousness.mean()) / conscientiousness.std()
    # neuroticism = (neuroticism - neuroticism.mean()) / neuroticism.std()
    # openness = (openness - openness.mean()) / openness.std()

    print('\n' + ' -' * 10 + f' for {method}, {movie_name} and {region} FOR: ', movie_name, ' -' * 10)

    X_movie = compute_X(PATH, movie_name, method=method, regions = region)
    X_movie = pd.DataFrame(X_movie)

    # Perform the CPM Behavioural analysis
    threshold = 0.01

    #X_train, X_test, y_train, y_test = train_test_split(X, extrovercy, test_size=0.2, random_state=0)

    # compute the CPM
    behav_pred_corr_neg, behav_pred_corr_pos, mse_pos, mse_neg = cpm(X_movie, extrovercy, threshold, movie_name, method, region)

    print('The correlation between the predicted and the observed behaviour for negative is: ', behav_pred_corr_neg[0], 'with a p-value of: ', behav_pred_corr_neg[1])
    print('The correlation between the predicted and the observed behaviour for positive is: ', behav_pred_corr_pos[0], 'with a p-value of: ', behav_pred_corr_pos[1])

    print('The mean squared error for negative is: ', mse_neg)
    print('The mean squared error for positive is: ', mse_pos)

    # Save the results
    PATH_SAVE = f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/prediction/CPM_{method}_{region}_results.csv'
    if os.path.exists(PATH_SAVE):
        CPM_results = pd.read_csv(PATH_SAVE)
    else:
        CPM_results = pd.DataFrame(columns = ['Correlation', 'P-value', 'MSE', 'Movie', 'Region', 'Type'])

    print('The shape of the CPM results is: ', CPM_results.shape)
    movies_done = CPM_results['Movie'].unique()
    print('The movies that CPM was trained on are: ', movies_done)

    if movie_name in movies_done:
        print('The movie was already done. We will not perform the CPM.')
        sys.exit()

    CPM_results = CPM_results.append({'Correlation': behav_pred_corr_neg[0], 'P-value': behav_pred_corr_neg[1], 'MSE': mse_neg, 'Movie': movie_name, 'Region': region, 'Type': 'Negative'}, ignore_index=True)
    CPM_results = CPM_results.append({'Correlation': behav_pred_corr_pos[0], 'P-value': behav_pred_corr_pos[1], 'MSE': mse_pos, 'Movie': movie_name, 'Region': region, 'Type': 'Positive'}, ignore_index=True)

    CPM_results.to_csv(PATH_SAVE, index=False)

    print('The shape of the CPM results after appending the new prediction output is: ', CPM_results.shape)

    