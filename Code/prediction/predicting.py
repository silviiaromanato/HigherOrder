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

def cpm(X_train, y_train, threshold):
    # 1. INPUTS: Divide the data into training and test sets
    M, N = X_train.shape[1], X_train.shape[0] 
    all_mats, all_behav = X_train, y_train
    all_mats = np.array(all_mats)
    all_behav = np.array(all_behav)

    # Preallocate prediction arrays
    behav_pred_pos = np.zeros((N, 1))
    behav_pred_neg = np.zeros((N, 1))
    
    # 2. CROSS VALIDATION: Perform leave-one-out procedure
    number_pos = []
    number_neg = []
    for leftout in range(N):
        # print(f'Leaving out subj # {leftout + 1:02.0f}')
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

        number_pos.append(np.sum(pos_mask) / M)
        number_neg.append(np.sum(neg_mask) / M)

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

    mean_neg = np.mean(number_neg) 
    mean_pos = np.mean(number_pos)    

    return behav_pred_pos, behav_pred_neg, all_behav, mean_neg, mean_pos

def plot_cpm(behav_pred_pos, behav_pred_neg, all_behav, mean_neg, mean_pos, movie, region):

    # 8. EVALUATION: Compute correlations between predicted and observed behavior
    behav_pred_corr_pos = scipy.stats.pearsonr(all_behav, behav_pred_pos[:, 0])
    behav_pred_corr_neg = scipy.stats.pearsonr(all_behav, behav_pred_neg[:, 0])

    plt.figure(figsize=(10, 5))
    palette = sns.color_palette()
    plt.suptitle(f'CPM prediction of extraversion {region} on {movie}', fontsize=16)
    # add text at the bottom of the figure
    plt.figtext(0.90, 0.95, f'threshold = {threshold}', wrap=True, 
                horizontalalignment='center', fontsize=10, bbox=dict(facecolor='red', alpha=0.1))
    plt.subplot(1, 2, 1)
    sns.regplot(x = all_behav, y = behav_pred_pos, color = palette[0])
    plt.annotate(f'r = {behav_pred_corr_pos[0]:.3f}\np = {behav_pred_corr_pos[1]:.3f}', xy=[0.80, 0.05], xycoords='axes fraction', ha='left', va='bottom',
                bbox=dict(facecolor='grey', alpha=0.1))
    plt.annotate(f'Mean fraction significant edges = {mean_pos:.2f}', xy=[0.03, 0.93], xycoords='axes fraction', ha='left', va='bottom',
                bbox=dict(facecolor=palette[0], alpha=0.5))
    plt.xlabel('Observed behavior')
    plt.ylabel('Predicted behavior')
    plt.title('Positive edges')

    plt.subplot(1, 2, 2)
    sns.regplot(x = all_behav, y = behav_pred_neg, color=palette[1])
    plt.annotate(f'r = {behav_pred_corr_neg[0]:.3f}\np = {behav_pred_corr_neg[1]:.3f}', xy=[0.80, 0.05], xycoords='axes fraction', ha='left', va='bottom',
                bbox=dict(facecolor='grey', alpha=0.1))
    plt.annotate(f'Mean fraction significant edges = {mean_neg:.2f}', xy=[0.03, 0.93], xycoords='axes fraction', ha='left', va='bottom',
                bbox=dict(facecolor=palette[1], alpha=0.5))
    plt.xlabel('Observed behavior')
    plt.ylabel('Predicted behavior')
    plt.title('Negative edges')
    plt.tight_layout()

    plt.savefig(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/HigherOrder/Data/CPM/{movie}_{region}_{threshold}_CPM.png')

PATH_YEO = '/media/miplab-nas2/Data2/Movies_Emo/Silvia/HigherOrder/Data/yeo_RS7_Schaefer100S.mat'
columns = ['BIG5_ext', 'BIG5_agr', 'BIG5_con', 'BIG5_neu', 'BIG5_ope']

if __name__ == '__main__': 
    PATH_MOVIE = sys.argv[1]
    movie = sys.argv[2]
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

    print('\n' + ' -' * 10 + f' for {method}, {movie} and {region} FOR: ', movie_name, ' -' * 10)

    # Save the results
    PATH_SAVE = f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/prediction/'

    for threshold in [0.1, 0.05, 0.01]:    
        print(f'Computing CPM for {movie} and {region} and {method} and {threshold}')
        X = compute_X(PATH_MOVIE, movie, method, regions = region)
        behav_pred_pos, behav_pred_neg, all_behav, mean_neg, mean_pos = cpm(X, extrovercy, threshold)

        # SAVE THE RESULTS
        if os.path.exists(PATH_SAVE + f'CPM_{method}_{region}_results.csv'):
            df_results = pd.read_csv(PATH_SAVE + f'CPM_{method}_{region}_results.csv')
        else:
            df_results = pd.DataFrame(columns = ['predicted_pos', 'predicted_neg', 'observed', 'movie', 'region', 
                                                'threshold', 'mean_neg', 'mean_pos', 'r_pos', 'p_pos', 'r_neg', 
                                                'p_neg', 'added'])
        
        added = False
        if behav_pred_neg.shape[0] != 30:
            added = True
            behav_pred_neg = np.concatenate((behav_pred_neg, np.zeros((30-behav_pred_neg.shape[0], 1))))
        if behav_pred_pos.shape[0] != 30:
            added = True
            behav_pred_pos = np.concatenate((behav_pred_pos, np.zeros((30-behav_pred_pos.shape[0], 1))))
        r_pos, p_pos = scipy.stats.pearsonr(all_behav, behav_pred_pos[:, 0])
        r_neg, p_neg = scipy.stats.pearsonr(all_behav, behav_pred_neg[:, 0])
        print(f'Positive edges: r = {r_pos:.3f}, p = {p_pos:.3f}')
        print(f'Negative edges: r = {r_neg:.3f}, p = {p_neg:.3f}')
        df = pd.DataFrame({'predicted_pos': behav_pred_pos[:, 0], 'predicted_neg': behav_pred_neg[:, 0], 'observed': all_behav})
        df['movie'] = movie
        df['region'] = region
        df['threshold'] = threshold
        df['mean_neg'] = mean_neg
        df['mean_pos'] = mean_pos
        df['r_pos'] = r_pos
        df['p_pos'] = p_pos
        df['r_neg'] = r_neg
        df['p_neg'] = p_neg
        df['added'] = added

        df_results = pd.concat([df_results, df], ignore_index=True)

        # PLOT
        plot_cpm(behav_pred_pos, behav_pred_neg, all_behav, mean_neg, mean_pos, movie, region)

    