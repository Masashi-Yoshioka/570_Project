import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


##################################################
##################################################
########## Helper functions for Project ##########
##################################################
##################################################


# Choose relevant data
def fn_generate_data(treat_id, control_id, df):
    '''
    treat_id: data_id of treatment (str; 'LT' or 'DWT')
    control_id: data_id of control (str; 'LC', 'DWC', 'PSID', 'PSID2', 'PSID3', 'CPS1', 'CPS2', 'CPS3')
    '''
        
    # Select the relevant rows
    df_tmp = df.loc[df['data_id'].isin([treat_id, control_id])].reset_index(drop = True)
    df_tmp = df_tmp.dropna(axis = 1)
    
    return df_tmp


# Define X, T and Y
def fn_generate_variables(data, outcome):
    '''
    outcome: 're78' or 'dif'
    '''
    
    # Define variables
    treat = np.array(data['treat'], ndmin = 2).T
    age = np.array(data['age'], ndmin = 2).T
    education = np.array(data['education'], ndmin = 2).T
    black = np.array(data['black'], ndmin = 2).T
    hispanic = np.array(data['hispanic'], ndmin = 2).T
    married = np.array(data['married'], ndmin = 2).T
    nodegree = np.array(data['nodegree'], ndmin = 2).T
    re75 = np.array(data['re75'], ndmin = 2).T
    
    T = treat.copy()
    X = np.concatenate((age, education, black, hispanic, married, nodegree, re75), axis = 1)
    Y = np.array(data[outcome], ndmin = 2).T
        
    return X, T, Y


def truncate_by_p(attribute, p, level = 0.01):
    
    keep_these = np.logical_and(p >= level, p <= 1.-level)

    return attribute[keep_these]


def fn_generate_df_prop(df, prop, truncate_level = 0.01):
    
    df_tmp = truncate_by_p(df, prop, level = truncate_level).reset_index(drop = True)
    p_tmp = truncate_by_p(prop, prop, level = truncate_level)
    
    df_tmp['propensity_score'] = p_tmp
    df_tmp['propensity_score_logit'] = np.log(p_tmp) - np.log(1 - p_tmp)
    
    return df_tmp


def fn_generate_df_matched(df, outcome, n_neighbors = 1):
    
    knn1 = NearestNeighbors(n_neighbors = n_neighbors)
    knn0 = NearestNeighbors(n_neighbors = n_neighbors)
    
    df_treat = df[df.treat == 1]
    df_contr = df[df.treat == 0]
    
    index1 = df_treat.index.to_numpy().reshape(-1,1)
    X1 = df_treat[['propensity_score_logit']].to_numpy()
    y1 = df_treat[[outcome]].to_numpy()
    knn1.fit(X1)
    
    index0 = df_contr.index.to_numpy().reshape(-1,1)
    X0 = df_contr[['propensity_score_logit']].to_numpy()
    y0 = df_contr[[outcome]].to_numpy()
    knn0.fit(X0)
    
    matched_index = np.zeros((len(df), n_neighbors))
    matched_outcome = np.zeros((len(df), n_neighbors))
    
    for i in range(len(df)):
        
        if df.treat[i] == 0:
            where = knn1.kneighbors([[df['propensity_score_logit'][i]]],
                                    n_neighbors = n_neighbors, return_distance = False).ravel()
            index = index1[where]
            outcome = y1[where]
            
        elif df.treat[i] == 1:
            where = knn0.kneighbors([[df['propensity_score_logit'][i]]],
                                    n_neighbors = n_neighbors, return_distance = False).ravel()
            index = index0[where]
            outcome = y0[where]
        
        matched_index[i, :] = index.ravel()
        matched_outcome[i, :] = outcome.ravel()
        
    df_matched_index = pd.DataFrame(matched_index)
    df_matched_outcome = pd.DataFrame(matched_outcome)
    
    df_matched = pd.concat([df_matched_index, df_matched_outcome], axis = 1)
    
    colnames = []
    
    for i in range(n_neighbors):
        colnames += ['matched_index_' + str(i + 1)]

    for i in range(n_neighbors):
        colnames += ['matched_outcome_' + str(i + 1)]
    
    df_matched.columns = colnames
        
    df_output = pd.concat([df, df_matched], axis=1)
    
    return df_output


def fn_IPTW(df, outcome):
    '''
    Calculate IPTW estimator
    '''
    
    weight = ((df['treat'] - df['propensity_score'])/(df['propensity_score']*(1. - df['propensity_score'])))
    ATE = np.mean(weight * df[outcome])
    
    return ATE


