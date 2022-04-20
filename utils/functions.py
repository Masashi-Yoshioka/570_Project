import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression as lr
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

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

def Regression_result(treat_id, control_id, df):
    '''
    Conduct three different regressions: (1)with no control, (2)with age and age squared, (3)with all the control
    '''
    
    df0 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    results1 = smf.ols(formula = 'dif ~ treat', data = df0).fit()
    df0['age2'] = df0['age'] ** 2
    results2 = smf.ols(formula = 'dif ~ treat + age + age2', data = df0).fit()
    results3 = smf.ols(formula = 'dif ~ treat + age + age2 + education + black + hispanic + nodegree', data = df0).fit()
    order = ['treat','age','age2','black','education','hispanic','nodegree']
    model_names = ['Without control','With age','With all control']
    sum = summary_col([results1, results2, results3], regressor_order = order, model_names = model_names)
    return sum


def fn_RF_treatment_effect(df,treat_id,control_id,outcome):
    '''
    Conduct random forest to find the treatment effect
    '''
    
    df0 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    df0['age2'] = df0['age'] ** 2
    
    y = df0[outcome]
    x_columns = [x for x in df0.columns if x not in ['data_id','re75','re78','dif','re74']]
    X = df0[x_columns]
    
    rf = RandomForestRegressor(n_estimators = 100, oob_score=True)
    rf.fit(X,y)

    treat = X[X.treat == 1]
    control = X[X.treat == 0]

    ATE = rf.predict(treat).mean() - rf.predict(control).mean()
    
    return ATE


def fn_RF_treatment_effect_CV(df, treat_id, control_id, outcome):
    '''
    Conduct random forest to find the treatment effect
    '''
    
    df0 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    df0['age2'] = df0['age'] ** 2
    
    y = df0[outcome]
    x_columns = [x for x in df0.columns if x not in ['data_id','re75','re78','dif','re74']]
    X = df0[x_columns]
    
    param_grid_p = {'n_estimators': [50, 100, 500, 1000], 'max_features': [2, 3, 4, 5]}
    
    rfc = GridSearchCV(RandomForestRegressor(), param_grid = param_grid_p, cv = 5,
                   scoring = 'neg_mean_squared_error', return_train_score = False, verbose = 1,
                   error_score = 'raise')
    
    rfc.fit(X, y)
    
    best = rfc.best_params_

    treat = X[X.treat == 1]
    control = X[X.treat == 0]

    ATE = rfc.predict(treat).mean() - rfc.predict(control).mean()
    
    return ATE


def fn_GB_treatment_effect(df, treat_id, control_id, outcome):
    '''
    Conduct Gradient Boosting to find the treatment effect
    '''
    
    df0 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    df0['age2'] = df0['age'] ** 2
    
    y = df0[outcome]
    x_columns = [x for x in df0.columns if x not in ['data_id','re75','re78','dif','re74']]
    X = df0[x_columns]
    
    gb = GradientBoostingRegressor(n_estimators = 100, loss = 'ls')
    gb.fit(X,y)

    treat = X[X.treat == 1]
    control = X[X.treat == 0]

    ATE = gb.predict(treat).mean() - gb.predict(control).mean()
    
    return ATE


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


def propensity_score_matching(df, treat_id, control_id, method = 'Random Forest', n_neighbors = 1):
    '''
    method: 'Random Forest' or 'logit'
    outcome: 're78' or 'dif'
    n_neighbors: any number larger than 0
    '''
    
    df1 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    
    if method == 'logit':
        X, T, Y = fn_generate_variables(data = df1, outcome = 'dif')
        pipe = Pipeline([('scaler', StandardScaler()),('logistic_classifier', lr())])
        pipe.fit(X, T.ravel())
        phat = np.array(pipe.predict_proba(X)[:,1], ndmin = 2).T
    
    elif method == 'Random Forest':
        X, T, Y = fn_generate_variables(data = df1, outcome = 'dif')
        # Estimate propensity score by Random Forest
        param_grid_p = {'n_estimators': [50, 100, 500, 1000], 'max_features': [2, 3, 4, 5]}
        rfc = GridSearchCV(RandomForestClassifier(), param_grid = param_grid_p, cv = 5,
                           scoring = 'neg_mean_squared_error', return_train_score = False, verbose = 1,
                           error_score = 'raise')
        rfc.fit(X, T.ravel())
        best = rfc.best_params_
        phat = np.array(rfc.predict_proba(X)[:, 1], ndmin = 2).T
    
    # Generate a data frame with propensity score
    # The data with extremely high or low pronepsity scores are removed
    df_prop = fn_generate_df_prop(df = df1, prop = phat, truncate_level = 0.01)
    df_matched = fn_generate_df_matched(df = df_prop, outcome = 'dif', n_neighbors = n_neighbors)
    
    if n_neighbors == 1:
        mean = df_matched[df_matched.treat == 1]['dif'].mean() - df_matched[df_matched.treat == 1]['matched_outcome_1'].mean()
        return mean, df_prop
    
    else:
        colnames = []
        for i in range(1, n_neighbors+1):
            colnames += ['matched_outcome_' + str(i)]
        tauhats = df_matched[df_matched.treat == 1]['dif'] - df_matched[df_matched.treat == 1][colnames].mean(axis = 1)
        return np.mean(tauhats), df_prop
    


def fn_IPTW(df, outcome):
    '''
    Calculate IPTW estimator
    '''
    
    weight = ((df['treat'] - df['propensity_score'])/(df['propensity_score']*(1. - df['propensity_score'])))
    ATE = np.mean(weight * df[outcome])
    
    return ATE

def fn_doubly_robust(df,outcome):
    Y = df[outcome]
    x_columns = [x for x in df.columns if x in ['treat','age','education','black','hispanic','married','nodegree']]
    X = df[x_columns]

    X_treat = X[X.treat == 1]
    X_control = X[X.treat == 0]

    Y_treat = Y[X.treat == 1]
    Y_control = Y[X.treat == 0]
    
    mu0 = LinearRegression().fit(X_control,Y_control).predict(X)
    mu1 = LinearRegression().fit(X_treat,Y_treat).predict(X)
    
    return (
        np.mean(df['treat']*(Y - mu1)/df['propensity_score'] + mu1) - 
        np.mean((1-df['treat'])*(Y - mu0)/(1-df['propensity_score']) + mu0)
    )