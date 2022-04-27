import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression as lr
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from tqdm import tqdm

##################################################
########## Helper functions for Project ##########
##################################################


# Choose relevant data
def fn_generate_data(treat_id, control_id, df):
    '''
    treat_id: data_id of treatment (str; 'LT' or 'DWT')
    control_id: data_id of control (str; 'LC', 'DWC', 'PSID', 'CPS')
    '''
        
    # Select the relevant rows
    df_tmp = df.loc[df['data_id'].isin([treat_id, control_id])].reset_index(drop = True)
    df_tmp = df_tmp.dropna(axis = 1)
    
    return df_tmp


def Regression_result(treat_id, control_id, outcome, df):
    '''
    Conduct three different regressions: (1) with no control, (2) with age and age squared, (3) with all the control
    
    outcome: dependent variable (str; 're78' or 'dif')
    treat_id: data_id of treatment (str; 'LT' or 'DWT')
    control_id: data_id of control (str; 'LC', 'DWC', 'PSID', 'CPS')
    df: base data (array_like)
    '''
    
    df0 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    results1 = smf.ols(formula = outcome + ' ~ treat', data = df0).fit()
    df0['age2'] = df0['age'] ** 2
    results2 = smf.ols(formula = outcome + ' ~ treat + age + age2', data = df0).fit()
    results3 = smf.ols(formula = outcome + ' ~ treat + age + age2 + education + black + hispanic + nodegree', data = df0).fit()
    order = ['treat', 'age', 'age2', 'black', 'education', 'hispanic', 'nodegree']
    model_names = ['Without control', 'With age', 'With all control']
    summary = summary_col([results1, results2, results3], regressor_order = order, model_names = model_names, stars = True)
    
    # Show what is the dependent variable
    if outcome == 're78': print('Dependent Variable: Earnings in 1978')
    elif outcome == 'dif': print('Dependent Variable: Difference between 1975 and 1978')
    
    # Show what is the treatment group
    if treat_id == 'LT': print('Treatment: LaLonde')
    elif treat_id == 'DWT': print('Treatment: Dehejia and Wahba')
    
    # Show what is the control group
    if control_id == 'LC': print('Control: LaLonde')
    elif control_id == 'DWC': print('Control: Dehejia and Wahba')
    elif control_id == 'CPS': print('Control: CPS')
    elif control_id == 'PSID': print('Control: PSID')
    
    return summary


########################################################
##### Combined Random Forest and Gradient Boosting #####
########################################################

# def fn_RF_treatment_effect(outcome, treat_id, control_id, df, param_grid, cv = 5, verbose = 0):
#     '''
#     Conduct Random Forest to estimate the treatment effect
    
#     outcome: dependent variable (str; 're78' or 'dif')
#     treat_id: data_id of treatment (str; 'LT' or 'DWT')
#     control_id: data_id of control (str; 'LC', 'DWC', 'PSID', 'CPS')
#     df: base data (array_like)
#     param_grid: grid of hyperparameters (dict)
#     cv: number of folds in cross validation (int)
#     verbose: controls the verbosity (int)
#     '''
    
#     df0 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    
#     # Choose the relevant variables
#     y = np.array(df0[outcome]).reshape(-1, 1)
#     X = np.array(df0[['age', 'education', 'black', 'hispanic', 'married', 'nodegree']])
#     D = np.array(df0['treat']).reshape(-1, 1)
    
#     n = X.shape[0]
#     DX = np.concatenate((D, X), axis = 1)
#     D1X = np.concatenate((np.ones([n, 1]), X), axis = 1)
#     D0X = np.concatenate((np.zeros([n, 1]), X), axis = 1)
    
#     rf = GridSearchCV(RandomForestRegressor(), param_grid = param_grid, cv = cv,
#                       scoring = 'neg_mean_squared_error', verbose = verbose)
#     rf.fit(DX, y.ravel())

#     muhat1 = np.array(rf.predict(D1X), ndmin = 2).T
#     muhat0 = np.array(rf.predict(D0X), ndmin = 2).T
    
#     tauhats = muhat1 - muhat0
#     ATE = np.mean(tauhats)
    
#     return ATE


# def fn_RF_results(df, param_grid, cv = 5, verbose = 0):
#     '''
#     Return all the results of Random Forests as a dataframe
    
#     df: base data (array_like)
#     param_grid: grid of hyperparameters (dict)
#     cv: number of folds in cross validation (int)
#     verbose: controls the verbosity (int)
#     '''
    
#     outcome_names = ['dif', 're78']
    
#     results = {}
    
#     control_names_L = ['LC', 'CPS', 'PSID']
#     control_names_DW = ['DWC', 'CPS', 'PSID']
#     colnames = ['Experiment', 'CPS', 'PSID']
    
#     for i in tqdm(range(len(colnames))):
#         col = colnames[i]
#         control_L = control_names_L[i]
#         control_DW = control_names_DW[i]
#         res1 = fn_RF_treatment_effect(outcome = 'dif', treat_id = 'LT', control_id = control_L,
#                                       df = df, param_grid = param_grid, cv = cv, verbose = verbose)
#         res2 = fn_RF_treatment_effect(outcome = 're78', treat_id = 'LT', control_id = control_L,
#                                       df = df, param_grid = param_grid, cv = cv, verbose = verbose)
#         res3 = fn_RF_treatment_effect(outcome = 'dif', treat_id = 'DWT', control_id = control_DW,
#                                       df = df, param_grid = param_grid, cv = cv, verbose = verbose)
#         res4 = fn_RF_treatment_effect(outcome = 're78', treat_id = 'DWT', control_id = control_DW,
#                                       df = df, param_grid = param_grid, cv = cv, verbose = verbose)
#         results[col] = [res1, res2, res3, res4]
        
#     df_results = pd.DataFrame(results)
#     df_results.index = ['LaLonde dif', 'LaLonde re78', 'DW dif', 'DW re78']
    
#     return df_results


# def fn_GB_treatment_effect(outcome, treat_id, control_id, df, param_grid, cv = 5, verbose = 0):
#     '''
#     Conduct Gradient Boosting to estimate the treatment effect
    
#     outcome: dependent variable (str; 're78' or 'dif')
#     treat_id: data_id of treatment (str; 'LT' or 'DWT')
#     control_id: data_id of control (str; 'LC', 'DWC', 'PSID', 'CPS')
#     df: base data (array_like)
#     param_grid: grid of hyperparameters (dict)
#     cv: number of folds in cross validation (int)
#     verbose: controls the verbosity (int)
#     '''
    
#     df0 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    
#     # Choose the relevant variables
#     y = np.array(df0[outcome]).reshape(-1, 1)
#     X = np.array(df0[['age', 'education', 'black', 'hispanic', 'married', 'nodegree']])
#     D = np.array(df0['treat']).reshape(-1, 1)
    
#     n = X.shape[0]
#     DX = np.concatenate((D, X), axis = 1)
#     D1X = np.concatenate((np.ones([n, 1]), X), axis = 1)
#     D0X = np.concatenate((np.zeros([n, 1]), X), axis = 1)
    
#     gb = GridSearchCV(GradientBoostingRegressor(), param_grid = param_grid, cv = cv,
#                       scoring = 'neg_mean_squared_error', verbose = verbose)
#     gb.fit(DX, y.ravel())

#     muhat1 = np.array(gb.predict(D1X), ndmin = 2).T
#     muhat0 = np.array(gb.predict(D0X), ndmin = 2).T
    
#     tauhats = muhat1 - muhat0
#     ATE = np.mean(tauhats)
    
#     return ATE


# def fn_GB_results(df, param_grid, cv = 5, verbose = 0):
#     '''
#     Return all the results of Gradient Boosting as a dataframe
    
#     df: base data (array_like)
#     param_grid: grid of hyperparameters (dict)
#     cv: number of folds in cross validation (int)
#     verbose: controls the verbosity (int)
#     '''
    
#     outcome_names = ['dif', 're78']
    
#     results = {}
    
#     control_names_L = ['LC', 'CPS', 'PSID']
#     control_names_DW = ['DWC', 'CPS', 'PSID']
#     colnames = ['Experiment', 'CPS', 'PSID']
    
#     for i in tqdm(range(len(colnames))):
#         col = colnames[i]
#         control_L = control_names_L[i]
#         control_DW = control_names_DW[i]
#         res1 = fn_GB_treatment_effect(outcome = 'dif', treat_id = 'LT', control_id = control_L,
#                                       df = df, param_grid = param_grid, cv = cv, verbose = verbose)
#         res2 = fn_GB_treatment_effect(outcome = 're78', treat_id = 'LT', control_id = control_L,
#                                       df = df, param_grid = param_grid, cv = cv, verbose = verbose)
#         res3 = fn_GB_treatment_effect(outcome = 'dif', treat_id = 'DWT', control_id = control_DW,
#                                       df = df, param_grid = param_grid, cv = cv, verbose = verbose)
#         res4 = fn_GB_treatment_effect(outcome = 're78', treat_id = 'DWT', control_id = control_DW,
#                                       df = df, param_grid = param_grid, cv = cv, verbose = verbose)
#         results[col] = [res1, res2, res3, res4]
        
#     df_results = pd.DataFrame(results)
#     df_results.index = ['LaLonde dif', 'LaLonde re78', 'DW dif', 'DW re78']
    
#     return df_results


def fn_ML_treatment_effect(treat_id, control_id, outcome, df, method, param_grid, cv = 5, verbose = 0):
    '''
    Conduct a machine learning method (Random Forest or Gradient Boosting) to estimate the treatment effect
    
    outcome: dependent variable (str; 're78' or 'dif')
    treat_id: data_id of treatment (str; 'LT' or 'DWT')
    control_id: data_id of control (str; 'LC', 'DWC', 'PSID', 'CPS')
    df: base data (array_like)
    method: ML method for estimation (str; 'RF' or 'GB')
    param_grid: grid of hyperparameters (dict)
    cv: number of folds in cross validation (int)
    verbose: controls the verbosity (int)
    '''
    
    df0 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    
    # Choose the relevant variables
    y = np.array(df0[outcome]).reshape(-1, 1)
    X = np.array(df0[['age', 'education', 'black', 'hispanic', 'married', 'nodegree']])
    D = np.array(df0['treat']).reshape(-1, 1)
    
    n = X.shape[0]
    DX = np.concatenate((D, X), axis = 1)
    D1X = np.concatenate((np.ones([n, 1]), X), axis = 1)
    D0X = np.concatenate((np.zeros([n, 1]), X), axis = 1)
    
    if method == 'RF':
        Regressor = RandomForestRegressor()
    
    elif method == 'GB':
        Regressor = GradientBoostingRegressor()
    
    mod = GridSearchCV(Regressor, param_grid = param_grid, cv = cv,
                       scoring = 'neg_mean_squared_error', verbose = verbose)
    mod.fit(DX, y.ravel())

    muhat1 = np.array(mod.predict(D1X), ndmin = 2).T
    muhat0 = np.array(mod.predict(D0X), ndmin = 2).T
    
    tauhats = muhat1 - muhat0
    ATE = np.mean(tauhats)
    
    return ATE


def fn_ML_results(df, method, param_grid, cv = 5, verbose = 0):
    '''
    Return all the results of Random Forest or Gradient Boosting as a dataframe
    
    df: base data (array_like)
    method: ML method for estimation (str; 'RF' or 'GB')
    param_grid: grid of hyperparameters (dict)
    cv: number of folds in cross validation (int)
    verbose: controls the verbosity (int)
    '''
    
    outcome_names = ['dif', 're78']
    
    results = {}
    
    control_names_L = ['LC', 'CPS', 'PSID']
    control_names_DW = ['DWC', 'CPS', 'PSID']
    colnames = ['Experiment', 'CPS', 'PSID']
    
    for i in tqdm(range(len(colnames))):
        col = colnames[i]
        control_L = control_names_L[i]
        control_DW = control_names_DW[i]
        res1 = fn_ML_treatment_effect(outcome = 'dif', treat_id = 'LT', control_id = control_L,
                                      df = df, method = method, param_grid = param_grid, cv = cv, verbose = verbose)
        res2 = fn_ML_treatment_effect(outcome = 're78', treat_id = 'LT', control_id = control_L,
                                      df = df, method = method, param_grid = param_grid, cv = cv, verbose = verbose)
        res3 = fn_ML_treatment_effect(outcome = 'dif', treat_id = 'DWT', control_id = control_DW,
                                      df = df, method = method, param_grid = param_grid, cv = cv, verbose = verbose)
        res4 = fn_ML_treatment_effect(outcome = 're78', treat_id = 'DWT', control_id = control_DW,
                                      df = df, method = method, param_grid = param_grid, cv = cv, verbose = verbose)
        results[col] = [res1, res2, res3, res4]
        
    df_results = pd.DataFrame(results)
    df_results.index = ['LaLonde dif', 'LaLonde re78', 'DW dif', 'DW re78']
    
    return df_results


# Define X, T and Y
def fn_generate_variables(outcome, df):
    '''
    Generate variables
    
    Parameters:
    outcome: 're78' or 'dif'
    df: base data (array_like)
    
    Returns:
    Covariates, treatment dummy, and outcome as arrays
    '''
    
    # Define variables
    treat = np.array(df['treat'], ndmin = 2).T
    age = np.array(df['age'], ndmin = 2).T
    education = np.array(df['education'], ndmin = 2).T
    black = np.array(df['black'], ndmin = 2).T
    hispanic = np.array(df['hispanic'], ndmin = 2).T
    married = np.array(df['married'], ndmin = 2).T
    nodegree = np.array(df['nodegree'], ndmin = 2).T
    re75 = np.array(df['re75'], ndmin = 2).T
    
    T = treat.copy()
    X = np.concatenate((age, education, black, hispanic, married, nodegree, re75), axis = 1)
    Y = np.array(df[outcome], ndmin = 2).T
        
    return X, T, Y


def fn_propensity_score(X, T, method, param_grid = None, cv = 5):
    '''
    Estimate the propensity score by logit, RF or GB
    
    X: covariates (array_like)
    T: treatment (array_like)
    method: estimation method (str; 'logit', 'RF' or 'GB')
    param_grid: grid of hyperparameters (dict)
    cv: number of folds in cross validation (int)
    '''
    
    if method == 'logit':
        mod = Pipeline([('scaler', StandardScaler()),
                        ('logistic_classifier', lr(penalty = 'none'))])
    
    if method == 'RF':
        mod = GridSearchCV(RandomForestClassifier(), param_grid = param_grid, cv = cv,
                           scoring = 'neg_mean_squared_error', return_train_score = False)
    
    if method == 'GB':
        mod = GridSearchCV(GradientBoostingClassifier(), param_grid = param_grid, cv = cv,
                           scoring = 'neg_mean_squared_error', return_train_score = False)
    
    mod.fit(X, T.ravel())
    phat = np.array(mod.predict_proba(X)[:,1], ndmin = 2).T
    
    return phat


def truncate_by_p(attribute, p, level = 0.01):
    
    keep_these = np.logical_and(p >= level, p <= 1.-level)

    return attribute[keep_these]


###### Combined this function with fn_generate_df_matched

# def fn_generate_df_prop(df, prop, truncate_level = 0.01):
    
#     df_tmp = truncate_by_p(df, prop, level = truncate_level).reset_index(drop = True)
#     p_tmp = truncate_by_p(prop, prop, level = truncate_level)
    
#     df_tmp['propensity_score'] = p_tmp
#     df_tmp['propensity_score_logit'] = np.log(p_tmp) - np.log(1 - p_tmp)
    
#     return df_tmp


def fn_generate_df_matched(outcome, df, prop, truncate_level = 0.01, n_neighbors = 1, caliper = np.inf):
    
    df_prop = truncate_by_p(df, prop, level = truncate_level).reset_index(drop = True)
    p_tmp = truncate_by_p(prop, prop, level = truncate_level)
    
    df_prop['propensity_score'] = p_tmp
    df_prop['propensity_score_logit'] = np.log(p_tmp) - np.log(1 - p_tmp)
    
    knn1 = NearestNeighbors(n_neighbors = n_neighbors)
    knn0 = NearestNeighbors(n_neighbors = n_neighbors)

    df_treat = df_prop[df_prop.treat == 1]
    df_contr = df_prop[df_prop.treat == 0]

    index1 = df_treat.index.to_numpy().reshape(-1, 1)
    X1 = df_treat[['propensity_score_logit']].to_numpy()
    y1 = df_treat[[outcome]].to_numpy()
    knn1.fit(X1)

    index0 = df_contr.index.to_numpy().reshape(-1, 1)
    X0 = df_contr[['propensity_score_logit']].to_numpy()
    y0 = df_contr[[outcome]].to_numpy()
    knn0.fit(X0)

    matched_index = np.zeros((len(df_prop), n_neighbors))
    matched_outcome = np.zeros((len(df_prop), n_neighbors))
    distance = np.zeros((len(df_prop), n_neighbors))

    for i in range(len(df_prop)):

        if df_prop.treat[i] == 0:
            dist, where = knn1.kneighbors([[df_prop['propensity_score_logit'][i]]],
                                          n_neighbors = n_neighbors, return_distance = True)
            index = index1[where.ravel()]
            outcome = y1[where.ravel()]

        elif df_prop.treat[i] == 1:
            dist, where = knn0.kneighbors([[df_prop['propensity_score_logit'][i]]],
                                          n_neighbors = n_neighbors, return_distance = True)
            index = index0[where.ravel()]
            outcome = y0[where.ravel()]

        matched_index[i, :] = index.ravel()
        matched_outcome[i, :] = outcome.ravel()
        distance[i, :] = dist.ravel()

    df_matched_index = pd.DataFrame(matched_index)
    colnames_matched_index = ['matched_index_' + str(i + 1) for i in range(n_neighbors)]
    df_matched_index.columns = colnames_matched_index
    
    df_matched_outcome = pd.DataFrame(matched_outcome)
    colnames_matched_outcome = ['matched_outcome_' + str(i + 1) for i in range(n_neighbors)]
    df_matched_outcome.columns = colnames_matched_outcome

    df_distance = pd.DataFrame(distance)
    colnames_distance = ['distance_' + str(i + 1) for i in range(n_neighbors)]
    df_distance.columns = colnames_distance
    
    df_tmp = pd.concat([df_prop, df_matched_index, df_matched_outcome, df_distance], axis = 1)
    
    matched_average = np.zeros((len(df_tmp), 1))

    for i in range(len(df_tmp)):

        j = 0

        while j < n_neighbors:
            if df_tmp[colnames_distance[j]][i] < caliper:
                j += 1
            else: break

        if j == 0:
            matched_average[i, :] = None
        else:
            matched_average[i, :] = np.mean(df_tmp[colnames_matched_outcome[0:j]].iloc[i])

    df_matched_average = pd.DataFrame(matched_average)
    df_matched_average.columns = ['matched_outcome_average']

    df_matched = pd.concat([df_tmp, df_matched_average], axis = 1)
    
    return df_matched


def propensity_score_matching(outcome, df, prop, estimand, truncate_level = 0.01, n_neighbors = 1, caliper = np.inf):
    
    df_matched = fn_generate_df_matched(outcome = outcome, df = df, prop = prop, truncate_level = truncate_level,
                                        n_neighbors = n_neighbors, caliper = caliper)
    df_tmp = df_matched.dropna()
    
    ATET1 = np.mean(df_tmp[df_tmp.treat == 1][outcome] - df_tmp[df_tmp.treat == 1]['matched_outcome_average'])
    
    if estimand == 'ATET':
        
        return ATET1
    
    elif estimand == 'ATE':
        
        ATET0 = np.mean(df_tmp[df_tmp.treat == 0]['matched_outcome_average'] - df_tmp[df_tmp.treat == 0][outcome])
        ATE = ATET1 * np.mean(df_tmp.treat) + ATET0 * (1 - np.mean(df_tmp.treat))
        
        return ATE


# def propensity_score_matching(outcome, treat_id, control_id, df, method, param_grid, cv = 5, n_neighbors = 1):
#     '''
#     method: 'Random Forest' or 'logit'
#     outcome: 're78' or 'dif'
#     n_neighbors: any number larger than 0
#     '''
    
#     df1 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
#     X, T, Y = fn_generate_variables(outcome = outcome, df = df1)
#     phat = fn_propensity_score(X = X, T = T, method = method, param_grid = param_grid, cv = cv)
    
#     # Generate a data frame with propensity score
#     # The data with extremely high or low pronepsity scores are removed
#     df_prop = fn_generate_df_prop(df = df1, prop = phat, truncate_level = 0.01)
#     df_matched = fn_generate_df_matched(df = df_prop, outcome = outcome, n_neighbors = n_neighbors)
    
#     if n_neighbors == 1:
#         mean = df_matched[df_matched.treat == 1]['dif'].mean() - df_matched[df_matched.treat == 1]['matched_outcome_1'].mean()
#         return mean, df_prop
    
#     else:
#         colnames = []
#         for i in range(1, n_neighbors+1):
#             colnames += ['matched_outcome_' + str(i)]
#         tauhats = df_matched[df_matched.treat == 1]['dif'] - df_matched[df_matched.treat == 1][colnames].mean(axis = 1)
#         return np.mean(tauhats), df_prop
    

# def propensity_score_matching(df, treat_id, control_id, method = 'Random Forest', n_neighbors = 1):
#     '''
#     method: 'Random Forest' or 'logit'
#     outcome: 're78' or 'dif'
#     n_neighbors: any number larger than 0
#     '''
    
#     df1 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    
#     if method == 'logit':
#         X, T, Y = fn_generate_variables(outcome = 'dif', df = df1)
#         pipe = Pipeline([('scaler', StandardScaler()),
#                          ('logistic_classifier', lr(penalty = 'none'))])
#         pipe.fit(X, T.ravel())
#         phat = np.array(pipe.predict_proba(X)[:, 1], ndmin = 2).T
    
#     elif method == 'Random Forest':
#         X, T, Y = fn_generate_variables(outcome = 'dif', df = df1)
#         # Estimate propensity score by Random Forest
#         param_grid_p = {'n_estimators': [50, 100, 500, 1000], 'max_features': [2, 3, 4, 5]}
#         rfc = GridSearchCV(RandomForestClassifier(), param_grid = param_grid_p, cv = 5,
#                            scoring = 'neg_mean_squared_error', return_train_score = False, verbose = 1,
#                            error_score = 'raise')
#         rfc.fit(X, T.ravel())
#         best = rfc.best_params_
#         phat = np.array(rfc.predict_proba(X)[:, 1], ndmin = 2).T
    
#     # Generate a data frame with propensity score
#     # The data with extremely high or low pronepsity scores are removed
#     df_prop = fn_generate_df_prop(df = df1, prop = phat, truncate_level = 0.01)
#     df_matched = fn_generate_df_matched(df = df_prop, outcome = 'dif', n_neighbors = n_neighbors)
    
#     if n_neighbors == 1:
#         mean = df_matched[df_matched.treat == 1]['dif'].mean() - df_matched[df_matched.treat == 1]['matched_outcome_1'].mean()
#         return mean, df_prop
    
#     else:
#         colnames = []
#         for i in range(1, n_neighbors+1):
#             colnames += ['matched_outcome_' + str(i)]
#         tauhats = df_matched[df_matched.treat == 1]['dif'] - df_matched[df_matched.treat == 1][colnames].mean(axis = 1)
#         return np.mean(tauhats), df_prop


def fn_IPTW(treat_id, control_id, outcome, df, prop, truncate_level = 0.01):
    '''
    Calculate IPTW estimator
    '''
    df_tmp1 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    df_tmp2 = fn_generate_df_matched(outcome = outcome, df = df_tmp1, prop = prop, truncate_level = truncate_level)
    
    W = df_tmp2['treat']
    phat = df_tmp2['propensity_score']
    y = df_tmp2[outcome]
    
    weight = (W - phat)/(phat * (1. - phat))
    ATE = np.mean(weight * y)
    
    return ATE


# def fn_IPTW(df, outcome):
#     '''
#     Calculate IPTW estimator
#     '''
    
#     weight = ((df['treat'] - df['propensity_score'])/(df['propensity_score']*(1. - df['propensity_score'])))
#     ATE = np.mean(weight * df[outcome])
    
#     return ATE


def fn_doubly_robust(treat_id, control_id, outcome, df, prop, method_mu, param_grid_mu, truncate_level = 0.01, cv = 5, verbose = 0):

    df_tmp1 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    X, T, Y = fn_generate_variables(outcome = outcome, df = df_tmp1)

    n = X.shape[0]
    TX = np.concatenate((T, X), axis = 1)
    T1X = np.concatenate((np.ones([n, 1]), X), axis = 1)
    T0X = np.concatenate((np.zeros([n, 1]), X), axis = 1)

    if method_mu == 'RF':
        Regressor = RandomForestRegressor()

    elif method_mu == 'GB':
        Regressor = GradientBoostingRegressor()

    mod = GridSearchCV(Regressor, param_grid = param_grid_mu, cv = cv,
                       scoring = 'neg_mean_squared_error', verbose = verbose)
    mod.fit(TX, Y.ravel())

    muhat1 = np.array(mod.predict(T1X), ndmin = 2).T
    muhat0 = np.array(mod.predict(T0X), ndmin = 2).T

    df_muhat1 = pd.DataFrame(muhat1)
    df_muhat1.columns = ['muhat1']
    df_muhat0 = pd.DataFrame(muhat0)
    df_muhat0.columns = ['muhat0']
    
    df_tmp2 = pd.concat([df_tmp1, df_muhat1, df_muhat0], axis = 1)
    df_tmp3 = fn_generate_df_matched(df = df_tmp2, prop = prop, outcome = outcome, truncate_level = truncate_level)

    W = df_tmp3['treat']
    phat = df_tmp3['propensity_score']
    y = df_tmp3[outcome]
    mu1 = df_tmp3['muhat1']
    mu0 = df_tmp3['muhat0']

    tauhats = (W * (y - mu1)/phat + mu1) - ((1. - W) * (y - mu0)/(1. - phat) + mu0)
    np.mean(tauhats)
    
    ATE = np.mean(tauhats)
    
    return ATE


# def fn_doubly_robust(df,outcome):
#     Y = df[outcome]
#     x_columns = [x for x in df.columns if x in ['treat','age','education','black','hispanic','married','nodegree']]
#     X = df[x_columns]

#     X_treat = X[X.treat == 1]
#     X_control = X[X.treat == 0]

#     Y_treat = Y[X.treat == 1]
#     Y_control = Y[X.treat == 0]
    
#     mu0 = LinearRegression().fit(X_control,Y_control).predict(X)
#     mu1 = LinearRegression().fit(X_treat,Y_treat).predict(X)
    
#     return (
#         np.mean(df['treat']*(Y - mu1)/df['propensity_score'] + mu1) - 
#         np.mean((1-df['treat'])*(Y - mu0)/(1-df['propensity_score']) + mu0)
#     )