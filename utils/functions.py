import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression as lr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


##################################################
########## Helper functions for Project ##########
##################################################


def fn_boxplots(df, re74 = True):
    '''
    Show group-by boxplots
    
    Parameters:
    df (DataFrame): data
    re74 (bool): if True, boxplot of earnings in 1974 will be drawn
    
    Returns:
    Show the boxplots of age, education, re75 (and re74) conditional on data_id
    '''

    # Include the boxplot of re74
    if re74 == True:
        fig, axes = plt.subplots(2, 2, figsize = (15, 10))

        sns.boxplot(x = 'data_id', y = 'age', data = df, ax = axes[0, 0])
        sns.boxplot(x = 'data_id', y = 'education', data = df, ax = axes[0, 1])
        sns.boxplot(x = 'data_id', y = 're74', data = df, ax = axes[1, 0])
        sns.boxplot(x = 'data_id', y = 're75', data = df, ax = axes[1, 1])

        axes[0, 0].set_title('Age')
        axes[0, 1].set_title('Education')
        axes[1, 0].set_title('Earnings in 1974')
        axes[1, 1].set_title('Earnings in 1975')
    
    # DO not include the boxplot of re74
    elif re74 == False:
        fig, axes = plt.subplots(1, 3, figsize = (15, 5))

        sns.boxplot(x = 'data_id', y = 'age', data = df, ax = axes[0])
        sns.boxplot(x = 'data_id', y = 'education', data = df, ax = axes[1])
        sns.boxplot(x = 'data_id', y = 're75', data = df, ax = axes[2])

        axes[0].set_title('Age')
        axes[1].set_title('Education')
        axes[2].set_title('Earnings in 1975')

    plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
    plt.show()


def fn_generate_data(treat_id, control_id, df):
    '''
    Generate a data frame that contains only the selected treatment and control groups
    
    Parameters:
    treat_id (str): data_id of treatment ('LT' or 'DWT')
    control_id (str): data_id of control ('LC', 'DWC', 'PSID' or 'CPS')
    df (DataFrame): base data, i.e., data that contains all the samples (defined as df in our code)
    
    Returns:
    DataFrame: DataFrame that contains only treat_id and control_id
    '''
        
    # Select the relevant rows
    df_tmp = df.loc[df['data_id'].isin([treat_id, control_id])].reset_index(drop = True)
    
    # Drop re74 when LaLonde dataset is used
    df_tmp = df_tmp.dropna(axis = 1)
    
    return df_tmp


def fn_regression_result(treat_id, control_id, outcome, df):
    '''
    Conduct three different regressions: (1) with no control, (2) with age and age squared, (3) with all the controls
    
    Parameters:
    treat_id (str): data_id of treatment ('LT' or 'DWT')
    control_id (str): data_id of control ('LC', 'DWC', 'PSID' or 'CPS')
    outcome (str): dependent variable ('re78' or 'dif')
    df (DataFrame): base data
    
    Returns:
    statsmodels.iolib.summary2.Summary: summarize all the results of regressions
    '''
    
    # Pick up the relevant samples
    df0 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    
    # Run OLS on three models
    results1 = smf.ols(formula = outcome + ' ~ treat', data = df0).fit()
    
    df0['age2'] = df0['age'] ** 2
    results2 = smf.ols(formula = outcome + ' ~ treat + age + age2', data = df0).fit()
    
    results3 = smf.ols(formula = outcome + ' ~ treat + age + age2 + education + black + hispanic + married + nodegree + re75',
                       data = df0).fit()
    
    # Summarize the results using summary_col()
    order = ['treat', 'age', 'age2', 'black', 'education', 'hispanic', 'nodegree', 're75']
    model_names = ['Without control', 'With age', 'With all controls']
    summary = summary_col([results1, results2, results3], regressor_order = order, 
                          model_names = model_names, stars = True)
    
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


def fn_regression_coef(treat_id, control_id, outcome, df):
    '''
    Summarize the OLS estimates for the coefficient on treatment dummy as a list
    Helper function for fn_regression_summary()
    
    Parameters:
    treat_id (str): data_id of treatment ('LT' or 'DWT')
    control_id (str): data_id of control ('LC', 'DWC', 'PSID' or 'CPS')
    outcome (str): dependent variable ('re78' or 'dif')
    df (DataFrame): base data
    
    Returns:
    list: summary of coefficients
    '''
    
    # Pick up the relevant samples
    df0 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    
    # Run OLS on three models
    results1 = smf.ols(formula = outcome + ' ~ treat', data = df0).fit()
    
    df0['age2'] = df0['age'] ** 2
    results2 = smf.ols(formula = outcome + ' ~ treat + age + age2', data = df0).fit()
    
    results3 = smf.ols(formula = outcome + ' ~ treat + age + age2 + education + black + hispanic + married + nodegree + re75',
                       data = df0).fit()
    
    # Summarize the estimated coefficients as a list
    list_coef = [outcome, treat_id, control_id, results1.params[1], results2.params[1], results3.params[1]]
    
    return list_coef


def fn_regression_summary(df, outcome):
    '''
    Summarize the results of linear regressions as a data frame
    
    Parameters:
    df (DataFrame): base data
    outcome (str): dependent variable ('re78' or 'dif')
    
    Returns:
    DataFrame: summary of linear regression results
    '''
    
    # Create a data frame that will contain all the results
    reg_summary = np.zeros([6, 6])
    reg_summary = pd.DataFrame(reg_summary)
    reg_summary.columns = ['Outcome', 'Treatment', 'Control', 'Without control', 'With age', 'With all controls']

    # Run OLS repeatedly over different samples
    treatment_names = ['LT', 'DWT']
    control_names_L = ['LC', 'CPS', 'PSID']
    control_names_DW = ['DWC', 'CPS', 'PSID']

    summary_all = []
    
    for treat_id in treatment_names:
        if treat_id == 'LT':
            for control_id in control_names_L:
                summary = fn_regression_coef(treat_id = treat_id, control_id = control_id, outcome = outcome, df = df)
                summary_all.append(summary)
        else:
            for control_id in control_names_DW:
                summary = fn_regression_coef(treat_id = treat_id, control_id = control_id, outcome = outcome, df = df)
                summary_all.append(summary)
                
    # Put the results into the data frame
    for i in range(len(reg_summary)):
        reg_summary.iloc[i] = summary_all[i]

    return reg_summary


def fn_ML_treatment_effect(treat_id, control_id, outcome, df, method, param_grid, cv = 5, verbose = 0):
    '''
    Implement a machine learning method (Random Forest or Gradient Boosting) to estimate ATE and ATET
    
    Parameters:
    treat_id (str): data_id of treatment ('LT' or 'DWT')
    control_id (str): data_id of control ('LC', 'DWC', 'PSID' or 'CPS')
    outcome (str): dependent variable ('re78' or 'dif')
    df (DataFrame): base data
    method (str): ML method for estimation ('RF' or 'GB')
    param_grid (dict): grid of hyperparameters
    cv (int): number of folds in cross validation
    verbose (int): controls the verbosity
    
    Returns:
    list of float: ATE and ATET estimates
    '''
    
    # Pick up the relevant samples
    df0 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    
    # Choose the relevant variables
    Y = np.array(df0[outcome]).reshape(-1, 1)
    X = np.array(df0[['age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're75']])
    D = np.array(df0['treat']).reshape(-1, 1)
    
    # Create variables for fitting and predicting
    n = X.shape[0]
    DX = np.concatenate((D, X), axis = 1)
    D1X = np.concatenate((np.ones([n, 1]), X), axis = 1)
    D0X = np.concatenate((np.zeros([n, 1]), X), axis = 1)
    
    # Run Random Forest
    if method == 'RF':
        Regressor = RandomForestRegressor()
    
    # Run Gradient Boosting
    elif method == 'GB':
        Regressor = GradientBoostingRegressor()
    
    # Tune hyperparameters by cross validation
    mod = GridSearchCV(Regressor, param_grid = param_grid, cv = cv,
                       scoring = 'neg_mean_squared_error', verbose = verbose)
    mod.fit(DX, Y.ravel())

    # Estimate conditional means muhat1 and muhat0
    muhat1 = np.array(mod.predict(D1X), ndmin = 2).T
    muhat0 = np.array(mod.predict(D0X), ndmin = 2).T
    
    # Compute ATE and ATET estimates
    mu1 = muhat1.ravel()
    mu0 = muhat0.ravel()
    y = Y.ravel(); d = D.ravel()
    
    ATET = np.mean(y[d == 1] - mu0[d == 1])
    ATEC = np.mean(mu1[d == 0] - y[d == 0])
    w = np.mean(d)
    ATE = ATET * w + ATEC * (1. - w)
    
    return ATE, ATET


def fn_ML_results(df, method, param_grid, cv = 5, verbose = 0):
    '''
    Return all the results of Random Forest or Gradient Boosting regression as a dataframe
    
    Parameters:
    df (DataFrame): base data
    method (str): ML method for estimation ('RF' or 'GB')
    param_grid (dict): grid of hyperparameters
    cv (int): number of folds in cross validation
    verbose (int): controls the verbosity
    
    Returns:
    DataFrame: all the results of ATE/ATET estimates by Random Forest or Gradient Boosting regression
    '''

    Dict = {'Outcome': [], 'Treatment': [], 'Control':[], 'Method': [],
            'Est_Imput': [], 'ATE': [], 'ATET': []}
    
    outcome_names = ['dif', 're78']
        
    treatment_names = ['LT', 'DWT']
    control_names_L = ['LC', 'CPS', 'PSID']
    control_names_DW = ['DWC', 'CPS', 'PSID']
    
    # Estimate ATE and ATET for both outcomes: dif and re78
    for outcome in tqdm(outcome_names):
        
        # Run regressions over all the samples
        for treat_id in treatment_names:
        
            if treat_id == 'LT':
                for control_id in control_names_L:
                    
                    # Run a regression
                    ATE, ATET = fn_ML_treatment_effect(outcome = outcome, treat_id = treat_id, control_id = control_id,
                                                       df = df, method = method, param_grid = param_grid, cv = cv,
                                                       verbose = verbose)
                    
                    # Store values
                    Dict['Outcome'] += [outcome]; Dict['Treatment'] += [treat_id]; Dict['Control'] += [control_id]
                    Dict['Method'] += ['Regression']; Dict['Est_Imput'] += [method]
                    Dict['ATE'] += [ATE]; Dict['ATET'] += [ATET]

            elif treat_id == 'DWT':
                for control_id in control_names_DW:

                    # Run a regression
                    ATE, ATET = fn_ML_treatment_effect(outcome = outcome, treat_id = treat_id, control_id = control_id,
                                                       df = df, method = method, param_grid = param_grid, cv = cv,
                                                       verbose = verbose)
                    
                    # Store values
                    Dict['Outcome'] += [outcome]; Dict['Treatment'] += [treat_id]; Dict['Control'] += [control_id]
                    Dict['Method'] += ['Regression']; Dict['Est_Imput'] += [method]
                    Dict['ATE'] += [ATE]; Dict['ATET'] += [ATET]
        
    # Convert the dictionary into a data frame
    df_results = pd.DataFrame.from_dict(Dict)
    
    return df_results


def fn_generate_variables(treat_id, control_id, outcome, df):
    '''
    Generate covariates, treatment dummy and outcome
    
    Parameters:
    treat_id (str): data_id of treatment ('LT' or 'DWT')
    control_id (str): data_id of control ('LC', 'DWC', 'PSID' or 'CPS')
    outcome (str): dependent variable ('re78' or 'dif') if None, Y will not be returned
    df (DataFrame): base data
    
    Returns:
    list of np.array: covariates, treatment dummy (and outcome if outcome != None)
    '''
    
    # Pick up the relevant samples
    df0 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    
    # Define variables as arrays
    treat = np.array(df0['treat'], ndmin = 2).T
    age = np.array(df0['age'], ndmin = 2).T
    education = np.array(df0['education'], ndmin = 2).T
    black = np.array(df0['black'], ndmin = 2).T
    hispanic = np.array(df0['hispanic'], ndmin = 2).T
    married = np.array(df0['married'], ndmin = 2).T
    nodegree = np.array(df0['nodegree'], ndmin = 2).T
    re75 = np.array(df0['re75'], ndmin = 2).T
    
    # Define D and X
    D = treat.copy()
    X = np.concatenate((age, education, black, hispanic, married, nodegree, re75), axis = 1)
    
    if outcome == None:
        
        return X, D
    
    else:
        
        # Define Y
        Y = np.array(df0[outcome], ndmin = 2).T
        
        return X, D, Y


def fn_propensity_score(treat_id, control_id, df, method, param_grid = None, cv = 5, verbose = 0):
    '''
    Estimate the propensity score by logit, Random Forest or Gradient Boosting
    
    Parameters:
    treat_id (str): data_id of treatment ('LT' or 'DWT')
    control_id (str): data_id of control ('LC', 'DWC', 'PSID' or 'CPS')
    df (DataFrame): base data
    method (str): estimation method for propensity score ('logit', 'RF' or 'GB')
    param_grid (dict): grid of hyperparameters
    cv (int): number of folds in cross validation
    verbose (int): controls the verbosity
    
    Returns:
    array: estimates of propensity score
    '''
    
    # Generate variables
    X, D = fn_generate_variables(treat_id = treat_id, control_id = control_id, outcome = None, df = df)
    
    # Define the model
        
    if method == 'logit':
        mod = Pipeline([('scaler', StandardScaler()),
                        ('logistic_classifier', lr(penalty = 'none'))])
    
    elif method == 'RF':
        mod = GridSearchCV(RandomForestClassifier(), param_grid = param_grid, cv = cv,
                           scoring = 'neg_log_loss', verbose = verbose)
    
    elif method == 'GB':
        mod = GridSearchCV(GradientBoostingClassifier(), param_grid = param_grid, cv = cv,
                           scoring = 'neg_log_loss', verbose = verbose)
    
    # Fit the model
    mod.fit(X, D.ravel())
    
    # Estimate the propensity score
    phat = np.array(mod.predict_proba(X)[:, 1], ndmin = 2).T
    
    return phat


def truncate_by_prop(attribute, prop, level = 0.01):
    '''
    Remove samples that have extremely large or small propensity score
    
    Parameters:
    attribute (array_like): data or variable that will be truncated
    prop (array_like): propensity score estimates
    level (float): truncate level, i.e., samples with (level < prop < 1 - level) will be kept
    
    Returns:
    array_like: truncated data or variable
    '''
    
    keep_these = np.logical_and(prop >= level, prop <= 1. - level)

    return attribute[keep_these]


def fn_generate_df_matched(treat_id, control_id, outcome, df, prop, truncate_level = 0.01, n_neighbors = 1, caliper_std = np.inf):
    '''
    Generate a data frame that contains propensity score and matched samples with each
    
    Parameters:
    treat_id (str): data_id of treatment ('LT' or 'DWT')
    control_id (str): data_id of control ('LC', 'DWC', 'PSID' or 'CPS')
    outcome (str): dependent variable ('re78' or 'dif')
    df (DataFrame): base data
    prop (array_like): propensity score estimates
    truncate_level (float): truncate level, i.e., samples with (level < prop < 1 - level) will be kept
    n_neighbors (int): number of samples that will be matched with each sample
    caliper_std (float): distance acceptable for matching measured in std of logit of propensity score
    
    Returns:
    DataFrame: contains base data, propensity score, logit of propensity score, indices of matched samples,
               outcomes of matched samples, distances from matched samples measured in logit of propensity score,
               average of matched outcomes whose distances are within the caliper
    '''
    
    # Pick up the relevant samples
    df0 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    
    # Drop samples that have an extremely high or low propensity score
    df_prop = truncate_by_prop(attribute = df0, prop = prop, level = truncate_level).reset_index(drop = True)
    p_tmp = truncate_by_prop(attribute = prop, prop = prop, level = truncate_level)
    
    # Add the propensity score and its logit to the data frame
    df_prop['propensity_score'] = p_tmp
    df_prop['propensity_score_logit'] = np.log(p_tmp) - np.log(1 - p_tmp)
    
    # Match samples using NearestNeighbors()
    knn1 = NearestNeighbors(n_neighbors = n_neighbors)
    knn0 = NearestNeighbors(n_neighbors = n_neighbors)
    
    # Divide the data frame into treatment and control groups
    df_treat = df_prop[df_prop.treat == 1]
    df_contr = df_prop[df_prop.treat == 0]

    # Fit NearestNeighbors() to treatment group based on logit of propensity score
    index1 = df_treat.index.to_numpy().reshape(-1, 1)
    X1 = df_treat[['propensity_score_logit']].to_numpy()
    y1 = df_treat[[outcome]].to_numpy()
    knn1.fit(X1)

    # Fit NearestNeighbors() to control group based on logit of propensity score
    index0 = df_contr.index.to_numpy().reshape(-1, 1)
    X0 = df_contr[['propensity_score_logit']].to_numpy()
    y0 = df_contr[[outcome]].to_numpy()
    knn0.fit(X0)

    # Arrays for storing the values
    matched_index = np.zeros((len(df_prop), n_neighbors))   # indices of matched samples
    matched_outcome = np.zeros((len(df_prop), n_neighbors)) # outcomes of matched samples
    distance = np.zeros((len(df_prop), n_neighbors))        # distances from matched samples

    for i in range(len(df_prop)):
        
        # If the sample is controlled, neighbors are chosen from the treatment group
        if df_prop.treat[i] == 0:
            dist, where = knn1.kneighbors([[df_prop['propensity_score_logit'][i]]],
                                          n_neighbors = n_neighbors, return_distance = True)
            index = index1[where.ravel()]
            outcome = y1[where.ravel()]
            
        # If the sample is treated, neighbors are chosen from the control group
        elif df_prop.treat[i] == 1:
            dist, where = knn0.kneighbors([[df_prop['propensity_score_logit'][i]]],
                                          n_neighbors = n_neighbors, return_distance = True)
            index = index0[where.ravel()]
            outcome = y0[where.ravel()]

        # Store the values
        matched_index[i, :] = index.ravel()
        matched_outcome[i, :] = outcome.ravel()
        distance[i, :] = dist.ravel()

    # Convert arrays into data frames
    df_matched_index = pd.DataFrame(matched_index)
    colnames_matched_index = ['matched_index_' + str(i + 1) for i in range(n_neighbors)]
    df_matched_index.columns = colnames_matched_index
    
    df_matched_outcome = pd.DataFrame(matched_outcome)
    colnames_matched_outcome = ['matched_outcome_' + str(i + 1) for i in range(n_neighbors)]
    df_matched_outcome.columns = colnames_matched_outcome

    df_distance = pd.DataFrame(distance)
    colnames_distance = ['distance_' + str(i + 1) for i in range(n_neighbors)]
    df_distance.columns = colnames_distance
    
    # Combine all the data frames
    df_tmp = pd.concat([df_prop, df_matched_index, df_matched_outcome, df_distance], axis = 1)
    
    ################### Compute average of matched outcomes ################### 
    
    # Define the caliper as measured in std of logit of propensity score
    caliper = caliper_std * np.std(df_tmp['propensity_score_logit'])
    
    # Array for storing the values
    matched_average = np.zeros((len(df_tmp), 1))

    for i in range(len(df_tmp)):

        j = 0
        
        # Compute how many neighbors are within the caliper
        while j < n_neighbors:
            if df_tmp[colnames_distance[j]][i] < caliper:
                j += 1
            else: break
         
        # None of the matched samples are within the caliper
        if j == 0:
            matched_average[i, :] = None
            
        # Take average of matched outcomes within the caliper
        else:
            matched_average[i, :] = np.mean(df_tmp[colnames_matched_outcome[0:j]].iloc[i])
    
    # Convert the array into a data frame
    df_matched_average = pd.DataFrame(matched_average)
    df_matched_average.columns = ['matched_outcome_average']
    
    ########################################################################### 

    # Combine the data frame
    df_matched = pd.concat([df_tmp, df_matched_average], axis = 1)
    
    # Define the categorical order for data_id
    df_matched.data_id = pd.Categorical(df_matched.data_id, ['LT', 'DWT', 'CPS', 'PSID'])
    
    return df_matched


def fn_generate_df_matched_treat(df_matched, treat_id):
    '''
    Generage a data frame that contains only the treatment group and the matched control samples 
    Only the first nearest match will be contained
    
    Parameters:
    df_matched (DataFrame): data frame generated by fn_generate_df_matched()
    treat_id (str): data_id of treatment ('LT' or 'DWT')
    
    Returns:
    DataFrame: contains only the treatment group and the matched control samples 
    '''
    
    # Treatment group
    df_matched_treat = df_matched[df_matched.data_id == treat_id]

    for i in range(len(df_matched_treat)):
        
        # Obtain the index of matched sample
        matched_index = int(df_matched_treat['matched_index_1'].iloc[i])
        
        # Obtain the row of matched sample
        df_matched_row = pd.DataFrame(df_matched.iloc[matched_index,]).T
        
        # Add the row of matched sample
        df_matched_treat = pd.concat([df_matched_treat, df_matched_row], axis = 0)

    return df_matched_treat


def fn_mean_by_group(df_matched_treat, treat_id):
    '''
    Generage a data frame that shows averages of variables within each group 

    Parameters:
    df_matched_treat (DataFrame): data frame generated by fn_generate_df_matched_treat()
    treat_id (str): data_id of treatment ('LT' or 'DWT')
    
    Returns:
    DataFrame: shows averages of variables within each group including propensity score
    '''

    # LaLonde dataset does not have re74
    if treat_id == 'LT':
        df_mean_by_group = df_matched_treat.groupby('data_id').mean().T[0:11]

    # DW subset has re74
    elif treat_id == 'DWT':
        df_mean_by_group = df_matched_treat.groupby('data_id').mean().T[0:12]
        
    return df_mean_by_group


def propensity_score_matching(treat_id, control_id, outcome, df, prop, truncate_level = 0.01,
                              n_neighbors = 1, caliper_std = np.inf):
    '''
    Estimate ATE and ATET by propensity score matching
    
    Parameters:
    treat_id (str): data_id of treatment ('LT' or 'DWT')
    control_id (str): data_id of control ('LC', 'DWC', 'PSID' or 'CPS')
    outcome (str): dependent variable ('re78' or 'dif')
    df (DataFrame): base data
    prop (array_like): propensity score estimates
    truncate_level (float): truncate level, i.e., samples with (level < prop < 1 - level) will be kept
    n_neighbors (int): number of samples that will be matched with each sample
    caliper_std (float): distance acceptable for matching measured in std of logit of propensity score
    
    Returns:
    list of float: ATE and ATET estimates
    '''
    
    # Generate a data frame form matching
    df_matched = fn_generate_df_matched(treat_id = treat_id, control_id = control_id, outcome = outcome, df = df, prop = prop,
                                        truncate_level = truncate_level, n_neighbors = n_neighbors, caliper_std = caliper_std)
    df_tmp = df_matched.dropna()
    
    # Compute ATE and ATET estimates
    ATET = np.mean(df_tmp[df_tmp.treat == 1][outcome] - df_tmp[df_tmp.treat == 1]['matched_outcome_average'])
    ATEC = np.mean(df_tmp[df_tmp.treat == 0]['matched_outcome_average'] - df_tmp[df_tmp.treat == 0][outcome])
    w = np.mean(df_tmp.treat)
    ATE = ATET * w + ATEC * (1. - w)
        
    return ATE, ATET


def fn_IPTW(treat_id, control_id, outcome, df, prop, truncate_level = 0.01):
    '''
    Compute IPTW estimator
    
    Parameters:
    treat_id (str): data_id of treatment ('LT' or 'DWT')
    control_id (str): data_id of control ('LC', 'DWC', 'PSID' or 'CPS')
    outcome (str): dependent variable ('re78' or 'dif')
    df (DataFrame): base data
    prop (array_like): propensity score estimates
    truncate_level (float): truncate level, i.e., samples with (level < prop < 1 - level) will be kept
    
    Returns:
    float: ATE estimate
    '''
    
    # Pick up the relevant samples
    df0 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    
    # Drop samples that have an extremely high or low propensity score
    df0 = truncate_by_prop(attribute = df0, prop = prop, level = truncate_level).reset_index(drop = True)
    W = np.array(df0['treat']).ravel()
    y = np.array(df0[outcome]).ravel()
    phat = truncate_by_prop(attribute = prop, prop = prop, level = truncate_level)
    phat = np.array(phat).ravel()
    
    # Compute ATE estimate
    weight = (W - phat)/(phat * (1. - phat))
    tauhats = weight * y
    ATE = np.mean(tauhats)
    
    return ATE


def fn_doubly_robust(treat_id, control_id, outcome, df, prop, method_mu, param_grid_mu,
                     truncate_level = 0.01, cv = 5, verbose = 0):
    '''
    Compute doubly robust estimator
    
    Parameters:
    treat_id (str): data_id of treatment ('LT' or 'DWT')
    control_id (str): data_id of control ('LC', 'DWC', 'PSID' or 'CPS')
    outcome (str): dependent variable ('re78' or 'dif')
    df (DataFrame): base data
    prop (array_like): propensity score estimates
    method_mu (str): ML method for estimation of mu, i.e., conditional mean of Y on D and X ('RF' or 'GB')
    param_grid_mu (dict): grid of hyperparameters for estimation of mu
    truncate_level (float): truncate level, i.e., samples with (level < prop < 1 - level) will be kept
    cv (int): number of folds in cross validation
    verbose (int): controls the verbosity    
    
    Returns:
    float: ATE estimate
    '''
    
    # Generate variables
    X, D, Y = fn_generate_variables(treat_id = treat_id, control_id = control_id, outcome = outcome, df = df)

    # Define variables for estimation of mu
    n = X.shape[0]
    DX = np.concatenate([D, X], axis = 1)
    D1X = np.concatenate([np.ones([n, 1]), X], axis = 1)
    D0X = np.concatenate([np.zeros([n, 1]), X], axis = 1)

    # Fit the model
    
    if method_mu == 'RF':
        Regressor = RandomForestRegressor()

    elif method_mu == 'GB':
        Regressor = GradientBoostingRegressor()

    mod = GridSearchCV(Regressor, param_grid = param_grid_mu, cv = cv,
                       scoring = 'neg_mean_squared_error', verbose = verbose)
    mod.fit(DX, Y.ravel())

    # Predict the conditional means
    muhat1 = np.array(mod.predict(D1X), ndmin = 2).T
    muhat0 = np.array(mod.predict(D0X), ndmin = 2).T
    
    # Drop samples that have an extremely high or low propensity score
    df0 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    df0 = truncate_by_prop(attribute = df0, prop = prop, level = truncate_level).reset_index(drop = True)
    d = np.array(df0['treat']).ravel()
    y = np.array(df0[outcome]).ravel()
    
    phat = truncate_by_prop(attribute = prop, prop = prop, level = truncate_level)
    phat = np.array(phat).ravel()
    
    mu1 = truncate_by_prop(attribute = muhat1, prop = prop, level = truncate_level)
    mu1 = np.array(mu1).ravel()
    
    mu0 = truncate_by_prop(attribute = muhat0, prop = prop, level = truncate_level)
    mu0 = np.array(mu0).ravel()

    # Compute ATE and ATET estimates
    tauhats1 = d * (y - mu1)/phat + mu1
    tauhats2 = (1. - d) * (y - mu0)/(1. - phat) + mu0
    tauhats = tauhats1 - tauhats2
    
    ATE = np.mean(tauhats)
    
    return ATE


def fn_generate_df_results(df, param_grid_p, param_grid_mu, truncate_level = 0.01, cv = 5,
                           list_n_neighbors = [1, 10], list_caliper_std = [0.1, 0.2, 1.], verbose = 0):
    '''
    Generate a data frame that contains all the ATE/ATET estimates from propensity score matching, IPTW estimator,
    and doubly robust estimator
    
    Parameters:
    df (DataFrame): base data
    param_grid_p (dict): grid of hyperparameters for propensity score estimation
    param_grid_mu (dict): grid of hyperparameters for estimation of mu
    truncate_level (float): truncate level, i.e., samples with (level < prop < 1 - level) will be kept
    cv (int): number of folds in cross validation
    list_n_neighbors (list): list of n_neighbors for propensity score matching
    list_caliper_std (list): list of caliper_std for propensity score matching
    verbose (int): controls the verbosity
    
    Returns:
    DataFrame: contains all the ATE/ATET estimates
    '''
    
    Dict = {'Outcome': [], 'Treatment': [], 'Control':[], 'Method': [],
            'Est_Prop': [], 'Est_Imput': [], 'Neighbors':[], 'Caliper_Std': [],
            'ATE': [], 'ATET': []}

    outcome_names = ['dif', 're78']
    treatment_names = ['LT', 'DWT']
    control_names = ['CPS', 'PSID']
    prop_method_names = ['logit', 'RF', 'GB'] # methods for propensity score estimation
    imput_method_names = ['RF', 'GB']         # methods for estimation of mu

    for treat_id in tqdm(treatment_names):
        
        for control_id in control_names:
            
            for prop_method in prop_method_names:
                
                # Propensity score estimation
                phat = fn_propensity_score(treat_id = treat_id, control_id = control_id, df = df,
                                           method = prop_method, param_grid = param_grid_p, cv = cv)

                for outcome in outcome_names:

                    # Propensity score matching
                    
                    for n_neighbors in list_n_neighbors:
                        
                        for caliper_std in list_caliper_std:
                    
                            ATE, ATET = propensity_score_matching(treat_id = treat_id, control_id = control_id,
                                                                  outcome = outcome, df = df, prop = phat,
                                                                  n_neighbors = n_neighbors, caliper_std = caliper_std)

                            Dict['Outcome'] += [outcome]; Dict['Treatment'] += [treat_id];
                            Dict['Control'] += [control_id]; Dict['Method'] += ['PSM'];
                            Dict['Est_Prop'] += [prop_method]; Dict['Est_Imput'] += [None]
                            Dict['Neighbors'] += [n_neighbors]; Dict['Caliper_Std'] += [caliper_std]
                            Dict['ATE'] += [ATE]; Dict['ATET'] += [ATET]

                    # IPTW estimator
                    ATE = fn_IPTW(treat_id = treat_id, control_id = control_id, df = df, outcome = outcome, prop = phat)

                    Dict['Outcome'] += [outcome]; Dict['Treatment'] += [treat_id]; Dict['Control'] += [control_id]
                    Dict['Method'] += ['IPTW']; Dict['Est_Prop'] += [prop_method]; Dict['Est_Imput'] += [None]
                    Dict['Neighbors'] += [None]; Dict['Caliper_Std'] += [None]
                    Dict['ATE'] += [ATE]; Dict['ATET'] += [None]

                    # Doubly robust estimator
                    for imput_method in imput_method_names:
                        
                        ATE = fn_doubly_robust(treat_id = treat_id, control_id = control_id, outcome = outcome,
                                               df = df, prop = phat, method_mu = imput_method, param_grid_mu = param_grid_mu,
                                               truncate_level = truncate_level, cv = cv, verbose = verbose)

                        Dict['Outcome'] += [outcome]; Dict['Treatment'] += [treat_id]; Dict['Control'] += [control_id]
                        Dict['Method'] += ['DR']; Dict['Est_Prop'] += [prop_method]; Dict['Est_Imput'] += [imput_method]
                        Dict['Neighbors'] += [None]; Dict['Caliper_Std'] += [None]
                        Dict['ATE'] += [ATE]; Dict['ATET'] += [None]

    # Convert the dictionary into a data frame
    df_results = pd.DataFrame.from_dict(Dict)
    
    # Define the categorical orders
    df_results.Outcome = pd.Categorical(df_results.Outcome, outcome_names)
    df_results.Treatment = pd.Categorical(df_results.Treatment, treatment_names)
    df_results.Control = pd.Categorical(df_results.Control, control_names)
    df_results.Est_Prop = pd.Categorical(df_results.Est_Prop, prop_method_names)
    df_results.Est_Imput = pd.Categorical(df_results.Est_Imput, imput_method_names)
    
    # Sort the data frame according to categorical orders
    df_results.sort_values(by = ['Outcome', 'Treatment', 'Control',
                                 'Est_Prop', 'Est_Imput']).reset_index(drop = True)
    
    return df_results


def fn_pick_results(df_results, method, outcome = None):
    '''
    Pick up relevant results from the data frame generated by fn_generate_df_results()
    
    Parameters:
    df_results (DataFrame): data frame generated by fn_generate_df_results()
    method (str): estimation method ('PSM', 'IPTW' or 'DR')
    outcome (str): dependent variable ('re78' or 'dif')
    
    Returns:
    DataFrame: part of the df_results
    '''
    
    # Choose the relevant data
    
    if outcome == None:
        df_picked = df_results[df_results.Method == method]
    
    else:
        df_picked = df_results[(df_results.Method == method) & (df_results.Outcome == outcome)]
    
    # Sort the data frame according to categorical orders
    df_picked = df_picked.sort_values(by = ['Outcome', 'Treatment', 'Control', 'Est_Prop', 'Est_Imput'])
    
    # Drop columns of NA
    df_picked = df_picked.dropna(axis = 'columns').reset_index(drop = True)
    
    return df_picked
    
    