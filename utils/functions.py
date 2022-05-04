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
    '''

    # Boxplots
    
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
    df_tmp = df_tmp.dropna(axis = 1)
    
    return df_tmp


def fn_regression_result(treat_id, control_id, outcome, df):
    '''
    Conduct three different regressions: (1) with no control, (2) with age and age squared, (3) with all the control
    
    Parameters:
    treat_id (str): data_id of treatment ('LT' or 'DWT')
    control_id (str): data_id of control ('LC', 'DWC', 'PSID' or 'CPS')
    outcome (str): dependent variable ('re78' or 'dif')
    df (DataFrame): base data
    
    Returns:
    statsmodels.iolib.summary2.Summary: summarize all the results of regressions
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


def fn_regression_coef(treat_id, control_id, outcome, df):
    '''
    
    
    Parameters:
    
    Returns:
    
    '''
    
    df0 = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
    results1 = smf.ols(formula = outcome + ' ~ treat', data = df0).fit()
    df0['age2'] = df0['age'] ** 2
    results2 = smf.ols(formula = outcome + ' ~ treat + age + age2', data = df0).fit()
    results3 = smf.ols(formula = outcome + ' ~ treat + age + age2 + education + black + hispanic + nodegree', data = df0).fit()
    list_coef = [outcome, treat_id, control_id, results1.params[1], results2.params[1], results3.params[1]]
    
    return list_coef


def fn_regression_summary(df, outcome):
    '''
    
    
    Parameters:
    
    Returns:
    
    '''
    
    reg_summary = np.zeros([6, 6])
    reg_summary = pd.DataFrame(reg_summary)
    reg_summary.columns = ['Outcome', 'Treatment', 'Control', 'Without control', 'With age', 'With all control']

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

    for i in range(len(reg_summary)):
        reg_summary.iloc[i] = summary_all[i]

    return reg_summary


def fn_ML_treatment_effect(treat_id, control_id, outcome, df, method, param_grid, cv = 5, verbose = 0):
    '''
    Conduct a machine learning method (Random Forest or Gradient Boosting) to estimate the treatment effect
    
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
    float: average treatment effect estimate
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
    
    Parameters:
    df (DataFrame): base data
    method (str): ML method for estimation ('RF' or 'GB')
    param_grid (dict): grid of hyperparameters
    cv (int): number of folds in cross validation
    verbose (int): controls the verbosity
    
    Returns:
    DataFrame: all the results of ATE estimates by Random Forest and Gradient Boosting
    '''

    Dict = {'Outcome': [], 'Treatment': [], 'Control':[], 'Method': [],
            'Est_Imput': [], 'ATE': []}
    
    outcome_names = ['dif', 're78']
        
    treatment_names = ['LT', 'DWT']
    control_names_L = ['LC', 'CPS', 'PSID']
    control_names_DW = ['DWC', 'CPS', 'PSID']
    
    for outcome in tqdm(outcome_names):
        
        for treat_id in treatment_names:
        
            if treat_id == 'LT':
                for control_id in control_names_L:
                
                    ATE = fn_ML_treatment_effect(outcome = outcome, treat_id = treat_id, control_id = control_id,
                                                 df = df, method = method, param_grid = param_grid, cv = cv, verbose = verbose)

                    Dict['Outcome'] += [outcome]; Dict['Treatment'] += [treat_id]; Dict['Control'] += [control_id]
                    Dict['Method'] += ['Regression']; Dict['Est_Imput'] += [method]; Dict['ATE'] += [ATE]

            elif treat_id == 'DWT':
                for control_id in control_names_DW:

                    ATE = fn_ML_treatment_effect(outcome = outcome, treat_id = treat_id, control_id = control_id,
                                                 df = df, method = method, param_grid = param_grid, cv = cv, verbose = verbose)

                    Dict['Outcome'] += [outcome]; Dict['Treatment'] += [treat_id]; Dict['Control'] += [control_id]
                    Dict['Method'] += ['Regression']; Dict['Est_Imput'] += [method]; Dict['ATE'] += [ATE]
        
    df_results = pd.DataFrame.from_dict(Dict)
    
    return df_results


def fn_generate_variables(outcome, df):
    '''
    Generate variables
    
    Parameters:
    outcome (str): dependent variable ('re78' or 'dif')
    df (DataFrame): base data
    
    Returns:
    list of np.array: covariates, treatment dummy and outcome
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
    
    if outcome == None:
        return X, T
    else:
        Y = np.array(df[outcome], ndmin = 2).T
        return X, T, Y


def fn_propensity_score(X, T, method, param_grid = None, cv = 5):
    '''
    Estimate the propensity score by logit, RF or GB
    
    Parameters:
    X (array_like): covariates
    T (array_like): treatment variable
    method (str): estimation method for propensity score ('logit', 'RF' or 'GB')
    param_grid (dict): grid of hyperparameters
    cv (int): number of folds in cross validation
    
    Returns:
    np.array: estimates of propensity score
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


def truncate_by_prop(attribute, prop, level = 0.01):
    '''
    Remove samples that have extremely large/small propensity score
    
    Parameters:
    attribute (array_like): data or variable that will be truncated
    prop (array_like): propensity score estimates
    level (float): truncate level, i.e., samples with (level < prop < 1 - level) will be kept
    
    Returns:
    array_like: truncated data or variable
    '''
    
    keep_these = np.logical_and(prop >= level, prop <= 1.-level)

    return attribute[keep_these]


def fn_generate_df_matched(outcome, df, prop, truncate_level = 0.01, n_neighbors = 1, caliper = np.inf):
    '''
    
    
    Parameters:
    
    Returns:
    
    '''
    
    df_prop = truncate_by_prop(attribute = df, prop = prop, level = truncate_level).reset_index(drop = True)
    p_tmp = truncate_by_prop(attribute = prop, prop = prop, level = truncate_level)
    
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
    
    df_matched.data_id = pd.Categorical(df_matched.data_id, ['LT', 'DWT', 'CPS', 'PSID'])
    
    return df_matched


def fn_generate_df_matched_treat(df_matched, treat_id):
    '''
    
    
    Parameters:
    
    Returns:
    
    '''
    
    df_matched_treat = df_matched[df_matched.data_id == treat_id]

    for i in range(len(df_matched_treat)):
        matched_index = int(df_matched_treat['matched_index_1'].iloc[i])
        df_matched_row = pd.DataFrame(df_matched.iloc[matched_index,]).T
        df_matched_treat = pd.concat([df_matched_treat, df_matched_row], axis = 0)

    return df_matched_treat


def fn_mean_by_group(df_matched_treat, treat_id):
    '''
    
    
    Parameters:
    
    Returns:
    
    '''

    if treat_id == 'LT':
        df_mean_by_group = df_matched_treat.groupby('data_id').mean().T[0:11]

    if treat_id == 'DWT':
        df_mean_by_group = df_matched_treat.groupby('data_id').mean().T[0:12]
        
    return df_mean_by_group


def propensity_score_matching(outcome, df, prop, estimand, truncate_level = 0.01, n_neighbors = 1, caliper = np.inf):
    '''
    
    
    Parameters:
    
    Returns:
    
    '''
    
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


def fn_doubly_robust(treat_id, control_id, outcome, df, prop, method_mu, param_grid_mu, truncate_level = 0.01, cv = 5, verbose = 0):
    '''
    
    
    Parameters:
    
    Returns:
    
    '''
    
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


def fn_generate_df_results(df, param_grid_p, param_grid_mu, cv = 5, truncate_level = 0.01, n_neighbors = 1, caliper = np.inf, verbose = 0):
    '''
    
    
    Parameters:
    
    Returns:
    
    '''
    
    Dict = {'Outcome': [], 'Treatment': [], 'Control':[], 'Method': [],
            'Est_Prop': [], 'Est_Imput': [], 'ATE': [], 'ATET': []}

    outcome_names = ['dif', 're78']
    treatment_names = ['LT', 'DWT']
    control_names = ['CPS', 'PSID']
    prop_method_names = ['logit', 'RF', 'GB']
    imput_method_names = ['RF', 'GB']

    for treat_id in tqdm(treatment_names):
        
        for control_id in control_names:
            
            df_tmp = fn_generate_data(treat_id = treat_id, control_id = control_id, df = df)
            X, T = fn_generate_variables(outcome = None, df = df_tmp)

            for prop_method in prop_method_names:
                
                if prop_method == 'logit':
                    phat = fn_propensity_score(X, T, method = prop_method, param_grid = None, cv = cv)
                else:
                    phat = fn_propensity_score(X, T, method = prop_method, param_grid = param_grid_p, cv = cv)

                for outcome in outcome_names:

                    # Propensity score matching
                    ATET = propensity_score_matching(df = df_tmp, prop = phat, outcome = outcome,
                                                     estimand = 'ATET', n_neighbors = n_neighbors, caliper = np.inf)
                    ATE = propensity_score_matching(df = df_tmp, prop = phat, outcome = outcome,
                                                    estimand = 'ATE', n_neighbors = n_neighbors, caliper = np.inf)

                    Dict['Outcome'] += [outcome]; Dict['Treatment'] += [treat_id]; Dict['Control'] += [control_id]
                    Dict['Method'] += ['PSM']; Dict['Est_Prop'] += [prop_method]; Dict['Est_Imput'] += [None]
                    Dict['ATE'] += [ATE]; Dict['ATET'] += [ATET]

                    # IPTW estimator
                    ATE = fn_IPTW(treat_id = treat_id, control_id = control_id, df = df, outcome = outcome, prop = phat)

                    Dict['Outcome'] += [outcome]; Dict['Treatment'] += [treat_id]; Dict['Control'] += [control_id]
                    Dict['Method'] += ['IPTW']; Dict['Est_Prop'] += [prop_method]; Dict['Est_Imput'] += [None]
                    Dict['ATE'] += [ATE]; Dict['ATET'] += [None]

                    # Doubly robust estimator
                    for imput_method in imput_method_names:
                        ATE = fn_doubly_robust(treat_id = treat_id, control_id = control_id, outcome = outcome,
                                               df = df, prop = phat, method_mu = imput_method, param_grid_mu = param_grid_mu,
                                               truncate_level = truncate_level, cv = cv, verbose = verbose)

                        Dict['Outcome'] += [outcome]; Dict['Treatment'] += [treat_id]; Dict['Control'] += [control_id]
                        Dict['Method'] += ['DR']; Dict['Est_Prop'] += [prop_method]; Dict['Est_Imput'] += [imput_method]
                        Dict['ATE'] += [ATE]; Dict['ATET'] += [None]

    df_results = pd.DataFrame.from_dict(Dict)
    
    df_results.Outcome = pd.Categorical(df_results.Outcome, outcome_names)
    df_results.Treatment = pd.Categorical(df_results.Treatment, treatment_names)
    df_results.Control = pd.Categorical(df_results.Control, control_names)
    df_results.Est_Prop = pd.Categorical(df_results.Est_Prop, prop_method_names)
    df_results.Est_Imput = pd.Categorical(df_results.Est_Imput, imput_method_names)
    
    df_results.sort_values(by = ['Outcome', 'Treatment', 'Control', 'Est_Prop', 'Est_Imput']).reset_index(drop = True)
    
    return df_results


def fn_pick_results(df_results, method, outcome = None):
    '''
    
    
    Parameters:
    
    Returns:
    
    '''
    
    if outcome == None:
        df_picked = df_results[df_results.Method == method]
    
    else:
        df_picked = df_results[(df_results.Method == method) & (df_results.Outcome == outcome)]
    
    df_picked = df_picked.sort_values(by = ['Outcome', 'Treatment', 'Control', 'Est_Prop', 'Est_Imput'])
    df_picked = df_picked.dropna(axis = 'columns').reset_index(drop = True)
    
    return df_picked
    
    