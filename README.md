# 570_Project
This repository includes files for final project of ECON 570: Big Data Econometrics in Spring 2022.

Masashi Yoshioka (3200-3439-52), Jincen Jiang (XXXX-XXXX-XX), Shuxian Mao (6221-2815-32)

## Abstract

We apply different methods of causal inferences to the dataset of LaLonde (1986), which attempts to estimate ATE/ATET of the experimental program of job training on earnings. In particular, LaLonde (1986) tried to replicate experimental estimates using nonexperimental data, basically resulting in failure. We will try to obtain reliable ATE/ATET estimates by nonexperimental data by relying on a variety of causal inference methods. Our methods include regression and propensity score matching, using both traditional regression (linear regression, logistic regression) and machine learning methods (Random Forest, Gradient Boosting). We have found that our methods fail to replicate experimental results by nonexperimental data although propensity score estimation by logistic regression and Gradient Boosting can make experimental and nonexperimental data somewhat comparable.

