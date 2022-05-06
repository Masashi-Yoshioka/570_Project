# Evaluation of Job Training Program with Nonexperimental Data

Masashi Yoshioka (3200-3439-52), Jincen Jiang (6650-5721-72), Shuxian Mao (6221-2815-32)

May 6, 2022

## Description

This repository summarizes the files for our final project of ECON 570: Big Data Econometrics (Spring 2022). The deliverable is 570_Project_Yoshioka_Jiang_Mao.ipynb, which includes code and explanation. data folder contains the data that we have analyzed for the project, which are retrieved February 2, 2022 from https://github.com/jjchern/lalonde/tree/master/data. utils folder contains .py files for helper functions. results folder contains ATE and ATET estimates that we have got in the project.

## Abstract

We apply different methods of causal inferences to the dataset of LaLonde (1986), which attempts to estimate the causal effect of the job training program on earnings. In particular, LaLonde (1986) tried to replicate experimental estimates with nonexperimental data, basically resulting in failure. We try to derive reliable ATE/ATET estimates with nonexperimental data through a variety of causal inference methods. Our methods include regression and propensity score methods, using both traditional regression (linear regression, logistic regression) and machine learning approach (Random Forest, Gradient Boosting). We have found that although propensity score estimation by logistic regression and Gradient Boosting can make experimental and nonexperimental data somewhat comparable, none of our methods can replicate experimental results robustly with nonexperimental data.

