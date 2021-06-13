# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains client data from a bank including client data (age, job, marital status, education...), 
information regarding last contact of the current campaign, and social and economic context. 
We are trying to predict whether the client subscribed to a term deposit

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

## Scikit-learn Pipeline
Pipeline architecture is as below:

input data -> train/test split -> classification algorithm ( Scikit Learn ) ->  Hyper Parameter Tuning ( Regularisation Strengt, and max iter are searched )
-> find best model that has max accuracy
Regularisation strength is selected from a uniform distribution between 0.0 and 1.0
Max Iter is selected from a uniform distribution between 100 and 2000, rounded to nearest integer

Random Parameter Sampling was employed which delivers results much faster than grid sampling due to limiting the search space.

Bandit policy was utilized as the early stopping policy. Runs are terminated if the accuracy is below max primary metric scaled by a slack factor.

# AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison

**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
