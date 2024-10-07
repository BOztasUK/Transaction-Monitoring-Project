## Development of Tab-AML : A Transformer Based Model for Transaction Monitoring in Anti-Money Laundering
This repository contains the code for the paper: 

The problem: Curently many banks are using rules based methods for transaction monitoing which is costly due to the number of false poitives thats generated. 

Motivation: Machine learning models such as xgboost have been shown to be effective in detecting anomalies in transaction data. However, there is a lack of research on using deep learning (especially transformer based models) for this task.

In this project we inicially conduct eda on the transaction data to understand the data better and to prepare the data for the training of the model. We experiment with various baseline models and deep learning models to find the best model for the task. We then devlop Tab-AML, a transformer based model for transaction monitoring.

Baseline Models: Xgboost, Logistic Regression, Random Forest, Decision Trees, KNN, and, Naive Bayes.
Deep Learning Models: TabTransformer, TabNet, and, Tab-AML.


EDA:

Findings:
AUC CURVES, CONFUSION MATRIX, FPR, AUC SCORES