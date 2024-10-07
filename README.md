# Development of Tab-AML: A Transformer-Based Model for Transaction Monitoring in Anti-Money Laundering

This repository contains the code for my PhD project: **Enhancing Transaction Monitoring Controls to Detect Money Laundering Using Machine Learning**.

## Problem Statement
Currently, many banks rely on rules-based methods for transaction monitoring, which can be highly inefficient due to the large number of false positives generated. These false positives lead to high operational costs and can strain investigative teams.

## Motivation
Machine learning models, such as XGBoost, have shown promise in detecting anomalies within transaction data. However, there is a significant lack of research exploring deep learning approaches—especially transformer-based models—for anti-money laundering (AML) transaction monitoring.

## Solution: Tab-AML
In this project, we aim to address these limitations by leveraging deep learning to improve transaction monitoring in the fight against money laundering. The development process of **Tab-AML** includes:

1. **Exploratory Data Analysis (EDA)**: Conducting a thorough EDA to better understand the data and prepare it for model training.
2. **Baseline Model Evaluation**: Experimenting with traditional machine learning models to establish baseline performance.
3. **Deep Learning Exploration**: Developing and experimenting with transformer-based models to determine their effectiveness compared to baseline methods.
4. **Development of Tab-AML**: Designing and training **Tab-AML**, a transformer-based model specifically for transaction monitoring in AML.

### Baseline Models
- **XGBoost**
- **Logistic Regression**
- **Random Forest**
- **Decision Trees**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**

### Deep Learning Models
- **TabTransformer**
- **TabNet**
- **Tab-AML** (our proposed transformer-based solution)

## Findings
Here are some visual results from our experiments:

### Model Performance
Validation and test AUC scores for different models with hyperparameter tuning:

![Validation and Test AUC Scores](Report/Screenshot%202024-10-07%20at%2013.33.17.png)

AUC curve and confusion matrix for the top two models:

![AUC Curve and Confusion Matrix](Report/Screenshot%202024-10-07%20at%2013.33.36.png)

### Exploratory Data Analysis (EDA) Diagrams
The following visualizations were created during the EDA phase to gain insights into the dataset:

- **Alerts by Location**:

  ![Alerts Location](Report/Figures/alerts_by_receiver_bank_location.png)

- **Laundering Typology Distribution**:

  ![Typology Distribution](Report/Figures/laundering_typology_distribution.png)

- **Feature Correlation Matrix**:

  ![Feature Correlation](Report/Figures/feature_correlation_matrix.png)

- **Laundering Alert Distribution**:

  ![Laundering Alert Distribution](Report/Figures/laundering_alerts_distribution.png)

- **Monthly Alerts by Payment Type**:

  ![Monthly Alerts](Report/Figures/monthly_alerts_by_payment_type.png)

## Repository Structure
- **data/**: Contains datasets used for model training and evaluation.
- **models/**: Includes saved versions of the machine learning and deep learning models.
- **src/**: Source code for data processing, model training, evaluation, and utilities.
- **Report/**: Report files, diagrams, and findings.
- **requirements.txt**: Python packages required to run the project.
- **README.md**: This documentation.

