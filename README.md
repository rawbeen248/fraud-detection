# Fraud Detection Model
This repository contains a machine learning project to detect fraudulent transactions. The project is implemented in Python and involves data exploration, visualization, feature selection, model training, and evaluation.

## Contents
* `fraud_detection.ipynb`: Jupyter notebook containing the code to run the fraud detection models.
* `utils.py`: Contains helper functions used in the notebook.
* `requirements.txt`: Contains the necessary packages to run the project.

## Overview
In this project, I trained a total of six different models and evaluated their performance on the fraud detection problem. I initially trained three models: Logistic Regression, Random Forest Classifier, and XGBoost Classifier on imbalanced data. Later, I trained those models on oversampled data. All the hyperparameters were tuned separately for each model, there are six different models in total. 

## Methodology
The steps involved in the project are:
1. Import libraries.
2. Read and load data.
3. Perform Exploratory Data Analysis (EDA).
4. Visualize data.
5. Select features.
6. Split the data.
7. Perform hyperparameter tuning.
8. Fit models.
9. Evaluate models.
10. Oversample data.
11. Perform hyperparameter tuning.
12. Fit models.
13. Evaluate models.
14. Draw conclusions.

## Conclusion
The results of all the models with and without oversampling seemed to be similar for this data. Possible improvements in the process include outlier removal, different feature selection approaches, and using different techniques for handling class imbalance. 

## Instructions
To run this project, clone the repository to your machine, navigate to the project directory, install the required libraries, and run the `fraud_detection.ipynb` notebook.

* To clone the repo: ```git clone https://github.com/rawbeen248/fraud-detection```
* To install libraries: ```!pip install -r requirements.txt```

