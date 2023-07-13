# importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score


# Function to load the data
def load_data(path):
    df = pd.read_csv(path) # Reading CSV file
    df = df.drop("Unnamed: 0", axis=1) # Removing unneccessary column
    return df # Returning dataframe


# Function to create box plots to check for outliers
def visualize_boxplots(df, shape=(5,6)):
    num_feat = df.select_dtypes(include=['int64', 'float64']).columns # Numerical features
    fig, axes = plt.subplots(shape[0], shape[1], figsize=(15, 10)) # Subplots
    axes = axes.flatten()
    for i, feature in enumerate(num_feat):
        df.boxplot(column=feature, ax=axes[i]) # Boxplot for each feature
        axes[i].set_xlabel(feature) # X-axis label
        axes[i].tick_params(axis='both', labelsize=8) # Tick parameters
    for j in range(i+1, len(axes)):
        axes[j].remove() # Removing empty subplots
    plt.tight_layout() # Layout adjustment
    plt.show()


# Function to check the class distribution
def class_difference(df, target, amount_column):
    not_fraud = round(df[target].value_counts(normalize=True)[0] * 100, 2) # Calculating not fraud percentage
    fraud = round(df[target].value_counts(normalize=True)[1] * 100, 2) # Calculating fraud percentage

    print('\nNot Fraud %:', not_fraud)
    print(df.loc[df[target] == 0, amount_column].describe().round(2))
    print('\nFraud %:', fraud)
    print(df.loc[df[target] == 1, amount_column].describe().round(2))


# Function to visualize class distribution
def visualize_class_difference(df, target):
    sns.countplot(x=target, data=df) # Count plot for class distribution
    plt.xlabel('Class') # X-axis label
    plt.ylabel('Count') # Y-axis label
    plt.title('Class Distribution') # Plot title
    plt.show()


# Function to visualize correlation heatmap
def visualize_heatmap(df, target):
    num_feat = df.drop([target], axis=1).columns # Numerical features
    plt.figure(figsize=(10, 6)) # Figure size
    sns.heatmap(df[num_feat].corr(), cmap='coolwarm', annot=False) # Heatmap
    plt.title('Correlation Heatmap') # Plot title
    plt.xticks(fontsize=8, rotation=90) # X-axis ticks
    plt.yticks(fontsize=8) # Y-axis ticks
    plt.show()


# Function to visualize scatter plots
def visualize_scatterplot(df, target, shape=(5,6)):
    num_feat = df.drop([target], axis=1).columns # Numerical features
    fig, axes = plt.subplots(shape[0], shape[1], figsize=(15, 10)) # Subplots
    axes = axes.flatten()
    for i, feature in enumerate(num_feat):
        sns.scatterplot(x=feature, y=target, data=df, ax=axes[i]) # Scatter plot for each feature
        axes[i].set_xlabel(feature) # X-axis label
        axes[i].tick_params(axis='both', labelsize=8) # Tick parameters
    for j in range(i+1, len(axes)):
        axes[j].remove() # Removing empty subplots
    plt.tight_layout() # Layout adjustment
    plt.show()


# Function to do feature selection
def feature_selection(df, target):
    log_reg = LogisticRegression() # Logistic regression model
    rfs = RFE(log_reg, n_features_to_select=None) # Recursive feature elimination
    X = df.drop(target, axis=1) # Features
    y = df[target] # Target variable
    rfs.fit(X, y) # Fitting the model
    features_selected = X.columns[rfs.support_] # Selected features
    return features_selected # Returning selected features


# Function to split the dataset
def split_dataset(df, target_var):
    features = df.drop(target_var, axis = 1).values # Features
    target = df[target_var].values # Target variable
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, stratify=target) # Splitting the data
    return X_train, X_test, y_train, y_test # Returning train and test data


# Function to print evaluation metrics
def print_evaluation_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred) # F1 score
    accuracy = accuracy_score(y_true, y_pred) # Accuracy
    precision = precision_score(y_true, y_pred) # Precision
    recall = recall_score(y_true, y_pred) # Recall
    print("\nF1 Score: ", f1) # Printing F1 score
    print("\nAccuracy Score: ", accuracy) # Printing accuracy
    print("\nPrecision Score: ", precision) # Printing precision
    print("\nRecall Score: ", recall) # Printing recall


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred) # Confusion matrix
    sns.set(font_scale=1.5) # Font size
    labels = ['0', '1'] # Labels
    sns.heatmap(conf_mat, annot=True, cmap='RdYlBu', xticklabels=labels, yticketlabels=labels) # Heatmap for confusion matrix
    plt.title('Confusion Matrix') # Plot title
    plt.xlabel('Predicted label') # X-axis label
    plt.ylabel('True label') # Y-axis label
    plt.show()


# Function to oversample the data
def oversample(X_train, y_train):
    smote = SMOTE() # SMOTE
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train) # Oversampling
    return X_train_oversampled, y_train_oversampled # Returning oversampled data


def hyperparameter_tuning_logistic(X_train, y_train):
    # Define the logistic regression model
    logic = LogisticRegression()

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'penalty': ['l1', 'l2', 'none'],
        'C': np.logspace(-4, 4, 20),
        'solver': ['newton-cg', 'liblinear', 'saga'],
        'max_iter': [100, 1000, 2500]
    }

    # Create the GridSearchCV object
    logic_reg = GridSearchCV(logic, param_grid=param_grid, cv=3, verbose=1)

    # Perform hyperparameter tuning
    best_logic_reg = logic_reg.fit(X_train, y_train)

    # Return the best estimator found during hyperparameter tuning
    return best_logic_reg.best_estimator_


def hyperparameter_tuning_rf(X_train, y_train):
    # Define the RandomForestClassifier
    random_forest = RandomForestClassifier()

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 6, 14],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(random_forest, param_grid, scoring='f1', cv=3)

    # Perform hyperparameter tuning
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Return the best hyperparameters
    return best_params


def hyperparameter_tuning_xgb(X_train, y_train):
    # Define the XGBClassifier
    xgb_model = XGBClassifier()

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [200, 500],
        'learning_rate': [0.01, 0.1],
        'min_child_weight': [3, 6],
        'colsample_bytree': [0.6, 0.8],
        'scale_pos_weight': [5, 10, 15]
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='f1', cv=3)

    # Perform hyperparameter tuning
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Return the best hyperparameters
    return best_params
