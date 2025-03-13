# PHASE-3-PROJECT
**Churn prediction project**
This repository contains the code and resources for a churn prediction project. The goal of this project is to develop a machine learning model that can predict customer churn based on ITS historical data.

## Project Overview

Customer churn, or customer attrition, is a critical issue for many businesses. Predicting which customers are likely to churn allows companies to take proactive measures to retain them, reducing revenue loss. This project focuses on building a predictive model using machine learning techniques to identify customers at risk of churn.

## Dataset

The dataset used in this project is >>
 The publicly available Telco Customer Churn dataset from Kaggle. It contains information about customers of a telecommunications company, including demographics, service usage, and churn status.
 Key features include tenure, monthly charges, contract type, and various service subscriptions.
## Files

* `churn_prediction.ipynb`: Jupyter Notebook containing the data preprocessing, model training, and evaluation code.
* `data/`: Directory containing the dataset file.
* `requirements.txt`: List of Python libraries required to run the code.
* `README.md`: This file, providing an overview of the project.
## Dependencies

To run the code in this project, you will need the following Python libraries:

* pandas
* scikit-learn
* matplotlib
* seaborn
* numpy

## Project Structure
1. Data Loading and Exploration: The notebook begins by loading the dataset and performing exploratory data analysis (EDA) to understand the data's characteristics.
2. Data Preprocessing: The data is cleaned and preprocessed, including handling missing values, encoding categorical variables, and scaling numerical features.
3. Feature Selection/Engineering:  Features are selected or engineered to improve model performance.
4. Model Training: A machine learning model (e.g., Random Forest, Decision Tree, Logistic Regression) is trained on the preprocessed data.
5. Model Evaluation: The model's performance is evaluated using appropriate metrics (e.g., accuracy, precision, recall, F1-score,1 AUC-ROC).   
6. Feature Importance: Feature importances are analyzed to understand which features are most influential in predicting churn.
7.Model Tuning:  Hyperparameter tuning is performed to optimize model performance.

## Results
-The logistic regression model has an overall accuracy of 79.6%, meaning it correctly predicts churn or no churn for about 80% of the customers
-The decision tree model has a slightly lower accuracy of 78.8% compared to logistic regression
-The most important features for predicting churn were monthly charges and contract type.

## Future Improvements
Explore other machine learning models and algorithms.
Implement more advanced feature engineering techniques.
Deploy the model as a web application or API.
Investigate the impact of different churn mitigation strategies.

Author
Evelyn Mwangi

