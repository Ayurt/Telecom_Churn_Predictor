Telco Customer Churn Prediction

ML Assignment – 2 (Machine Learning)

Name: Ayush Rawat
Student ID: 2025aa05400
Program: M.Tech (AI/ML) – Work Integrated Learning
Course: Machine Learning

GitHub Repository:
https://github.com/yuganshjindal/Telco_Customer_Churn.git

Live Streamlit App:
https://2025ab05174-ml-assignment2.streamlit.app/

1. Problem Statement

Customer churn is a critical challenge in the telecom industry, directly impacting revenue and customer lifetime value.
The objective of this project is to predict whether a telecom customer is likely to churn based on demographic information, service usage patterns, contract details, and billing attributes using multiple machine learning classification models.

This project demonstrates a complete end-to-end ML workflow, including data preprocessing, model training, evaluation, interactive visualization, and deployment using Streamlit.

2. Dataset Description

Dataset Name: Telco Customer Churn

Source: Kaggle

Total Records: 7,043

Total Features: 20 (after cleaning)

Target Variable: Churn (Binary: Yes / No)

Key Preprocessing Steps: 

  - Removed identifier column (customerID)

  - Converted TotalCharges from object to numeric and handled missing values

  - Encoded categorical features using One-Hot Encoding

  - Scaled numerical features for distance-based and linear models

  - Stratified train-test split to handle class imbalance

  - Fixed random_state = 25 for reproducibility and originality

3. Models Implemented

The following six classification models were implemented and evaluated on the same dataset:

  - Logistic Regression

  - Decision Tree Classifier

  - K-Nearest Neighbors (KNN)

  - Naive Bayes (Gaussian)

  - Random Forest (Ensemble)

  - XGBoost (Ensemble)

4. Model Performance Comparison

| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression | 0.8006   | 0.8314 | 0.6440    | 0.5562 | 0.5968 | 0.4675 |
| Decision Tree       | 0.7693   | 0.7909 | 0.5698    | 0.5348 | 0.5517 | 0.3970 |
| KNN                 | 0.7750   | 0.8127 | 0.5748    | 0.5856 | 0.5801 | 0.4265 |
| Naive Bayes         | 0.6792   | 0.8054 | 0.4435    | 0.8182 | 0.5752 | 0.3950 |
| Random Forest       | 0.7885   | 0.8121 | 0.6250    | 0.5080 | 0.5605 | 0.4271 |
| XGBoost             | 0.7878   | 0.8280 | 0.6206    | 0.5160 | 0.5635 | 0.4280 |

5. Observations and Analysis

| ML Model                | Observation                                                                                                                                         |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression** | Achieved the highest overall Accuracy and AUC, indicating strong baseline performance and good class separation despite its linear nature.          |
| **Decision Tree**       | Showed lower generalization performance due to overfitting tendencies and sensitivity to data splits.                                               |
| **KNN**                 | Performed moderately well after scaling, but remained sensitive to class imbalance and neighborhood selection.                                      |
| **Naive Bayes**         | Achieved very high Recall, making it effective at identifying churners, but suffered from low Precision due to its strong independence assumptions. |
| **Random Forest**       | Provided balanced performance across metrics by reducing variance through ensemble learning.                                                        |
| **XGBoost**             | Delivered strong AUC and MCC scores, demonstrating its ability to capture complex nonlinear relationships using gradient boosting.                  |


6. Streamlit Application Features

The deployed Streamlit application includes:

  - CSV upload functionality for test datasets

  - Model selection dropdown

  - Display of model evaluation metrics

  - Prediction output with churn probability

  - Confusion matrix and classification report (when ground-truth labels are provided)

  - Interactive visual insights for unlabeled test data

To ensure deployment safety and repository cleanliness, models are trained dynamically inside the Streamlit app using cached resources instead of storing large binary files.

7. Deployment Details

  - Platform: Streamlit Community Cloud

  - Branch: master

  - Main File: app.py

  - Dependencies: Managed via requirements.txt
