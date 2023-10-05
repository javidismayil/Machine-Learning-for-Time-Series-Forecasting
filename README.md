# Machine-Learning-for-Time-Series-Forecasting
 This repository contains a Python code example for a time series forecasting and regression analysis using various machine learning algorithms. The code is designed to work with economic data from the "EVDS" dataset, and it includes the following key steps:

Data Loading and Lag Feature Engineering: The code starts by loading the economic data from an Excel file. It then generates lag features from a specific column, creating lagged versions of the data to capture temporal dependencies.

Data Preprocessing: The dataset is preprocessed, including handling missing values and creating lagged features. The target variable (y) and predictor variables (X) are defined.

Data Splitting: The dataset is split into training and testing sets using the train_test_split function from scikit-learn, allowing for model evaluation.

Support Vector Regression (SVR): The code implements SVR for regression tasks. Hyperparameter tuning is performed using grid search, and the best SVR model is selected based on cross-validation results. The Mean Squared Error (MSE) is used as the evaluation metric.

Decision Tree Regression: A Decision Tree regression model is trained and evaluated. Hyperparameter tuning is again applied to optimize the decision tree model.

Random Forest Regression: The code includes a Random Forest regression model with hyperparameter tuning using grid search. The best Random Forest model is selected based on cross-validation results.

K-Nearest Neighbors (KNN) Regression: A K-Nearest Neighbors regression model is implemented. Hyperparameter tuning is performed to find the optimal parameters for KNN.

Linear Regression: A simple Linear Regression model is applied, and hyperparameter tuning is conducted using grid search.

The code provides a comprehensive example of how to handle time series data, engineer lag features, and apply different regression algorithms for economic data analysis and prediction. It also demonstrates the importance of hyperparameter tuning for improving model performance. This repository can serve as a valuable reference for time series forecasting and regression tasks in the context of economic data analysis.
