CODE
# **Import necessary libraries**

import numpy as np  # For numerical computations and handling arrays
import pandas as pd  # For data manipulation and analysis

# For creating and plotting the correlation matrix
import matplotlib.pyplot as plt  # For creating static, interactive, and animated visualizations
from matplotlib import gridspec  # For creating complex grid layouts for plots

import kagglehub  # For downloading datasets from Kaggle

from imblearn.over_sampling import SMOTE  # For handling imbalanced datasets by oversampling the minority class
from sklearn.model_selection import train_test_split  # For splitting datasets into training and testing sets
from sklearn.linear_model import LogisticRegression  # For applying logistic regression models
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc  # For evaluating model performance
from sklearn.preprocessing import StandardScaler  # For standardizing features by scaling them
from sklearn.metrics import precision_recall_curve  # For plotting precision-recall curves
from sklearn.metrics import precision_score, recall_score, f1_score  # For calculating precision, recall, and F1 score
from sklearn.calibration import calibration_curve  # For assessing how well predicted probabilities match actual outcomes
import joblib  # For saving and loading models to/from disk
from sklearn.ensemble import RandomForestClassifier  # For creating ensemble models using random forests
from sklearn.model_selection import learning_curve  # For evaluating model performance over different training set sizes
import seaborn as sns  # For creating attractive and informative statistical graphics

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

## **Loading the data and understanding it**

#download the dataset and import the csv file using pandas
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data_frame = pd.read_csv(f"{path}/creditcard.csv")
#print some data to analyze
print(data_frame.head)

# **Display the data frame shape and information**
Display the shape

Information

Description

#print data frame
data_frame.shape

#print dataframe info
data_frame.info()

data_frame.describe()

# **Check for any missing values**

# Cheking percent of missing values in columns
df_missing_columns = (round(((data_frame.isnull().sum()/len(data_frame.index))*100),2).to_frame('null')).sort_values('null', ascending=False)
df_missing_columns

# **Analyze the class distribution**

classes = data_frame['Class'].value_counts()
classes

valid_transactions_share =round((classes[0]/data_frame['Class'].count()*100),2)
valid_transactions_share

fraud_transactions_share = round((classes[1]/data_frame['Class'].count()*100),2)
fraud_transactions_share

# Creating dataframe for fraudulent transactions
fraudulent_transactions= data_frame[data_frame['Class'] == 1]

# Creating dataframe for non-fraudulent transactions
valid_transactions= data_frame[data_frame['Class'] == 0]


import pandas as pd

# Load the dataset (assuming it's already downloaded)

# Check for negative time values
negative_time = data_frame[data_frame['Time'] < 0]

# Print result
if negative_time.empty:
    print("No negative time values exist.")
else:
    print(negative_time)


# **Analyze the time distribution**

# Distribution plot
plt.figure(figsize=(8, 5))

# Plot for fraudulent transactions
sns.kdeplot(data=fraudulent_transactions['Time'], label='Fraudulent', fill=False)

# Plot for non-fraudulent transactions
sns.kdeplot(data=valid_transactions['Time'], label='Non-Fraudulent', fill=False)


plt.xlim(0, max(fraudulent_transactions['Time'].max(), valid_transactions['Time'].max()))

# Adding labels and title
plt.xlabel('Transaction Time (Seconds)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Transaction Time Distribution: Fraudulent vs Non-Fraudulent', fontsize=14)
plt.legend()

# Display the plot
plt.show()


# **Analyze the transaction amounts**

# Distribution plot
plt.figure(figsize=(8, 5))

# Plot for fraudulent transaction amounts
fraudulent_amount_plot = sns.kdeplot(data=fraudulent_transactions['Amount'], label='Fraudulent', fill=False)

# Plot for non-fraudulent transaction amounts
non_fraudulent_amount_plot = sns.kdeplot(data=valid_transactions['Amount'], label='Non-Fraudulent', fill=False)

# Adding labels and title
fraudulent_amount_plot.set(xlabel='Transaction Amount', ylabel='Probability Density')
plt.title('Distribution of Transaction Amounts: Fraudulent vs Non-Fraudulent')
plt.legend()

# Display the plot
plt.show()

# **Compute the correlation matrix**

# Compute the correlation matrix
# This captures the pairwise relationships between numerical features in the dataset
corr_matrix = data_frame.corr()

# Step 4: Visualize the correlation matrix as a heatmap
# The heatmap provides an intuitive view of the strength and direction of correlations
plt.figure(figsize=(12, 10))  # Set the figure size for better readability
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)  # Create heatmap
plt.title("Feature Correlation Heatmap")  # Add a descriptive title
plt.show()

# **Split into testing and training samples**

# Separate features and target/labels
X = data_frame.drop(columns=['Class'])
y = data_frame['Class']

# Split data into training and testing sets using the sklearn library
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print('Length of training smaples:',len(X_train))
print('Length of testing smaples:',len(X_test))

# Listing the columns
all_columns = X_train.columns
all_columns

import matplotlib.pyplot as plt
import seaborn as sns

plot_counter = 0
plt.figure(figsize=(17, 28))

for column in all_columns:
    plot_counter += 1
    plt.subplot(6, 5, plot_counter)
    sns.histplot(X_train[column], kde=True)
    plt.title(f"{column} Skewness: {X_train[column].skew():.2f}")

    # Remove x-axis labels and ticks
    plt.xticks([])

plt.tight_layout()  # To avoid overlapping of subplots
plt.show()


# **Make up for class imbalance using SMOTE**
**Synthetic Minority Oversampling Technique is used to create new samples for any particular class**

**Here we will use it to create more fraudulent transaction samples**

# Initial resampling using SMOTE
desired_samples_class_1 = 90000

# Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy={1: desired_samples_class_1}, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check the number of samples
print("Original training samples:", len(X_train))
print("Synthetic samples generated for class 1:", len(X_train_smote)-len(X_train))
print("Total samples in combined training dataset:", len(X_train_smote))

# **Apply Scaling Before Logistic Regression**

# Scaling the data
# Logistic regression is sensitive to feature scales, so we use StandardScaler to standardize the data
# StandardScaler scales the data so that each feature has a mean of 0 and a standard deviation of 1
scaler = StandardScaler()

# Fit the scaler on the training data and transform it to make the data standardized
X_train_smote_scaled = scaler.fit_transform(X_train_smote)

# Use the same scaler (already fit on the training data) to transform the test data.
# This ensures that the test data is scaled using the same scaling factors as the training data.
X_test_scaled = scaler.transform(X_test)

# Convert the scaled training data to a DataFrame for display
X_train_scaled_df = pd.DataFrame(X_train_smote_scaled, columns=X_train_smote.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Display the first few rows of the scaled training data
print("Scaled Training Data:")
print(X_train_scaled_df.head())

# Display the first few rows of the scaled test data
print("\nScaled Test Data:")
print(X_test_scaled_df.head())

# **Hyper Parameter Tuning**

# Creating KFold object with 5 splits
folds = KFold(n_splits=5, shuffle=True, random_state=4)

# Specify parameter values
params = {"C": [0.01, 0.1, 1, 10, 100, 1000,10000]}

#Setting the scoring metric to recall
logistic_regression_cv = GridSearchCV(estimator = LogisticRegression(),
                        param_grid = params,
                        scoring= 'roc_auc',
                        cv = folds,
                        verbose = 1,
                        return_train_score=True)

logistic_regression_cv.fit(X_train_smote_scaled, y_train_smote)

#the results
cv_results = pd.DataFrame(logistic_regression_cv.cv_results_)
cv_results

# plot of C versus train and validation scores
plt.figure(figsize=(8, 6))
plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
plt.plot(cv_results['param_C'], cv_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('roc_auc')
plt.legend(['test result', 'train result'], loc='upper left')
plt.xscale('log')

# Best score with best C
best_score =logistic_regression_cv.best_score_
best_C = logistic_regression_cv.best_params_['C']

print(" The highest test roc_auc is {0} at C = {1}".format(best_score, best_C))

# **LOGISTIC REGRESSION**

#Initialize the Logistic Regression model
# Here, we are specifying:
# `max_iter=5000`: Allow the model to run up to 5000 iterations (the model is having trouble converging with less iterations than this)
# solver='saga'`: This is an optimization algorithm, particularly good for large datasets or sparse data(so that the model converges with ease)
# `random_state=42`: Ensures reproducibility of results
log_reg = LogisticRegression(max_iter=5000, solver='saga', random_state=42,C=1000)

#Train the Logistic Regression model
# .fit() method trains the model using the scaled training data (X_train_smote_scaled)
history=log_reg.fit(X_train_smote_scaled, y_train_smote)

# Make predictions on the test data
y_pred = log_reg.predict(X_test_scaled)

# .predict_proba() method gives probabilities for each class (0 or 1).
# We're taking probabilities for the positive class (fraudulent transactions) with [:, 1]
y_pred_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

print('done')


# **Plot confusion matrix**

# Confusion Matrix
# The confusion matrix helps us evaluate the performance of our model by showing the count of the following
# True Positive (TP): correctly predicted fraud cases, True Negative (TN): correctly predicted valid transactions
# False Positive (FP): incorrectly predicted fraud cases (valid transactions predicted as fraud), False Negative (FN): incorrectly predicted valid transactions (fraud cases predicted as valid)
conf_matrix = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print("\n ===Confusion Matrix ===")
print(conf_matrix)

# Visualizing the Confusion Matrix
# using a heatmap for  a better visual representation
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu", xticklabels=['Valid', 'Fraud'], yticklabels=['Valid', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# **Analyze Class Report**

# The classification report provides a detailed evaluation of the model
# Precision: The percentage of correctly predicted fraud cases
# Recall: The percentage of correctly predicted fraud instances among all actual fraud cases
# F1-score: The harmonic mean of precision and recall
#  Support: The number of actual instances of each class in the dataset
class_report = classification_report(y_test, y_pred)
print("\n=== Classification Report ===")
print(class_report)

# **Calculate Accuracy**

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# **Receiver Operating Curve**

# Calculate ROC AUC Score
roc_auc = roc_auc_score(y_test,y_pred_prob)

from sklearn.metrics import roc_curve

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test,y_pred_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2,label="ROC Curve")
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Print AUC
print("ROC AUC Score:", roc_auc)


# **Precision Recall Curve**

# Calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# **Plot Model Metrics**
**Precision**

**Recall**

**F1 Score**

# Compute metrics at a particular threshold
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics = [precision, recall, f1]
metric_names = ['Precision', 'Recall', 'F1 Score']

plt.bar(metric_names, metrics, color=['blue', 'orange', 'green'])
plt.title("Model Evaluation Metrics")
plt.ylabel("Score")
plt.ylim([0, 1])
plt.show()

### **Random Forest**

### **Hyperparameter Tuning with GridSearchCV**

param_grid = {
    'max_depth': range(5,10,5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
    'n_estimators': [100,200,300],
    'max_features': [10, 20]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf,
                           param_grid = param_grid,
                           cv = 2,
                           n_jobs = -1,
                           verbose = 1,
                           return_train_score=True)

# Fit the model
grid_search.fit(X_train_smote, y_train_smote)

# printing the optimal accuracy score and hyperparameters
print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)

# **Training the model with optimal params**


# Import necessary libraries
import joblib  # For saving the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the Random Forest Classifier
random_forest_model = RandomForestClassifier(bootstrap=True,
                             max_depth=10,
                             min_samples_leaf=50,
                             min_samples_split=100,
                             max_features=10,
                             n_estimators=200)

# Fit the model on the training data
history_rf=random_forest_model.fit(X_train_smote, y_train_smote)

# Save the model to a file
joblib.dump(random_forest_model, 'random_forest_model.pkl')
print("Random Forest model saved as 'random_forest_model.pkl'.")


# **Analyze Class Report**

# Make predictions
y_predicted_random_forest = random_forest_model.predict(X_test)

# Evaluate the Random Forest model
print("Random Forest Classifier Performance:")
print("Accuracy Score:", accuracy_score(y_test, y_predicted_random_forest))
print("Classification Report:\n", classification_report(y_test, y_predicted_random_forest))


# **Condusion Matrix**

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_predicted_random_forest)
print("Confusion Matrix:\n", conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

accuracy = accuracy_score(y_test, y_predicted_random_forest)
print(f"Accuracy: {accuracy * 100:.2f}%")

# **Receiver Operating Curve**

# Calculate ROC AUC Score
roc_auc_rf = roc_auc_score(y_test,y_predicted_random_forest.predict_proba(X_test)[:, 1])

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)  # y_pred_prob is the predicted probabilities
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.legend(loc="lower right")
plt.show()

# **Precision Recall Curve**

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# Compute Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, random_forest_model.predict_proba(X_test)[:, 1])

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Average Precision Score
print("Average Precision Score:", average_precision_score(y_test, random_forest_model.predict_proba(X_test)[:, 1]))


# **Analyze Feature Importance**

# Get feature importances
feature_importances = random_forest_model.feature_importances_

# Sort feature importances in descending order
indices = feature_importances.argsort()[::-1]

# Plot Feature Importance
plt.figure(figsize=(12, 8))
plt.barh(range(X_train_smote.shape[1]), feature_importances[indices], align="center")
plt.yticks(range(X_train_smote.shape[1]), X_train.columns[indices])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()


# **XG Boost**

# **Tuning Hyperparams**

# hyperparameter tuning with XGBoost

# creating a KFold object
folds = 3

# specify range of hyperparameters
param_grid = {'learning_rate': [0.2, 0.6],
             'subsample': [0.3, 0.6, 0.9]}


# specify model
xgb_model = XGBClassifier(max_depth=2, n_estimators=200)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model,
                        param_grid = param_grid,
                        scoring= 'roc_auc',
                        cv = folds,
                        verbose = 1,
                        return_train_score=True)

# fit the model
model_cv.fit(X_train_smote, y_train_smote)

# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results
model_cv.best_params_

# chosen hyperparameters
# 'objective':'binary:logistic' outputs probability rather than label, which we need for calculating auc
params = {'learning_rate': 0.2,
          'max_depth': 2,
          'n_estimators':200,
          'subsample':0.9,
         'objective':'binary:logistic'}

# fit model on training data
xgb_imb_model = XGBClassifier(params = params)
xgb_imb_model.fit(X_train_smote, y_train_smote)

# Predictions on the train set
y_train_pred = xgb_imb_model.predict(X_train_smote)



# Predictions on the test set
y_test_pred = model_cv.best_estimator_.predict(X_test)
y_test_proba = model_cv.best_estimator_.predict_proba(X_test)[:, 1]  # Probability for ROC AUC


# **Confusion Matrix**

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# **Classification Report**

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_test_pred))


accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_test_proba)
print("ROC AUC Score:", roc_auc)


# **ROC Curve**

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()

# Feature Importance Plot
feature_importance = model_cv.best_estimator_.feature_importances_
sorted_idx = feature_importance.argsort()

# **Analyze Feature Importance**

plt.figure(figsize=(8, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), X_train.columns[sorted_idx])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance Plot")
plt.show()


# Save model for future use
import joblib
joblib.dump(model_cv.best_estimator_, "xgboost_model.pkl")
print("Model saved as xgboost_model.pkl")

# **Precision Recall Curve**

# Precision-Recall Curve
from sklearn.metrics import precision_recall_curve, average_precision_score
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_test_proba)
avg_precision = average_precision_score(y_test, y_test_proba)

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='purple', label=f"PR Curve (AP = {avg_precision:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# Print Average Precision Score
print("Average Precision Score (AP):", avg_precision)

# **Multilayer Perceptron MLP**
**Hyper-paramter tuning using grid search**

from sklearn.neural_network import MLPClassifier
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)
# Define the parameter grid for MLPClassifier
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50), (150, 100)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}
mlp = MLPClassifier(max_iter=300, random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=mlp,
                           param_grid=param_grid,
                           cv=3,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=2,
                           return_train_score=True)

# Fit the model with GridSearchCV
grid_search.fit(X_train_scaled, y_train_smote)

# Initialize MLPClassifier with the best parameters
mlp = MLPClassifier(
    hidden_layer_sizes=(150,),  # based on best parameters
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    alpha=0.001,
    max_iter=10000,             # Large number of iterations to ensure convergence
    random_state=42             # For reproducibility
)

# Train the model
mlp.fit(X_train_scaled, y_train_smote)

# Make predictions
y_pred = mlp.predict(X_test_scaled)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# **Confusion Matrix**

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# **ROC Curve**

# ROC Curve
y_pred_prob = mlp.predict_proba(X_test_scaled)[:, 1]  # Get probabilities for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# **Precision-Recall Curve**

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_test, mlp.predict_proba(X_test_scaled)[:, 1])

# Calculate Average Precision Score
avg_precision = average_precision_score(y_test, mlp.predict_proba(X_test_scaled)[:, 1])

# Plot the Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'AP = {avg_precision:.2f}', color='blue', linewidth=2)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve', fontsize=16)
plt.legend(loc='lower left', fontsize=12)
plt.grid(alpha=0.3)
plt.show()

