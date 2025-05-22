import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, make_scorer
from xgboost import XGBClassifier
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt


# Load dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)


# Preview data
#print(df.head())

# Data cleaning: Convert 'TotalCharges' to numeric, coerce errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Inspection of the missing data shows clear relationship that the TotalCharges should be = 0 because the client's tenure = 0, i.e. doesn't have a total charges yet
df.loc[df['tenure'] == 0, 'TotalCharges'] = 0

# Check for any missing values
#print(df.isnull().sum())

# Encode binary categorical columns using LabelEncoder
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

# Use a new LabelEncoder per column to avoid conflicts
for col in binary_cols:
    # print(f"Encoding column: {col}")
    if df[col].isnull().sum() > 0:
        # print(f"Warning: Column {col} has missing values. Filling with mode.")
        df[col].fillna(df[col].mode()[0], inplace=True)
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))


# Apply One-Hot Encoding with dtype=int
df = pd.get_dummies(df, columns=['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines','InternetService', 'Contract', 'PaymentMethod'], dtype=int)


# Clean column names (optional: only if needed globally)
df.columns = df.columns.str.replace(' ', '_')

# Manually encode 'Churn' (target variable): Yes -> 1, No -> 0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})


# Drop customerID as it's not a feature
df.drop('customerID', axis=1, inplace=True)


# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

#Outputting sample csv for app______________________________

# Random sample of 100 rows
X_sample = X.sample(n=1000, random_state=42)

# Save to CSV if needed
X_sample.to_csv("X_sample.csv", index=False)

sample_with_labels = X.sample(n=1000, random_state=42)
sample_with_labels["true_label"] = y.loc[sample_with_labels.index]

# Save to CSV if needed
sample_with_labels.to_csv("X_sample_with_labels.csv", index=False)

# Save to CSV if needed
X.to_csv("X_processed.csv", index=False)


# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Hyperparameter search for optimal hyperparameters_______________________________________________

# Define the XGBoost classifier
xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

# Define hyperparameter search space
param_dist = {
    'n_estimators': [300, 350, 400],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.015, 0.02, 0.025],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.5, 0.6, 0.7],
    'reg_alpha': [0.05, 0.1, 0.2],
    'reg_lambda': [1.5, 1.7, 1.9],
    'scale_pos_weight': [0.8, 1.0, 1.2]
    # 'scale_pos_weight': [1, sum(y_train == 0) / sum(y_train == 1)]  # Adjust for class imbalance
}

# Define scoring metric (ROC AUC)
scorer = make_scorer(roc_auc_score, needs_proba=True)

# Create Stratified K-Fold cross-validator
strat_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=50,          # Number of parameter settings sampled
    scoring=scorer,
    cv = strat_kfold,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit on training data
random_search.fit(X_train, y_train)

# Best parameters found
#print("Best hyperparameters:", random_search.best_params_)

# Evaluate best model on test set
best_model = random_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_test)[:,1]
#print("Test ROC AUC:", roc_auc_score(y_test, y_pred_proba))


# Predict on test data
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:,1]  # probabilities for ROC AUC

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# print(f"Accuracy: {accuracy:.4f}")
# print(f"ROC AUC: {roc_auc:.4f}")
# print("Classification Report:\n", report)
# print("Confusion Matrix:\n", conf_matrix)



#Best Simple Model Found (Accuracy = 0.82, ROC AUC = 0.86)__________________________





#Saving & Outputting model____________________________________



# Save model to file
joblib.dump(best_model, 'telco_churn_xgb_model.pkl')

# print("Model training complete and saved as 'telco_churn_xgb_model.pkl'")
