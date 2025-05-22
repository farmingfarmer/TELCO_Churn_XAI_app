# TELCO_Churn_XAI_app
Web App for Dynamic Customer Churn Predictions and XAI explanations for TELCO dataset

# 📊 Telco Customer Churn Prediction with Explainable AI (XAI)

This project demonstrates how to train an **XGBoost** model on the IBM Telco Customer Churn dataset and explain its predictions using **SHAP (SHapley Additive exPlanations)**. The goal is not only to achieve high prediction performance, but also to make the model’s decisions understandable and transparent through an interactive explanation app.

---

## 📁 Project Overview

- **Dataset:** IBM Telco Customer Churn  
- **Model:** XGBoost Classifier with hyperparameter tuning using `RandomizedSearchCV`
- **Interpretability:** SHAP for feature importance and individual prediction explanations
- **Application:** Interactive SHAP dashboard or notebook visualizations
- **Objective:** Predict whether a customer is likely to churn, and explain the why behind each prediction

---

## 🧠 Dataset Summary

The dataset comes from IBM and contains features such as:

- Customer demographics (e.g., gender, senior citizen)
- Account information (e.g., contract type, tenure)
- Service usage (e.g., internet service, streaming)
- Target: `Churn` — whether the customer left the company

---

## 🛠️ Requirements

Install all required libraries using:

pip install -r requirements.txt

 
## 🚀 How to Run
1.	Clone the repository:

git clone https://github.com/farmingfarmer/TELCO_Churn_XAI_app.git
cd TELCO_Churn_XAI_app

2.	Install dependencies:

pip install -r requirements.txt

3.	Run the main Python script to train the model:

python train_telco_churn_xgb_V5.py

This will:
o	Preprocess the dataset
o	Train and tune an XGBoost classifier
o	Save the best model to telco_churn_xgb_model.pkl


4.	Run the SHAP explanation app:
   
Open the app.py to interactively explore real time predictions and SHAP explanations.

 
## 📈 Model Performance
•	Best Accuracy: ~0.82
•	Best ROC AUC: ~0.86
•	These scores were achieved with hyperparameter tuning via 10-fold cross-validation on an XGBoost model.
 
## 🔍 Explainability with SHAP

We use SHAP to:

•	Understand global feature importance: Which features most influence churn overall?
•	Generate local explanations: Why did a specific customer get classified as “likely to churn”?

Why SHAP?

SHAP uniquely:
•	Considers feature interactions
•	Avoids independence assumptions made by simpler methods
•	Provides consistent and additive explanations
•	Helps build trust in AI systems
 
## 📂 Project Structure

### Main model training and saving script
train_telco_churn_xgb_V5.py

### python file that runs the interactive app
app.py

### Sample data for explanation app
X_sample.csv

### Processed explanatory variables used in the app for displaying the Partial Dependence Plots
X_processed.csv

### Sample data including target labels
X_sample_with_labels.csv

### Trained and saved model
telco_churn_xgb_model.pkl 

### Python dependencies
requirements.txt

### Project overview and setup instructions
README.md

### Report on the app, including screenshots of its features
Report on XAI app.pdf 

### Various png images for header images within app
churn.png                  
customerdatainput.png
no_churn.png
predictandexplainsystem.png
telcoheader.png
 
## 🧠 Credits
•	Dataset: IBM Telco Customer Churn
•	Libraries: XGBoost, SHAP, scikit-learn, matplotlib, pandas, joblib, streamlit, numpy, Pillow
 
📌 License
MIT License. Feel free to use, adapt, and share this project with attribution.


